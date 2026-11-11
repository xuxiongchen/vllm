# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Service authentication and finite requests, without a model executor."""

import asyncio
import json
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from vllm.entrypoints.serve.kv_snapshot.api_router import attach_router
from vllm.entrypoints.serve.middleware.authenticate import AuthenticationMiddleware

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_control_authentication_and_bounds_precede_engine_side_effects(tmp_path):
    key = tmp_path / "control-key"
    key.write_text("service-test-secret\n")
    app = FastAPI(root_path="/worker")
    calls = []

    class Engine:
        async def kv_snapshot_control(self, *args):
            calls.append(args)
            return {"status": "complete"}

    app.state.engine_client = Engine()
    attach_router(app, str(key), ["inference-test-secret"])
    app.add_middleware(AuthenticationMiddleware, tokens=["inference-test-secret"])
    body = {"operation": "capabilities", "rank": 7, "namespace": "test"}
    good = {
        "Authorization": "Bearer inference-test-secret",
        "X-KV-Snapshot-Control": "service-test-secret",
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test/worker"
    ) as client:
        for headers in [
            {},
            {"Authorization": good["Authorization"]},
            {**good, "X-KV-Snapshot-Control": "wrong"},
        ]:
            response = await client.post("/v1/kv_snapshot", json=body, headers=headers)
            assert response.status_code == 401
        assert not calls
        response = await client.post("/v1/kv_snapshot", json=body, headers=good)
        assert response.status_code == 200
        assert calls == [(7, "capabilities", "test", None, None, None)]
        for invalid in [
            {**body, "operation": "restart_stream"},
            {**body, "rank": True},
            {**body, "extra": "bad"},
            {**body, "operation": "open", "lease_id": "not-a-valid-lease" * 2},
            {**body, "operation": "open", "lease_id": uuid4().hex},
        ]:
            response = await client.post("/v1/kv_snapshot", json=invalid, headers=good)
            assert response.status_code == 400
        response = await client.post(
            "/v1/kv_snapshot",
            content=json.dumps(body) + " " * 4096,
            headers=good,
        )
        assert response.status_code == 413
        assert len(calls) == 1
    with pytest.raises(ValueError, match="distinct service credential"):
        attach_router(FastAPI(), str(key), ["service-test-secret"])


@pytest.mark.anyio
async def test_cancelled_control_open_releases_client_owned_lease(tmp_path):
    key = tmp_path / "control-key"
    key.write_text("service-test-secret")
    app = FastAPI()
    started = asyncio.Event()
    released = asyncio.Event()
    lease = uuid4().hex

    class Engine:
        async def kv_snapshot_control(
            self, rank, operation, namespace, lease_id, after, identity
        ):
            assert lease_id == lease
            if operation == "open":
                started.set()
                await asyncio.Event().wait()
            assert operation == "release"
            released.set()
            return {"status": "complete", "lease_id": lease}

    app.state.engine_client = Engine()
    attach_router(app, str(key), None)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        task = asyncio.create_task(
            client.post(
                "/v1/kv_snapshot",
                headers={"X-KV-Snapshot-Control": "service-test-secret"},
                json={
                    "operation": "open",
                    "rank": 7,
                    "namespace": "test",
                    "identity": {"rank": 7},
                    "lease_id": lease,
                },
            )
        )
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert released.is_set()


@pytest.mark.anyio
async def test_single_engine_control_cancel_and_late_reply_leave_utilities_usable():
    from vllm.v1.engine import UtilityOutput
    from vllm.v1.engine.core_client import DPLBAsyncMPClient, _process_utility_output
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, UtilityResult

    client = DPLBAsyncMPClient.__new__(DPLBAsyncMPClient)
    client.engine_ranks_managed = [3, 7]
    client.core_engine = b"\x07\x00"
    client.client_index = 0
    client.utility_results = {}
    client.encoder = MsgpackEncoder()
    sent = asyncio.Queue()

    async def send(message, engine):
        await sent.put((engine, MsgpackDecoder().decode(message[1:])))

    client._send_input_message = send
    client._ensure_output_queue_task = lambda: None
    ambiguous = await client.kv_snapshot_control_async(7, "capabilities", "test")
    assert ambiguous == {"status": "unsupported", "reason": "ambiguous_http_engine"}
    assert sent.empty()
    client.engine_ranks_managed = [7]
    task = asyncio.create_task(
        client.kv_snapshot_control_async(
            7,
            "capabilities",
            "test",
        )
    )
    engine, (_, control_id, method, _) = await asyncio.wait_for(sent.get(), 1)
    assert engine == b"\x07\x00"
    assert method == "kv_snapshot_control"
    assert control_id <= -3
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not client.utility_results
    _process_utility_output(
        UtilityOutput(control_id, result=UtilityResult({"status": "complete"})),
        client.utility_results,
    )
    normal = asyncio.create_task(
        client._call_utility_async(
            "get_supported_tasks",
            engine=client.core_engine,
        )
    )
    engine, (_, normal_id, method, _) = await asyncio.wait_for(sent.get(), 1)
    assert engine == b"\x07\x00" and normal_id >= 0
    _process_utility_output(
        UtilityOutput(normal_id, result=UtilityResult(("generate",))),
        client.utility_results,
    )
    assert await normal == ("generate",)
    assert not client.utility_results
    rejected = await client.kv_snapshot_control_async(0, "open", "test")
    assert rejected == {"status": "invalid", "reason": "rank_not_managed"}
    assert sent.empty()
    with pytest.raises(OverflowError):
        await client.kv_snapshot_control_async(
            7,
            "open",
            "test",
            identity={"invalid_integer": 2**200},
        )
    assert not client.utility_results
    assert sent.empty()
