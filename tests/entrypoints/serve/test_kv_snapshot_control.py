# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Service authentication and finite requests, without a model executor."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Event
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


@pytest.mark.parametrize(
    ("raw", "valid"),
    [
        (b"service-test-secret", True),
        (b"service-test-secret\n", True),
        (b"service-test-secret\r\n", True),
        (b" service-test-secret", False),
        (b"service-test-secret ", False),
        (b"service\ttest-secret", False),
        (b"service-test-secret\n\n", False),
        (b"service-test-secret\r", False),
        (b"service-test-secr\xc3\xa9t", False),
        (b"", False),
    ],
)
def test_control_key_file_has_unambiguous_visible_ascii(tmp_path, raw, valid):
    key = tmp_path / "control-key"
    key.write_bytes(raw)
    app = FastAPI()
    if valid:
        attach_router(app, str(key), ["inference-test-secret"])
        assert any(route.path == "/v1/kv_snapshot" for route in app.routes)
    else:
        with pytest.raises(ValueError, match="credential"):
            attach_router(app, str(key), ["inference-test-secret"])


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
async def test_single_engine_control_cancel_and_late_reply_leave_utilities_usable(
    tmp_path,
):
    import zmq

    from vllm.config import DeviceConfig, VllmConfig
    from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequestType, UtilityOutput
    from vllm.v1.engine.core_client import AsyncMPClient
    from vllm.v1.executor.abstract import Executor
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, UtilityResult

    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    addresses = {
        "input_address": f"ipc://{tmp_path}/input",
        "output_address": f"ipc://{tmp_path}/output",
    }
    sent = Queue()
    release = Event()

    def external_engine():
        with (
            zmq.Context() as context,
            context.socket(zmq.DEALER) as incoming,
            context.socket(zmq.PUSH) as output,
        ):
            incoming.setsockopt(zmq.IDENTITY, b"\x00\x00")
            incoming.setsockopt(zmq.RCVTIMEO, 1000)
            incoming.setsockopt(zmq.LINGER, 0)
            output.setsockopt(zmq.SNDTIMEO, 1000)
            output.setsockopt(zmq.LINGER, 0)
            incoming.connect(addresses["input_address"])
            output.connect(addresses["output_address"])
            incoming.send(b"")  # The supported legacy READY envelope.
            for result in ({"status": "complete"}, ("generate",)):
                message = incoming.recv_multipart()
                assert message[0] == EngineCoreRequestType.UTILITY.value
                _, call_id, method, _ = MsgpackDecoder().decode(message[1:])
                sent.put((call_id, method))
                assert release.wait(1)
                response = EngineCoreOutputs(
                    utility_output=UtilityOutput(call_id, result=UtilityResult(result))
                )
                output.send_multipart(MsgpackEncoder().encode(response))

    with ThreadPoolExecutor(max_workers=1) as executor:
        peer = executor.submit(external_engine)
        client = None
        try:
            client = AsyncMPClient(config, Executor, False, addresses)
            task = asyncio.create_task(
                client.kv_snapshot_control_async(0, "capabilities", "test")
            )
            control_id, method = await asyncio.to_thread(sent.get, timeout=1)
            assert method == "kv_snapshot_control" and control_id <= -3
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not client.utility_results
            release.set()
            assert await asyncio.wait_for(client.get_supported_tasks_async(), 1) == (
                "generate",
            )
            normal_id, method = sent.get(timeout=1)
            assert method == "get_supported_tasks" and normal_id >= 0
            assert not client.utility_results
            rejected = await client.kv_snapshot_control_async(7, "open", "test")
            assert rejected == {"status": "invalid", "reason": "rank_not_managed"}
            assert sent.empty()
            with pytest.raises(OverflowError):
                await client.kv_snapshot_control_async(
                    0, "open", "test", identity={"invalid_integer": 2**200}
                )
            assert not client.utility_results
            assert sent.empty()
            await asyncio.to_thread(peer.result, timeout=1)
        finally:
            release.set()
            if client is not None:
                client.shutdown()
