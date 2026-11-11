# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicitly enabled, service-authenticated cache metadata controls."""

import asyncio
import json
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.datastructures import Headers
from starlette.responses import JSONResponse, Response

from vllm import envs
from vllm.entrypoints.serve.middleware.authenticate import AuthenticationMiddleware

MAX_REQUEST_BYTES = 4096
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
CONTROL_TIMEOUT_SECONDS = 3


class SnapshotControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    operation: Literal["capabilities", "open", "catch_up", "live", "release"]
    rank: int = Field(ge=0, le=65535)
    namespace: str = Field(min_length=1, max_length=256)
    lease_id: str | None = Field(default=None, min_length=32, max_length=36)
    after: int | None = Field(default=None, ge=-1, le=2**63 - 1)
    identity: dict[str, Any] | None = None


def attach_router(app: FastAPI, key_file: str, api_keys: list[str] | None) -> None:
    if envs.VLLM_DEBUG_LOG_API_SERVER_RESPONSE:
        raise ValueError("snapshot control cannot enable response payload logging")
    with Path(key_file).open("rb") as stream:
        raw_key = stream.read(MAX_REQUEST_BYTES + 1)
    if len(raw_key) > MAX_REQUEST_BYTES:
        raise ValueError("snapshot control credential file is too large")
    if raw_key.endswith(b"\r\n"):
        raw_key = raw_key[:-2]
    elif raw_key.endswith(b"\n"):
        raw_key = raw_key[:-1]
    if not raw_key or any(byte < 0x21 or byte > 0x7E for byte in raw_key):
        raise ValueError("snapshot control credential must be visible ASCII")
    key = raw_key.decode("ascii")
    if key in (api_keys or []):
        raise ValueError("snapshot control requires a distinct service credential")
    auth = AuthenticationMiddleware(app, [key])
    permits = asyncio.Semaphore(2)

    @app.post("/v1/kv_snapshot", include_in_schema=False)
    async def control(request: Request) -> Response:
        # Do this before reading a body or creating a remote owner session. The
        # ordinary /v1 Bearer middleware additionally applies when configured.
        if not auth.verify_token(
            Headers(
                {
                    "Authorization": "Bearer "
                    + request.headers.get("X-KV-Snapshot-Control", "")
                }
            )
        ):
            return JSONResponse({"status": "invalid", "reason": "unauthorized"}, 401)
        if permits.locked():
            return JSONResponse(
                {"status": "capacity", "reason": "control_requests_full"}, 503
            )
        command = None
        open_dispatched = False
        reply_ready = False
        engine = request.app.state.engine_client
        async with permits:
            try:
                async with asyncio.timeout(CONTROL_TIMEOUT_SECONDS):
                    body = bytearray()
                    async for chunk in request.stream():
                        if len(body) + len(chunk) > MAX_REQUEST_BYTES:
                            return JSONResponse(
                                {"status": "invalid", "reason": "request_bytes_full"},
                                413,
                            )
                        body.extend(chunk)
                    command = SnapshotControlRequest.model_validate_json(body)
                    if command.operation != "capabilities":
                        if command.identity is None or command.lease_id is None:
                            raise ValueError("missing identity or lease")
                        UUID(command.lease_id)
                    if (command.operation == "live") != (command.after is not None):
                        raise ValueError("invalid cursor")
                    open_dispatched = command.operation == "open"
                    result = await engine.kv_snapshot_control(
                        command.rank,
                        command.operation,
                        command.namespace,
                        command.lease_id,
                        command.after,
                        command.identity,
                    )
                    encoded = await asyncio.to_thread(
                        lambda: json.dumps(result, separators=(",", ":")).encode()
                    )
                    if len(encoded) > MAX_RESPONSE_BYTES:
                        raise ValueError("response_bytes_full")
                    reply_ready = True
                    return Response(encoded, media_type="application/json")
            except (ValidationError, ValueError):
                return JSONResponse(
                    {"status": "invalid", "reason": "invalid_control_request"}, 400
                )
            except TimeoutError:
                return JSONResponse(
                    {"status": "unavailable", "reason": "control_timeout"}, 504
                )
            except Exception:
                return JSONResponse(
                    {"status": "unavailable", "reason": "control_failed"}, 503
                )
            finally:
                # Open IDs belong to the client, so a lost response can also be
                # released by it. Release a locally failed/cancelled open here.
                if command is not None and open_dispatched and not reply_ready:
                    # A disconnected owner still has a finite lease.
                    with suppress(Exception, asyncio.CancelledError):
                        await asyncio.wait_for(
                            engine.kv_snapshot_control(
                                command.rank,
                                "release",
                                command.namespace,
                                command.lease_id,
                                None,
                                command.identity,
                            ),
                            timeout=1,
                        )
