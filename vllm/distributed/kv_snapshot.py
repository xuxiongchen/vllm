# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, cache-owner-serialized metadata snapshots (candidate schema).

This journal is not the legacy ZMQ stream. Its control replies carry their own
incarnations and immutable payloads; seq zero never establishes completeness.
All operations run on the same EngineCore thread as the attached BlockPool.
"""

import hashlib
import json
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import msgspec

from vllm.distributed.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
    KVEventBatch,
)
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    get_block_hash,
    get_group_id,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
)

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.request import Request


class SnapshotError(Exception):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


@dataclass(frozen=True)
class SnapshotLimits:
    max_entries: int = 16_384
    max_catalogue_bytes: int = 4 * 1024 * 1024
    max_journal_batches: int = 1_024
    max_journal_bytes: int = 4 * 1024 * 1024
    max_sessions: int = 2
    max_snapshot_bytes: int = 8 * 1024 * 1024
    lease_seconds: float = 30.0


@dataclass
class _Session:
    watermark: int
    expires: float
    high: int | None = None
    delivered: int = -1
    acknowledged: int = -1


class KvSnapshotCatalogue:
    """Bounded descriptions and residency rooted in one real cache lifecycle.

    Namespace is an explicit deployment cache-identity binding, not a model
    name guess. The scheduler additionally checks the actual layout/profile.
    """

    def __init__(
        self,
        pool: "BlockPool",
        namespace: str,
        source_id: str,
        rank: int,
        block_size: int,
        *,
        limits: SnapshotLimits | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if limits is None:
            limits = SnapshotLimits()
        if not namespace or not source_id or rank < 0 or block_size <= 0:
            raise ValueError("snapshot identity/profile must be explicit")
        if any(
            value <= 0
            for value in (
                limits.max_entries,
                limits.max_catalogue_bytes,
                limits.max_journal_batches,
                limits.max_journal_bytes,
                limits.max_sessions,
                limits.max_snapshot_bytes,
                limits.lease_seconds,
            )
        ):
            raise ValueError("snapshot limits must be positive")
        self.pool = pool
        self.limits = limits
        self.clock = clock
        self._thread = threading.get_ident()
        self.identity = {
            "cache_id": uuid.uuid4().hex,
            "stream_id": uuid.uuid4().hex,
            "source_id": source_id,
            "rank": rank,
            "namespace": namespace,
            "profile": "local-full-attention-v1",
            "block_size": block_size,
            "lookahead": 0,
        }
        self._descriptions: dict[bytes, tuple[BlockStored, bytes]] = {}
        self._resident: set[bytes] = set()
        self._catalogue_bytes = 0
        self._journal: deque[tuple[int, bytes]] = deque()
        self._journal_bytes = 0
        self._next_seq = 0
        self._sessions: dict[str, _Session] = {}
        self._profile_unavailable: str | None = None
        # An already warm pool or unconsumed prior events cannot prove history.
        self._unavailable = (
            "unknown_cache_origin"
            if len(pool.cached_block_hash_to_block) or pool.kv_event_queue
            else None
        )

    def _owner(self) -> None:
        if threading.get_ident() != self._thread:
            raise SnapshotError("invalid", "not_cache_owner_thread")

    @staticmethod
    def _key(block_hash: bytes | int) -> bytes:
        return msgspec.msgpack.encode(block_hash)

    def _payload(self, events: list[KVCacheEvent]) -> bytes:
        return msgspec.msgpack.encode(
            KVEventBatch(
                ts=0.0, events=events, data_parallel_rank=self.identity["rank"]
            )
        )

    def _poison(self, reason: str) -> None:
        self._unavailable = reason
        self._descriptions.clear()
        self._resident.clear()
        self._catalogue_bytes = 0
        self._journal.clear()
        self._journal_bytes = 0
        self._sessions.clear()

    def unavailable(self, reason: str) -> None:
        self._owner()
        self._profile_unavailable = reason
        self._poison(reason)

    def _describe(self, event: BlockStored) -> list[BlockStored]:
        size = self.identity["block_size"]
        if (
            event.block_size != size
            or event.group_idx not in (None, 0)
            or event.medium not in (None, MEDIUM_GPU)
            or event.locality not in (None, "LOCAL")
            or event.ownership is not None
            or event.lora_name is not None
            or event.lora_id not in (None, 0)
            or event.kv_cache_spec_kind not in (None, "full_attention")
            or event.kv_cache_spec_sliding_window is not None
            or getattr(event, "lookahead_token_ids", None)
            or (event.extra_keys and any(key is not None for key in event.extra_keys))
            or (
                event.extra_keys is not None
                and len(event.extra_keys) != len(event.block_hashes)
            )
            or len(event.token_ids) != size * len(event.block_hashes)
            or any(
                type(token) is not int or not 0 <= token <= 0xFFFFFFFF
                for token in event.token_ids
            )
        ):
            raise SnapshotError("unsupported", "unsupported_cache_profile")
        parent = event.parent_block_hash
        descriptions = []
        for offset, block_hash in enumerate(event.block_hashes):
            key = self._key(block_hash)
            if parent is not None and self._key(parent) not in self._descriptions:
                raise SnapshotError("unavailable", "missing_parent_description")
            single = BlockStored(
                block_hashes=[block_hash],
                parent_block_hash=parent,
                token_ids=event.token_ids[offset * size : (offset + 1) * size],
                block_size=size,
                lora_id=None,
                medium=MEDIUM_GPU,
                lora_name=None,
                group_idx=0,
                kv_cache_spec_kind="full_attention",
                locality="LOCAL",
            )
            payload = self._payload([single])
            previous = self._descriptions.get(key)
            if previous is not None and previous[1] != payload:
                raise SnapshotError("conflict", "conflicting_block_description")
            if previous is None:
                self._descriptions[key] = (single, payload)
                self._catalogue_bytes += len(payload)
            descriptions.append(single)
            parent = block_hash
        return descriptions

    def _actual_resident(self) -> set[bytes]:
        resident = set()
        # This is the authoritative pool map, including duplicate physical blocks.
        for block_hash in self.pool.cached_block_hash_to_block._cache:
            if get_group_id(block_hash) != 0:
                raise SnapshotError("unsupported", "unsupported_cache_group")
            resident.add(
                self._key(maybe_convert_block_hash(get_block_hash(block_hash)))
            )
        if not resident.issubset(self._descriptions):
            raise SnapshotError("unavailable", "missing_resident_description")
        return resident

    def _collect_descriptions(self, roots: set[bytes]) -> None:
        needed: set[bytes] = set()
        pending = list(roots)
        while pending:
            key = pending.pop()
            if key in needed or key not in self._descriptions:
                continue
            needed.add(key)
            parent = self._descriptions[key][0].parent_block_hash
            if parent is not None:
                pending.append(self._key(parent))
        for key in self._descriptions.keys() - needed:
            _, payload = self._descriptions.pop(key)
            self._catalogue_bytes -= len(payload)

    def _touched_residents(self, touched: dict[bytes, bytes | int]) -> set[bytes]:
        cache = self.pool.cached_block_hash_to_block
        if any(isinstance(block_hash, int) for block_hash in touched.values()):
            # The legacy integer projection is not invertible. Use the actual
            # pool keys rather than manufacture a raw hash from that projection.
            return {
                key
                for raw in cache._cache
                if get_group_id(raw) == 0
                and (key := self._key(maybe_convert_block_hash(get_block_hash(raw))))
                in touched
            }
        resident = set()
        for key, block_hash in touched.items():
            assert isinstance(block_hash, bytes)
            if (
                cache.get_one_block(
                    make_block_hash_with_group_id(BlockHash(block_hash), 0)
                )
                is not None
            ):
                resident.add(key)
        return resident

    def publish(
        self, events: list[KVCacheEvent], active_requests: Iterable["Request"] = ()
    ) -> None:
        """Fold the normal owner drain, then assign the capability stream seq.

        The legacy batch is not modified. Logical Remove is emitted only after
        the final physical copy disappears from the real pool map.
        """
        self._owner()
        if not events:
            return
        normalized: list[KVCacheEvent] = []
        touched: set[bytes] = set()
        hashes = {
            self._key(block_hash): block_hash
            for event in events
            if isinstance(event, (BlockStored, BlockRemoved))
            for block_hash in event.block_hashes
        }
        resident = self._touched_residents(hashes)
        if not any(isinstance(event, AllBlocksCleared) for event in events):
            resident.update(self._resident - hashes.keys())
        active = {
            self._key(maybe_convert_block_hash(block_hash))
            for request in active_requests
            for block_hash in request.block_hashes
        }
        parents = {
            self._key(event.parent_block_hash)
            for event in events
            if isinstance(event, BlockStored) and event.parent_block_hash is not None
        }
        self._collect_descriptions(resident | active | parents)
        previous_resident = self._resident
        try:
            for event in events:
                if isinstance(event, AllBlocksCleared):
                    self._descriptions.clear()
                    self._resident.clear()
                    self._catalogue_bytes = 0
                    self._unavailable = self._profile_unavailable
                    previous_resident = set()
                    normalized = [AllBlocksCleared()]
                    touched.clear()
                elif self._unavailable is not None:
                    continue
                elif isinstance(event, BlockStored):
                    descriptions = self._describe(event)
                    normalized.extend(descriptions)
                    touched.update(self._key(e.block_hashes[0]) for e in descriptions)
                elif isinstance(event, BlockRemoved):
                    if event.group_idx not in (None, 0) or event.medium not in (
                        None,
                        MEDIUM_GPU,
                    ):
                        raise SnapshotError("unsupported", "unsupported_remove_scope")
                else:
                    raise SnapshotError("unsupported", "unsupported_event")
            if self._unavailable is not None:
                return
            if not resident.issubset(self._descriptions):
                raise SnapshotError("unavailable", "missing_resident_description")
            removed = (previous_resident | touched) - resident
            if removed:
                normalized.append(
                    BlockRemoved(
                        block_hashes=[hashes[key] for key in sorted(removed)],
                        medium=MEDIUM_GPU,
                        group_idx=0,
                    )
                )
            self._resident = resident
            self._collect_descriptions(resident | active)
            if (
                len(self._descriptions) > self.limits.max_entries
                or self._catalogue_bytes > self.limits.max_catalogue_bytes
            ):
                raise SnapshotError("unavailable", "catalogue_capacity")
            if normalized:
                self._append(self._payload(normalized))
        except SnapshotError as error:
            self._poison(error.reason)

    def _expire(self) -> None:
        now = self.clock()
        self._sessions = {
            key: value for key, value in self._sessions.items() if value.expires > now
        }

    def _append(self, payload: bytes) -> None:
        self._expire()
        self._journal.append((self._next_seq, payload))
        self._next_seq += 1
        self._journal_bytes += len(payload)
        while self._journal and (
            len(self._journal) > self.limits.max_journal_batches
            or self._journal_bytes > self.limits.max_journal_bytes
        ):
            seq, old = self._journal.popleft()
            self._journal_bytes -= len(old)
            # Capacity loss invalidates every pin that needed this record.
            self._sessions = {
                key: value
                for key, value in self._sessions.items()
                if value.acknowledged >= seq
            }

    def _ready(self) -> None:
        self._owner()
        if self._unavailable is not None:
            raise SnapshotError("unavailable", self._unavailable)
        if self.pool.kv_event_queue:
            raise SnapshotError("invalid", "owner_events_not_flushed")
        try:
            resident = self._actual_resident()
        except SnapshotError as error:
            self._poison(error.reason)
            raise
        if resident != self._resident:
            self._poison("cache_changed_without_events")
            raise SnapshotError("unavailable", "cache_changed_without_events")

    def capabilities(self) -> dict[str, Any]:
        self._ready()
        return {
            "status": "complete",
            "schema": 1,
            "capability": "kv-snapshot-v1",
            "identity": self.identity.copy(),
            "limits": {
                "lease_seconds": self.limits.lease_seconds,
                "max_snapshot_bytes": self.limits.max_snapshot_bytes,
                "max_sessions": self.limits.max_sessions,
                "max_journal_batches": self.limits.max_journal_batches,
            },
        }

    def _bounded_reply(self, reply: dict[str, Any]) -> dict[str, Any]:
        # Each wire byte expands to at most three JSON digits plus a comma.
        # Bound small envelope metadata and fixed per-record fields without
        # serializing multi-MB payload arrays on the cache owner thread.
        metadata = {
            key: value
            for key, value in reply.items()
            if key not in ("descriptors", "replay")
        }
        upper = len(json.dumps(metadata, separators=(",", ":")).encode()) + 64
        for field in ("descriptors", "replay"):
            upper += sum(128 + 4 * len(row["payload"]) for row in reply.get(field, ()))
        if upper > self.limits.max_snapshot_bytes:
            raise SnapshotError("capacity", "snapshot_bytes_full")
        return reply

    def open_snapshot(self, lease_id: str | None = None) -> dict[str, Any]:
        """Freeze S(W) and pin W+1 in the same owner-loop operation."""
        self._ready()
        self._expire()
        if len(self._sessions) >= self.limits.max_sessions:
            raise SnapshotError("capacity", "snapshot_sessions_full")
        watermark = self._next_seq - 1
        descriptors = []
        digest = hashlib.sha256()
        for key, (_, payload) in self._descriptions.items():
            resident = key in self._resident
            digest.update(bytes([resident]))
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
            descriptors.append({"payload": list(payload), "resident": resident})
        lease = lease_id or uuid.uuid4().hex
        if lease in self._sessions:
            raise SnapshotError("invalid", "duplicate_lease")
        envelope = {
            "status": "complete",
            "schema": 1,
            "capability": "kv-snapshot-v1",
            "snapshot_required": True,
            "identity": self.identity.copy(),
            "snapshot_id": uuid.uuid4().hex,
            "watermark": watermark,
            "descriptors": descriptors,
            "descriptor_count": len(descriptors),
            "digest": digest.hexdigest(),
            "lease_id": lease,
            "lease_seconds": self.limits.lease_seconds,
            "high_watermark": watermark,
            "retained_from": self._journal[0][0] if self._journal else self._next_seq,
            "replay": [],
        }
        self._bounded_reply(envelope)
        self._sessions[lease] = _Session(
            watermark,
            self.clock() + self.limits.lease_seconds,
            delivered=watermark,
            acknowledged=watermark,
        )
        return envelope

    def _session(self, lease: str) -> _Session:
        self._owner()
        session = self._sessions.get(lease)
        if session is None or session.expires <= self.clock():
            self._sessions.pop(lease, None)
            raise SnapshotError("expired", "lease_expired_or_retention_lost")
        if self._unavailable is not None:
            raise SnapshotError("unavailable", self._unavailable)
        return session

    def catch_up(self, lease: str) -> dict[str, Any]:
        session = self._session(lease)
        if self.pool.kv_event_queue:
            raise SnapshotError("invalid", "owner_events_not_flushed")
        if session.high is None:
            session.high = self._next_seq - 1
        reply = self._bounded_reply(
            {
                **self._range(session.watermark, session.high),
                "lease_id": lease,
                "lease_seconds": self.limits.lease_seconds,
            }
        )
        session.delivered = max(session.delivered, session.high)
        return reply

    def _range(self, after: int, high: int) -> dict[str, Any]:
        retained = self._journal[0][0] if self._journal else self._next_seq
        frames = [
            (seq, payload) for seq, payload in self._journal if after < seq <= high
        ]
        if high - after != len(frames):
            raise SnapshotError("unavailable", "retained_range_missing")
        return {
            "status": "complete",
            "identity": self.identity.copy(),
            "high_watermark": high,
            "retained_from": retained,
            "replay": [
                {"seq": seq, "payload": list(payload)} for seq, payload in frames
            ],
        }

    def live(self, lease: str, after: int) -> dict[str, Any]:
        """Read H+1 onward from the same pinned immutable log, without END inference."""
        session = self._session(lease)
        if (
            session.high is None
            or not session.acknowledged <= after <= session.delivered
            or after < session.high
        ):
            raise SnapshotError("invalid", "invalid_live_cursor")
        reply = self._bounded_reply(
            {
                **self._range(after, self._next_seq - 1),
                "lease_id": lease,
                "lease_seconds": self.limits.lease_seconds,
            }
        )
        # ACK only already delivered data. An idle poll renews the finite lease;
        # no session permanently pins its initial snapshot watermark.
        session.acknowledged = after
        session.delivered = reply["high_watermark"]
        session.expires = self.clock() + self.limits.lease_seconds
        return reply

    def release(self, lease: str) -> None:
        self._owner()
        self._sessions.pop(lease, None)

    def restart_stream(self) -> dict[str, Any]:
        """Keep cache provenance, replace publication identity and invalidate pins."""
        self._owner()
        self.identity["stream_id"] = uuid.uuid4().hex
        self._sessions.clear()
        self._journal.clear()
        self._journal_bytes = 0
        self._next_seq = 0
        return {"identity": self.identity.copy(), "snapshot_required": True}
