# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Owner-loop snapshot proofs using real Request, hash and BlockPool behavior.

No model execution is needed: the protected contract is ordered cache metadata,
resident membership, retention and producer identity, not physical GPU contents.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from queue import Queue
from threading import Barrier
from uuid import uuid4

import msgspec
import pytest

from vllm.config.kv_events import KVEventsConfig
from vllm.distributed.kv_events import BlockRemoved, EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_snapshot import (
    KvSnapshotCatalogue,
    SnapshotError,
    SnapshotLimits,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256_cbor
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("namespace", [None, "synthetic-cache"])
def test_snapshot_config_keeps_legacy_publisher_factory_working(namespace):
    config = KVEventsConfig(
        enable_kv_cache_events=True,
        endpoint=f"inproc://snapshot-factory-{uuid4().hex}",
        snapshot_namespace=namespace,
    )
    publisher = EventPublisherFactory.create(config)
    try:
        assert publisher.get_publisher_config().endpoint == config.endpoint
    finally:
        publisher.shutdown()


def make_pool(*, limits=None, clock=None, source="producer-A"):
    init_none_hash(sha256_cbor)
    pool = BlockPool(32, True, 16, enable_kv_cache_events=True)
    options = {"limits": limits or SnapshotLimits(max_journal_batches=2)}
    if clock is not None:
        options["clock"] = clock
    catalogue = KvSnapshotCatalogue(
        pool, "synthetic-full-attention-sha256cbor-B16", source, 0, 16, **options
    )
    return pool, catalogue


def test_live_ack_renews_finite_lease_and_releases_old_retention():
    now = [0.0]
    pool, catalogue = make_pool(clock=lambda: now[0])
    _, old = store(pool)
    flush(pool, catalogue)
    snapshot = catalogue.open_snapshot()
    lease = snapshot["lease_id"]
    assert catalogue.catch_up(lease)["high_watermark"] == 0
    for step in range(1, 8):
        now[0] += 10
        pool.evict_blocks({block.block_id for block in old})
        _, old = store(pool, 1000 + step * 100, f"live-{step}")
        flush(pool, catalogue)
        response = catalogue.live(lease, step - 1)
        assert [frame["seq"] for frame in response["replay"]] == [step]
        assert response["identity"] == snapshot["identity"]
        assert response["high_watermark"] == step
    assert now[0] > catalogue.limits.lease_seconds
    assert response["lease_id"] == lease
    with pytest.raises(SnapshotError, match="duplicate_lease"):
        catalogue.open_snapshot(lease)
    assert catalogue._sessions[lease].acknowledged == 6
    assert len(catalogue._journal) == 2
    catalogue.release(lease)
    assert not catalogue._sessions


@pytest.mark.parametrize("max_bytes", [1200, 8192])
def test_catch_up_only_marks_a_bounded_reply_delivered(max_bytes):
    pool, catalogue = make_pool(
        limits=replace(SnapshotLimits(), max_snapshot_bytes=max_bytes)
    )
    snapshot = catalogue.open_snapshot()
    lease = snapshot["lease_id"]
    session = catalogue._sessions[lease]
    assert (session.delivered, session.acknowledged) == (-1, -1)
    store(pool)
    flush(pool, catalogue)
    if max_bytes == 1200:
        with pytest.raises(SnapshotError, match="snapshot_bytes_full"):
            catalogue.catch_up(lease)
        assert session.high == 0  # The finite cut may remain fixed after refusal.
        assert (session.delivered, session.acknowledged) == (-1, -1)
        with pytest.raises(SnapshotError, match="invalid_live_cursor"):
            catalogue.live(lease, 0)
        assert (session.delivered, session.acknowledged) == (-1, -1)
    else:
        reply = catalogue.catch_up(lease)
        assert [frame["seq"] for frame in reply["replay"]] == [0]
        assert (session.delivered, session.acknowledged) == (0, -1)
        assert catalogue.live(lease, 0)["replay"] == []
        assert (session.delivered, session.acknowledged) == (0, 0)
    catalogue.release(lease)
    assert not catalogue._sessions


@pytest.mark.parametrize("failure", ["forward", "regression", "expired", "restart"])
def test_live_ack_rejects_unproven_or_old_session(failure):
    now = [0.0]
    pool, catalogue = make_pool(clock=lambda: now[0])
    store(pool)
    flush(pool, catalogue)
    lease = catalogue.open_snapshot()["lease_id"]
    catalogue.catch_up(lease)
    if failure == "forward":
        after = 1
    elif failure == "regression":
        store(pool, 1000, "Y")
        flush(pool, catalogue)
        catalogue.live(lease, 0)
        catalogue.live(lease, 1)
        after = 0
    elif failure == "expired":
        now[0] = catalogue.limits.lease_seconds
        after = 0
    else:
        catalogue.restart_stream()
        after = 0
    with pytest.raises(SnapshotError):
        catalogue.live(lease, after)
    if failure in ("expired", "restart"):
        assert lease not in catalogue._sessions


def store(pool, start=100, name="X"):
    request = Request(
        name,
        list(range(start, start + 33)),
        SamplingParams(max_tokens=1),
        None,
        block_hasher=get_request_block_hasher(16, sha256_cbor),
    )
    blocks = pool.get_new_blocks(2)
    pool.cache_full_blocks(request, blocks, 0, 2, 16, 0)
    pool.free_blocks(blocks)
    return request, blocks


def flush(pool, catalogue, active_requests=()):
    events = pool.take_events()
    catalogue.publish(events, active_requests)
    return events


def cold_fixture(*, start=100, source="producer-A", remove=False):
    pool, catalogue = make_pool(source=source)
    request, blocks = store(pool, start)
    flush(pool, catalogue)
    unrelated, _ = store(pool, 1000, "unrelated-Q")
    flush(pool, catalogue)
    pool.emit_cached_block_events(unrelated, 2, 16, 0)
    flush(pool, catalogue)
    snapshot = catalogue.open_snapshot()
    assert snapshot["watermark"] == 2
    assert snapshot["retained_from"] == 1  # Earliest Store is no longer replayable.
    assert sum(d["resident"] for d in snapshot["descriptors"]) == 4
    if remove:
        pool.evict_blocks({block.block_id for block in blocks})
        assert len(pool.cached_block_hash_to_block) == 2  # Unrelated Q survives.
        assert all(pool.get_cached_block(h, [0]) is None for h in request.block_hashes)
    else:
        # The replay suffix must not reconstruct the requested X/Y/B prefix.
        pool.emit_cached_block_events(unrelated, 2, 16, 0)
    flush(pool, catalogue)
    snapshot.update(catalogue.catch_up(snapshot["lease_id"]))
    assert snapshot["high_watermark"] == 3
    assert [frame["seq"] for frame in snapshot["replay"]] == [3]
    catalogue.release(snapshot["lease_id"])
    assert not catalogue._sessions
    return snapshot


def structure_fixture():
    pool, catalogue = make_pool()
    request, blocks = store(pool)
    flush(pool, catalogue)
    for _ in range(2):
        pool.emit_cached_block_events(request, 2, 16, 0)
        flush(pool, catalogue)
    pool.evict_blocks({blocks[0].block_id})
    flush(pool, catalogue)
    assert len(pool.cached_block_hash_to_block) == 1
    snapshot = catalogue.open_snapshot()
    assert [d["resident"] for d in snapshot["descriptors"]] == [False, True]
    snapshot.update(catalogue.catch_up(snapshot["lease_id"]))
    catalogue.release(snapshot["lease_id"])
    return snapshot


def test_snapshot_cut_pins_suffix_and_live_bytes():
    captured = Queue()
    response_held = Barrier(2)

    def owner():
        pool, catalogue = make_pool()
        _, blocks = store(pool)
        flush(pool, catalogue)
        snapshot = catalogue.open_snapshot()
        captured.put(snapshot)
        try:
            # The response is captured but not accepted by the consumer yet.
            response_held.wait(timeout=5)
            pool.evict_blocks({block.block_id for block in blocks})
            flush(pool, catalogue)
            caught = catalogue.catch_up(snapshot["lease_id"])
            assert caught["high_watermark"] == 1
            store(pool, 300, "Y-after-cut")
            flush(pool, catalogue)
            live = catalogue.live(snapshot["lease_id"], after=1)
            assert [frame["seq"] for frame in live["replay"]] == [2]
            journal = dict(catalogue._journal)
            assert bytes(caught["replay"][0]["payload"]) == journal[1]
            assert bytes(live["replay"][0]["payload"]) == journal[2]
            # The fixed H is not an unbounded tail chase.
            again = catalogue.catch_up(snapshot["lease_id"])
            assert again["high_watermark"] == caught["high_watermark"]
            assert again["replay"] == caught["replay"]
            assert again["retained_from"] == 1
            return caught
        finally:
            catalogue.release(snapshot["lease_id"])
            assert not catalogue._sessions

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(owner)
        snapshot = captured.get(timeout=5)
        before = msgspec.msgpack.encode(snapshot)
        response_held.wait(timeout=5)
        caught = future.result(timeout=10)
        # Accept S only after real owner changes have occurred during its delivery.
        assert snapshot["watermark"] == 0
        assert all(d["resident"] for d in snapshot["descriptors"])
        assert msgspec.msgpack.encode(snapshot) == before
        removed = msgspec.msgpack.decode(
            bytes(caught["replay"][0]["payload"]), type=KVEventBatch
        )
        assert len(removed.events) == 1
        assert isinstance(removed.events[0], BlockRemoved)
        assert len(removed.events[0].block_hashes) == 2


def test_last_physical_remove_retains_only_structural_ancestor():
    pool, catalogue = make_pool()
    _, first = store(pool)
    flush(pool, catalogue)
    _, second = store(pool, name="duplicate")
    flush(pool, catalogue)
    pool.evict_blocks({first[0].block_id})
    flush(pool, catalogue)
    assert sum(d["resident"] for d in catalogue.open_snapshot()["descriptors"]) == 2
    pool.evict_blocks({second[0].block_id})
    flush(pool, catalogue)
    snapshot = catalogue.open_snapshot()
    assert [d["resident"] for d in snapshot["descriptors"]] == [False, True]
    assert len(pool.cached_block_hash_to_block) == 1
    # Description remains for the descendant, but is not a resident claim.
    decoder = msgspec.msgpack.Decoder(KVEventBatch)
    last = decoder.decode(catalogue._journal[-1][1])
    assert len(last.events) == 1 and isinstance(last.events[0], BlockRemoved)
    assert len(last.events[0].block_hashes) == 1
    for lease in list(catalogue._sessions):
        catalogue.release(lease)


def test_real_clear_resets_lost_completeness_not_later_store():
    limits = replace(SnapshotLimits(), max_entries=1)
    pool, catalogue = make_pool(limits=limits)
    store(pool)
    flush(pool, catalogue)
    with pytest.raises(SnapshotError, match="catalogue_capacity"):
        catalogue.open_snapshot()
    store(pool, 300)
    flush(pool, catalogue)
    with pytest.raises(SnapshotError):
        catalogue.open_snapshot()
    assert pool.reset_prefix_cache()
    flush(pool, catalogue)
    snapshot = catalogue.open_snapshot()
    assert snapshot["descriptor_count"] == 0
    catalogue.release(snapshot["lease_id"])


def test_warm_attach_is_not_complete():
    pool, original = make_pool()
    store(pool)
    flush(pool, original)
    late = KvSnapshotCatalogue(pool, "namespace", "source", 0, 16)
    with pytest.raises(SnapshotError, match="unknown_cache_origin"):
        late.open_snapshot()


def test_restart_preserves_cache_but_replaces_stream_and_sessions():
    pool, catalogue = make_pool()
    store(pool)
    flush(pool, catalogue)
    old = catalogue.open_snapshot()
    replacement = catalogue.restart_stream()
    assert replacement["identity"]["cache_id"] == old["identity"]["cache_id"]
    assert replacement["identity"]["stream_id"] != old["identity"]["stream_id"]
    assert replacement["snapshot_required"] is True
    with pytest.raises(SnapshotError):
        catalogue.catch_up(old["lease_id"])
    new = catalogue.open_snapshot()
    assert new["watermark"] == -1 and len(new["descriptors"]) == 2
    assert new["snapshot_required"] is True
    _, other = make_pool()
    assert other.identity["cache_id"] != catalogue.identity["cache_id"]
    catalogue.release(new["lease_id"])


@pytest.mark.parametrize("failure", ["expired", "retention", "sessions", "bytes"])
def test_snapshot_limits_fail_closed_and_release(failure):
    now = [0.0]
    limits = replace(
        SnapshotLimits(),
        max_journal_batches=1,
        max_sessions=1,
        max_snapshot_bytes=1 if failure == "bytes" else 1024 * 1024,
    )
    pool, catalogue = make_pool(limits=limits, clock=lambda: now[0])
    request, _ = store(pool)
    flush(pool, catalogue)
    if failure == "bytes":
        with pytest.raises(SnapshotError, match="snapshot_bytes_full"):
            catalogue.open_snapshot()
        assert not catalogue._sessions
        return
    snapshot = catalogue.open_snapshot()
    if failure == "expired":
        now[0] = limits.lease_seconds
    elif failure == "retention":
        for _ in range(2):
            pool.emit_cached_block_events(request, 2, 16, 0)
            flush(pool, catalogue)
    else:
        with pytest.raises(SnapshotError, match="snapshot_sessions_full"):
            catalogue.open_snapshot()
        catalogue.release(snapshot["lease_id"])
    with pytest.raises(SnapshotError):
        catalogue.catch_up(snapshot["lease_id"])
    assert not catalogue._sessions


def test_pending_owner_events_cannot_be_skipped_by_snapshot():
    pool, catalogue = make_pool()
    store(pool)
    with pytest.raises(SnapshotError, match="owner_events_not_flushed"):
        catalogue.open_snapshot()
    flush(pool, catalogue)
    snapshot = catalogue.open_snapshot()
    assert len(snapshot["descriptors"]) == 2
    catalogue.release(snapshot["lease_id"])


def test_profile_rejection_survives_clear_and_unknown_metadata_fails_closed():
    pool, catalogue = make_pool()
    catalogue.unavailable("unsupported_scheduler_profile")
    assert pool.reset_prefix_cache()
    flush(pool, catalogue)
    with pytest.raises(SnapshotError, match="unsupported_scheduler_profile"):
        catalogue.open_snapshot()
    pool, catalogue = make_pool()
    store(pool)
    # A missing descriptor is detected against the actual resident map.
    pool.take_events()
    catalogue.publish([])
    with pytest.raises(SnapshotError, match="missing_resident_description"):
        catalogue.open_snapshot()


def test_cold_artifacts_and_remove_suffix():
    regular = cold_fixture()
    removed = cold_fixture(remove=True)
    assert regular["descriptor_count"] == removed["descriptor_count"] == 4
    positive = msgspec.msgpack.decode(
        bytes(regular["replay"][0]["payload"]), type=KVEventBatch
    )
    assert all(event.token_ids[0] >= 1000 for event in positive.events)
    decoded = msgspec.msgpack.decode(
        bytes(removed["replay"][0]["payload"]), type=KVEventBatch
    )
    assert len(decoded.events) == 1 and isinstance(decoded.events[0], BlockRemoved)
    assert len(decoded.events[0].block_hashes) == 2


@pytest.mark.parametrize(
    ("same_drain", "int_hashes"),
    [(False, True), (True, True), (True, False)],
    ids=["separate-int", "same-int", "same-bytes"],
)
def test_finished_prefix_churn_preserves_capture(same_drain, int_hashes, monkeypatch):
    monkeypatch.setenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", str(int(int_hashes)))
    pool, catalogue = make_pool(limits=replace(SnapshotLimits(), max_entries=2))
    _, blocks = store(pool)
    flush(pool, catalogue)
    frozen = catalogue.open_snapshot()
    encoded = msgspec.msgpack.encode(frozen)
    pool.evict_blocks({block.block_id for block in blocks})
    if not same_drain:
        flush(pool, catalogue)
    store(pool, 300, "unrelated-Y")
    flush(pool, catalogue)
    current = catalogue.open_snapshot()
    assert current["descriptor_count"] == 2
    assert all(d["resident"] for d in current["descriptors"])
    decoded = [
        msgspec.msgpack.decode(bytes(d["payload"]), type=KVEventBatch)
        for d in current["descriptors"]
    ]
    assert [batch.events[0].token_ids[0] for batch in decoded] == [300, 316]
    assert msgspec.msgpack.encode(frozen) == encoded
    catalogue.catch_up(frozen["lease_id"])
    catalogue.release(frozen["lease_id"])
    catalogue.release(current["lease_id"])
    assert not catalogue._sessions


@pytest.mark.parametrize("same_drain", [False, True])
def test_active_and_pending_request_keep_parent_for_later_store(same_drain):
    pool, catalogue = make_pool()
    request, blocks = store(pool)
    flush(pool, catalogue)
    pool.evict_blocks({block.block_id for block in blocks})
    if not same_drain:
        flush(pool, catalogue, [request])
        assert len(catalogue._descriptions) == 2
        assert not catalogue._resident
    request.append_output_token_ids(list(range(133, 149)))
    tail = pool.get_new_blocks(1)
    pool.cache_full_blocks(request, blocks + tail, 2, 3, 16, 0)
    pool.free_blocks(tail)
    # The request may already be absent from Scheduler.requests at this drain.
    flush(pool, catalogue)
    snapshot = catalogue.open_snapshot()
    assert [d["resident"] for d in snapshot["descriptors"]] == [False, False, True]
    decoded = [
        msgspec.msgpack.decode(bytes(d["payload"]), type=KVEventBatch).events[0]
        for d in snapshot["descriptors"]
    ]
    assert decoded[2].parent_block_hash == decoded[1].block_hashes[0]
    assert decoded[1].parent_block_hash == decoded[0].block_hashes[0]
    catalogue.release(snapshot["lease_id"])
