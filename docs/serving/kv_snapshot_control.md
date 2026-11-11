# Experimental worker-local KV snapshot control

This proposal supplies a coherent full-attention cache-metadata cut to an
explicitly enrolled Router. It supports recovery after early KV events have
left replay; it does not transfer tensors or guarantee instantaneous device
residency. The interface is default-off and remains a proposal associated with
[Router recovery RFC #426](https://github.com/vllm-project/router/issues/426).

## Enable the reference implementation

Configure KV events and an explicit deployment namespace, and supply a private
file containing a separate control credential:

```sh
VLLM_USE_RUST_FRONTEND=0 vllm serve MODEL --enable-prefix-caching --block-size 16 \
  --kv-events-config '{"enable_kv_cache_events":true,"snapshot_namespace":"MODEL-INPUT-HASH-B16"}' \
  --kv-snapshot-control-key-file /private/snapshot-control.key
```

This proposal registers the control route on the Python HTTP frontend only;
the Rust frontend, gRPC and headless mode do not expose it. Use one API server
for the independently addressable engine. The chosen model/backend must retain
the configured block size and satisfy the profile below; a parser accepting the
options is not proof that engine startup has activated the capability.

The namespace binds the actual model, input and block-hash contract. Do not
reuse it for incompatible deployments. Only one independently addressable,
managed Engine is supported; its global rank may be nonzero. A load-balanced
HTTP endpoint serving multiple Engines is refused before a lease is created.
The initial profile requires one full-attention cache group, matching hash and
cache block sizes, prefix caching and zero lookahead. Hybrid, shared-pool,
adapter and native shared-endpoint DP layouts are not supported by this
capability. Unsupported Workers remain available for ordinary inference.

The control credential is distinct from inference API keys. Its file contains
visible ASCII bytes (0x21–0x7e), optionally followed by one LF or CRLF.
Leading/trailing spaces, embedded whitespace, extra newlines, control bytes and
non-ASCII are rejected rather than normalized. Keep the file private. Enabling
the route requires no development endpoints. Response-payload logging must be
disabled because descriptors can contain request token IDs.

## Control and ownership

`POST /v1/kv_snapshot` accepts only `capabilities`, `open`, `catch_up`,
`live` and `release`. It requires `X-KV-Snapshot-Control`; when ordinary
API-key authentication is configured, its Bearer credential is also required.
The existing middleware protects the `/v1` namespace, including under a
configured root path. Inference credentials alone cannot authorize metadata
export. Use a trusted service boundary and verified TLS outside loopback; never
expose the control credential through a customer-facing inference proxy.

The request names the configured namespace and actual global Engine rank.
Typed utility routing checks the sole managed Engine, then dispatches to its
owner rather than broadcasting. The owner drains cache events and freezes
S(W) and retention together. Each reply identifies the source, cache and stream
incarnations. Descriptors include nonresident ancestors needed for structure,
but those ancestors do not earn cache credit.

The client supplies its lease UUID before open so a lost response can be
released without guessing whether the open had side effects. Catch-up fixes a
finite H and returns W+1 through H. The same journal then serves live ranges.
An ACK cannot exceed successfully delivered data; refusing an oversized reply
does not advance delivered or acknowledged progress. After applying a range,
the client acknowledges its cursor, releasing old retention and renewing the
finite lease. Clear resets the applicable ancestry boundary.

The proposed Router consumer stages and validates the cut and suffix before
publishing worker-local credit through its registration/attempt fence. It
consumes already-received post-H changes before publication and does not splice
legacy ZMQ sequence numbers into this journal. A gap, stale identity, expired
lease or lost retention revokes the affected source's credit, not its ability
to receive ordinary requests. Lookup validity is at query completion, not an
atomic reservation-time promise.

## Effective limits

| Boundary | Limit and measurement |
|---|---|
| HTTP control | 4 KiB request; 8 MiB serialized JSON response; two concurrent requests; finite 3-second operation timeout. |
| Producer catalogue defaults | 16,384 retained descriptions and 4 MiB encoded descriptor bytes. Required ancestors count; this is not a peak RSS bound. |
| Producer journal defaults | 1,024 batches / 4 MiB encoded journal bytes; two sessions; finite 30-second renewable lease. |
| Proposed Router initial cut | 4,096 descriptors and 1 MiB combined descriptor/replay **wire payload** bytes, not JSON envelope bytes. |
| Proposed Router semantic history | 4,096 descriptions / 1 MiB **token bytes**, followed by a new authoritative snapshot at history pressure. |

The usable deployment capacity is the intersection of both sides' limits, not
the larger advertised number. A producer fitting its own catalogue can exceed
the initial Router bound. Such a response must be reported as unsupported or
insufficient capacity, not accepted partially or retried in a hot loop.
Snapshot entry counts, wire bytes, token bytes and JSON response bytes are
different measurements.

Cold attachment to an already warm pool without maintained provenance remains
unavailable. A later live Store cannot reconstruct missing history. Stream
restart invalidates old sessions; owner reconstruction changes the source and
cache identity. Release affects only the named session in its matched identity.
If the control channel is gone, expiry bounds remaining pins but is not proof
of immediate remote cleanup.

## Validation boundary

CPU tests cover the actual catalogue and typed control layers with synthetic
requests. The separate CPU peer assembles engine/client objects without their
normal constructors and uses test-owned socket-to-queue forwarding around the
real utility dispatcher. Production HTTP/auth, client codec, owner dispatch,
Scheduler publication and Router consumption execute; stock process
startup/I/O pumps and Scheduler constructor activation are not covered.
This is not evidence of model execution, Ascend/GPU support, every vLLM version,
or measured serving-performance improvement.
