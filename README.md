# oxideav-vp6

A pure-Rust VP6 video codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

**Orphan-rebuild scaffold (2026-05-18).** The prior implementation was
retired under the workspace
[clean-room policy](https://github.com/OxideAV/oxideav/blob/master/docs/IMPLEMENTOR_ROUND.md):
multiple source files acknowledged that their implementations were
direct ports of an external library's VP6 codebase — which violates
the clean-room provenance requirement even though VP6 has no public
written specification. Master history was fully erased per the Hat-3
cold-enforcement procedure.

The implementation will be re-built against fresh reverse-engineering
of real VP6 bitstreams (byte traces only, no external library source)
in a future clean-room round.

## License

MIT — see [LICENSE](./LICENSE).
