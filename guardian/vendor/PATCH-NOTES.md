# vendored rustables 0.8.7 — fd double-close fix

Upstream 0.8.7 (latest published) requires `nix ^0.30`, whose `socket()` returns
an `OwnedFd`. `src/query.rs::socket_close_wrapper` still manually
`nix::unistd::close()`s the raw fd, so the fd is closed twice (once manually,
once when the `OwnedFd` drops). On Rust 1.95 the std IO-safety guard turns the
second close into `fatal runtime error: IO Safety violation` and aborts.

Fix: remove the manual `close()`; both callers (`batch.rs::send`, the `query.rs`
list functions) keep the `OwnedFd` alive to end-of-scope, so its `Drop` closes
the fd once. Verified against a real netns (add + remove a drop rule).

Upstream PR: file the same one-line change against the rustables repo; drop this
vendor copy once a fixed version is published.
