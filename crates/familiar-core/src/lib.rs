#![forbid(unsafe_code)]
//! familiar-core — the portable, deterministic guardian spine.
//!
//! Invariant: this crate makes no OS calls and runs no model. All timestamps
//! and ids are supplied by the caller. It is identical on every platform.

pub mod advisor;
pub mod audit;
pub mod capabilities;
pub mod events;
pub mod permission;
pub mod policy;

/// Unix epoch milliseconds, supplied by the caller. The core never reads a clock.
pub type Timestamp = u64;
/// OS process id, supplied by an adapter. The core never enumerates processes.
pub type Pid = u32;
