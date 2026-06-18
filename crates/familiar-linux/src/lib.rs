#![forbid(unsafe_code)]
//! familiar-linux — the Linux implementation of the familiar-platform seam.
//!
//! Safe Rust only: NFQUEUE via `nfq`, the reversible block rule via the `nft`
//! binary, `/proc` attribution and the cgroup-v2 freezer via std `fs`. The
//! only `unsafe` in the whole workspace lives in the separate, privileged
//! `familiar-fanotify-helper`; this crate never holds CAP_SYS_ADMIN.

pub mod actuators;
pub mod attribution;
pub mod cgroup;
pub mod nfqueue;
pub mod nft;
pub mod notifier;
pub mod sensors;
pub mod wire;

pub use actuators::LinuxActuators;
pub use notifier::LinuxNotifier;
pub use sensors::LinuxSensors;
pub use wire::FileReadEvent;
