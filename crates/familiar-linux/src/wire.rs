//! The helper↔daemon wire type. Mirrors `familiar_core::events::Event::FileRead`'s
//! payload; serialized as one JSON object per line over a Unix socket.
use serde::{Deserialize, Serialize};

/// A sensitive-path read observed by the privileged fanotify helper.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileReadEvent {
    pub at: u64,
    pub pid: u32,
    pub exe: String,
    pub path: String,
}
