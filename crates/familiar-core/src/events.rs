use crate::{Pid, Timestamp};

/// A process as an adapter sees it: its id and the resolved executable path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProcessRef {
    pub pid: Pid,
    pub exe: String,
}

/// The normalized, platform-neutral event vocabulary. Adapters translate OS
/// specifics into this; the core only ever sees this.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    /// A process read a file.
    FileRead {
        at: Timestamp,
        process: ProcessRef,
        path: String,
    },
    /// A process opened (or attempted) an outbound connection.
    OutboundConn {
        at: Timestamp,
        process: ProcessRef,
        dst_ip: String,
        dst_port: u16,
    },
    /// A process started.
    ProcessStart {
        at: Timestamp,
        process: ProcessRef,
        parent: Pid,
    },
    /// A process exited.
    ProcessExit { at: Timestamp, pid: Pid },
}

impl Event {
    /// The timestamp the adapter stamped on this event.
    pub fn at(&self) -> Timestamp {
        match self {
            Event::FileRead { at, .. }
            | Event::OutboundConn { at, .. }
            | Event::ProcessStart { at, .. }
            | Event::ProcessExit { at, .. } => *at,
        }
    }

    /// The subject process id.
    pub fn pid(&self) -> Pid {
        match self {
            Event::FileRead { process, .. }
            | Event::OutboundConn { process, .. }
            | Event::ProcessStart { process, .. } => process.pid,
            Event::ProcessExit { pid, .. } => *pid,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(pid: Pid) -> ProcessRef {
        ProcessRef {
            pid,
            exe: "/usr/bin/curl".into(),
        }
    }

    #[test]
    fn at_returns_the_stamped_time() {
        let ev = Event::OutboundConn {
            at: 42,
            process: p(7),
            dst_ip: "1.1.1.1".into(),
            dst_port: 443,
        };
        assert_eq!(ev.at(), 42);
    }

    #[test]
    fn pid_extracts_the_subject_process() {
        assert_eq!(
            Event::FileRead {
                at: 1,
                process: p(7),
                path: "/x".into()
            }
            .pid(),
            7
        );
        assert_eq!(Event::ProcessExit { at: 1, pid: 9 }.pid(), 9);
    }
}
