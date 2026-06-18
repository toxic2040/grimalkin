//! The Linux event source: NFQUEUE outbound SYNs (attributed via /proc) plus
//! FileRead events streamed from the privileged fanotify helper. Both arrive on
//! channels filled by background threads; `poll()` drains what is queued.
use crate::attribution;
use crate::nfqueue::OutboundSyn;
use crate::wire::FileReadEvent;
use familiar_core::events::{Event, ProcessRef};
use familiar_platform::Sensors;
use std::sync::mpsc::Receiver;
use std::time::{SystemTime, UNIX_EPOCH};

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub struct LinuxSensors {
    syn_rx: Receiver<OutboundSyn>,
    file_rx: Receiver<FileReadEvent>,
    clock: fn() -> u64,
}

impl LinuxSensors {
    pub fn new(syn_rx: Receiver<OutboundSyn>, file_rx: Receiver<FileReadEvent>) -> Self {
        Self {
            syn_rx,
            file_rx,
            clock: now_ms,
        }
    }
}

impl Sensors for LinuxSensors {
    fn poll(&mut self) -> Vec<Event> {
        let mut events = Vec::new();
        // FileRead events from the helper (already carry pid/exe/path).
        while let Ok(fr) = self.file_rx.try_recv() {
            events.push(Event::FileRead {
                at: fr.at,
                process: ProcessRef {
                    pid: fr.pid,
                    exe: fr.exe,
                },
                path: fr.path,
            });
        }
        // Outbound SYNs: attribute the PID now (best-effort; unknown => pid 0).
        while let Ok(syn) = self.syn_rx.try_recv() {
            let process = attribution::attribute(syn.src_port).unwrap_or(ProcessRef {
                pid: 0,
                exe: String::new(),
            });
            events.push(Event::OutboundConn {
                at: (self.clock)(),
                process,
                dst_ip: syn.dst.to_string(),
                dst_port: syn.dport,
            });
        }
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::channel;

    #[test]
    fn poll_maps_file_reads_and_drains_both_channels() {
        let (_syn_tx, syn_rx) = channel();
        let (file_tx, file_rx) = channel();
        file_tx
            .send(FileReadEvent {
                at: 1000,
                pid: 7,
                exe: "/usr/bin/curl".into(),
                path: "/home/u/.ssh/id".into(),
            })
            .unwrap();
        let mut sensors = LinuxSensors::new(syn_rx, file_rx);
        let events = sensors.poll();
        assert_eq!(events.len(), 1);
        match &events[0] {
            Event::FileRead { at, process, path } => {
                assert_eq!(*at, 1000);
                assert_eq!(process.pid, 7);
                assert_eq!(path, "/home/u/.ssh/id");
            }
            other => panic!("expected FileRead, got {other:?}"),
        }
        assert!(sensors.poll().is_empty());
    }
}
