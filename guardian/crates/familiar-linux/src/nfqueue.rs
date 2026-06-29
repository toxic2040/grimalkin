//! NFQUEUE outbound-SYN reader. Sense-only: the verdict is always Accept;
//! containment is a separate nft rule (see `nft`/`actuators`). Confirmed against
//! a real netns by the Plan B spike.
use nfq::{Queue, Verdict};
use std::net::Ipv4Addr;
use std::sync::mpsc::{SyncSender, TrySendError};

/// Bound on the outbound-SYN channel between the NFQUEUE reader and the
/// supervisor. Sensing must never grow memory if the supervisor stalls, so the
/// channel is bounded and the reader drops on full rather than blocking the
/// verdict (see `forward`).
pub const SYN_CHANNEL_CAP: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutboundSyn {
    pub dst: Ipv4Addr,
    pub dport: u16,
    pub src_port: u16,
}

/// Parse an IPv4+TCP packet, returning the tuple only for a *connection-opening*
/// segment (SYN set, ACK clear). The queue rule diverts all outbound TCP, so the
/// SYN filter here is what makes this "new outbound connection" sensing rather
/// than per-packet noise; it also drops the listener's SYN-ACK.
pub fn parse_ipv4_tcp_syn(pkt: &[u8]) -> Option<OutboundSyn> {
    if pkt.len() < 20 || (pkt[0] >> 4) != 4 {
        return None;
    }
    let ihl = ((pkt[0] & 0x0f) as usize) * 4;
    if pkt[9] != 6 || pkt.len() < ihl + 14 {
        return None; // not TCP, or truncated before the flags byte
    }
    let flags = pkt[ihl + 13]; // TCP flags byte
    let is_pure_syn = (flags & 0x02) != 0 && (flags & 0x10) == 0; // SYN set, ACK clear
    if !is_pure_syn {
        return None;
    }
    let dst = Ipv4Addr::new(pkt[16], pkt[17], pkt[18], pkt[19]);
    let src_port = u16::from_be_bytes([pkt[ihl], pkt[ihl + 1]]);
    let dport = u16::from_be_bytes([pkt[ihl + 2], pkt[ihl + 3]]); // +2 = dest port
    Some(OutboundSyn {
        dst,
        dport,
        src_port,
    })
}

/// Hand `syn` to the supervisor without ever blocking the verdict. A full
/// channel means the supervisor has stalled: drop the event (not the packet) so
/// memory stays bounded, and count it. A disconnected supervisor is also a drop
/// — the reader keeps issuing Accept verdicts regardless, since stalling the
/// kernel queue would be worse than losing a sensing event.
fn forward(tx: &SyncSender<OutboundSyn>, syn: OutboundSyn, dropped: &mut u64) {
    match tx.try_send(syn) {
        Ok(()) => {}
        Err(TrySendError::Full(_)) => {
            *dropped += 1;
            // Throttle: a wedged supervisor must not flood the journal. Log on
            // each power-of-two so the running total stays visible.
            if dropped.is_power_of_two() {
                eprintln!(
                    "[familiar] nfqueue: supervisor stalled; dropped {dropped} outbound-SYN event(s)"
                );
            }
        }
        Err(TrySendError::Disconnected(_)) => {}
    }
}

/// Drive the NFQUEUE: receive, parse, forward, ACCEPT. Sensing only — the
/// verdict is always Accept; containment is a separate nft rule. Runs until a
/// recv/verdict error; intended for a background thread.
pub fn run_reader(queue_num: u16, tx: SyncSender<OutboundSyn>) -> std::io::Result<()> {
    let mut queue = Queue::open()?;
    queue.bind(queue_num)?;
    let mut dropped: u64 = 0;
    loop {
        let mut msg = queue.recv()?;
        if let Some(syn) = parse_ipv4_tcp_syn(msg.get_payload()) {
            forward(&tx, syn, &mut dropped);
        }
        msg.set_verdict(Verdict::Accept);
        queue.verdict(msg)?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::sync_channel;

    fn syn() -> OutboundSyn {
        OutboundSyn {
            dst: Ipv4Addr::new(203, 0, 113, 9),
            dport: 443,
            src_port: 34000,
        }
    }

    #[test]
    fn forward_drops_and_counts_when_full_without_blocking() {
        // Capacity-1 channel, receiver held alive so it is Full (not Disconnected).
        let (tx, _rx) = sync_channel::<OutboundSyn>(1);
        let mut dropped = 0u64;
        forward(&tx, syn(), &mut dropped); // fills the only slot
        assert_eq!(dropped, 0, "first event fits");
        forward(&tx, syn(), &mut dropped); // full -> must drop, not block
        assert_eq!(dropped, 1, "the over-capacity event is dropped and counted");
    }
}
