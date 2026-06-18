//! NFQUEUE outbound-SYN reader. Sense-only: the verdict is always Accept;
//! containment is a separate nft rule (see `nft`/`actuators`). Confirmed against
//! a real netns by the Plan B spike.
use nfq::{Queue, Verdict};
use std::net::Ipv4Addr;
use std::sync::mpsc::Sender;

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

/// Drive the NFQUEUE: receive, parse, forward, ACCEPT. Sensing only — the
/// verdict is always Accept; containment is a separate nft rule. Runs until a
/// recv/verdict error; intended for a background thread.
pub fn run_reader(queue_num: u16, tx: Sender<OutboundSyn>) -> std::io::Result<()> {
    let mut queue = Queue::open()?;
    queue.bind(queue_num)?;
    loop {
        let mut msg = queue.recv()?;
        if let Some(syn) = parse_ipv4_tcp_syn(msg.get_payload()) {
            // A full channel (daemon stalled) must not block the verdict; drop
            // the event rather than the packet.
            let _ = tx.send(syn);
        }
        msg.set_verdict(Verdict::Accept);
        queue.verdict(msg)?;
    }
}
