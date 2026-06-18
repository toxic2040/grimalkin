use familiar_linux::nfqueue::{OutboundSyn, parse_ipv4_tcp_syn};

/// IPv4(20)+TCP header with the flags byte at TCP offset 13.
fn syn_packet(flags: u8) -> Vec<u8> {
    let mut pkt = vec![0u8; 34];
    pkt[0] = 0x45; // version 4, ihl 5
    pkt[9] = 6; // TCP
    pkt[16..20].copy_from_slice(&[203, 0, 113, 9]);
    pkt[20..22].copy_from_slice(&34000u16.to_be_bytes()); // src port
    pkt[22..24].copy_from_slice(&443u16.to_be_bytes()); // dst port
    pkt[20 + 13] = flags; // TCP flags
    pkt
}

#[test]
fn parses_dst_and_dport_from_a_syn() {
    let syn = parse_ipv4_tcp_syn(&syn_packet(0x02)).expect("parse"); // SYN
    assert_eq!(
        syn,
        OutboundSyn {
            dst: "203.0.113.9".parse().unwrap(),
            dport: 443,
            src_port: 34000
        }
    );
}

#[test]
fn rejects_non_ipv4_non_tcp_and_non_syn() {
    assert!(parse_ipv4_tcp_syn(&[0x60; 34]).is_none()); // v6
    let mut udp = vec![0u8; 34];
    udp[0] = 0x45;
    udp[9] = 17; // UDP
    assert!(parse_ipv4_tcp_syn(&udp).is_none());
    assert!(parse_ipv4_tcp_syn(&syn_packet(0x12)).is_none()); // SYN+ACK (not a new outbound)
    assert!(parse_ipv4_tcp_syn(&syn_packet(0x10)).is_none()); // ACK only
}

#[test]
fn captures_a_real_outbound_syn_in_netns() {
    use std::process::Command;
    if std::env::var("FAMILIAR_IN_NETNS").is_err() {
        let exe = std::env::current_exe().unwrap();
        let out = Command::new("unshare")
            .args(["-Urn"])
            .arg(&exe)
            .arg("--exact")
            .arg("captures_a_real_outbound_syn_in_netns")
            .arg("--test-threads=1")
            .arg("--nocapture")
            .env("FAMILIAR_IN_NETNS", "1")
            .output()
            .unwrap();
        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr);
            if err.contains("Operation not permitted") || err.contains("namespace") {
                eprintln!(
                    "SKIP captures_a_real_outbound_syn_in_netns: unprivileged userns unavailable"
                );
                return;
            }
            panic!("netns child failed:\n{err}");
        }
        return;
    }
    let _ = Command::new("ip")
        .args(["link", "set", "lo", "up"])
        .status();
    // Exercise the production divert path: the familiar table's sense chain.
    familiar_linux::nft::ensure_table().expect("create table");
    familiar_linux::nft::install_queue_rule(0).expect("install queue rule");

    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = familiar_linux::nfqueue::run_reader(0, tx);
    });
    std::thread::sleep(std::time::Duration::from_millis(200)); // let bind happen
    let l = std::net::TcpListener::bind(("127.0.0.1", 8443)).unwrap();
    std::thread::spawn(move || {
        let _ = l.accept();
    });
    let _ = std::net::TcpStream::connect(("127.0.0.1", 8443));

    let syn = rx
        .recv_timeout(std::time::Duration::from_secs(3))
        .expect("captured a syn");
    assert_eq!(syn.dport, 8443);
}
