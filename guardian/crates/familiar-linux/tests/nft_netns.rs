//! Real netlink rule add/remove. Must run inside `unshare -Urn` (each test
//! re-execs itself into a private user+net namespace). Skips cleanly if
//! unprivileged user namespaces are unavailable.
use std::net::Ipv4Addr;
use std::process::Command;

/// Re-exec this test binary inside `unshare -Urn`, running only `name`. Returns
/// true if the child ran (and we should return), false if we are already inside.
fn reexec_in_netns(name: &str) -> bool {
    if std::env::var("FAMILIAR_IN_NETNS").is_ok() {
        return false;
    }
    let exe = std::env::current_exe().unwrap();
    let out = Command::new("unshare")
        .args(["-Urn"])
        .arg(&exe)
        .arg("--exact")
        .arg(name)
        .arg("--test-threads=1")
        .arg("--nocapture")
        .env("FAMILIAR_IN_NETNS", "1")
        .output()
        .expect("unshare");
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        if err.contains("Operation not permitted")
            || err.contains("unshare") && !out.status.success() && err.contains("namespace")
        {
            eprintln!("SKIP {name}: unprivileged userns unavailable");
            return true;
        }
        panic!("netns child failed for {name}:\n{}", err);
    }
    true
}

fn ruleset() -> String {
    String::from_utf8(
        Command::new("nft")
            .args(["list", "ruleset"])
            .output()
            .unwrap()
            .stdout,
    )
    .unwrap()
}

#[test]
fn block_rule_is_installed_then_fully_reversed() {
    if reexec_in_netns("block_rule_is_installed_then_fully_reversed") {
        return;
    }
    familiar_linux::nft::ensure_table().expect("create table");
    let handle =
        familiar_linux::nft::block_outbound(Ipv4Addr::new(203, 0, 113, 9), 443).expect("block");
    let after = ruleset();
    assert!(
        after.contains("familiar") && after.contains("drop"),
        "rule present:\n{after}"
    );
    assert!(handle.contains("203.0.113.9"), "handle names the dst");

    familiar_linux::nft::delete_table().expect("reverse");
    let clean = ruleset();
    assert!(!clean.contains("familiar"), "table gone:\n{clean}");
}

#[test]
fn actuators_block_outbound_records_and_reverses() {
    if reexec_in_netns("actuators_block_outbound_records_and_reverses") {
        return;
    }
    use familiar_core::events::ProcessRef;
    use familiar_core::policy::ProposedAction;
    use familiar_platform::Actuators;

    let mut act = familiar_linux::LinuxActuators::new("/sys/fs/cgroup").expect("new");
    let action = ProposedAction::BlockOutbound {
        process: ProcessRef {
            pid: 7,
            exe: "/usr/bin/curl".into(),
        },
        dst_ip: "203.0.113.9".into(),
        dst_port: 443,
    };
    let outcome = act.apply(&action).expect("apply");
    assert!(outcome.note.contains("203.0.113.9"));
    assert!(ruleset().contains("drop"));

    act.reverse_all().expect("reverse");
    // reverse_all flushes the block chain only; the table stays so sensing keeps running.
    assert!(!ruleset().contains("drop"), "block rules cleared");
}

#[test]
fn reverse_all_clears_blocks_but_keeps_the_sense_chain() {
    if reexec_in_netns("reverse_all_clears_blocks_but_keeps_the_sense_chain") {
        return;
    }
    use familiar_core::events::ProcessRef;
    use familiar_core::policy::ProposedAction;
    use familiar_linux::{LinuxActuators, nft};
    use familiar_platform::Actuators;
    use std::process::Command;

    let ruleset = || -> String {
        String::from_utf8(
            Command::new("nft")
                .args(["list", "ruleset"])
                .output()
                .unwrap()
                .stdout,
        )
        .unwrap()
    };

    let mut act = LinuxActuators::new("/sys/fs/cgroup").expect("actuators"); // ensures the table
    nft::install_queue_rule(0).expect("sense chain"); // the NFQUEUE divert lives in the same table
    act.apply(&ProposedAction::BlockOutbound {
        process: ProcessRef {
            pid: 7,
            exe: "/x".into(),
        },
        dst_ip: "203.0.113.9".into(),
        dst_port: 443,
    })
    .expect("block");
    assert!(ruleset().contains("drop"), "block installed");

    act.reverse_all().expect("reverse_all");
    let rs = ruleset();
    assert!(!rs.contains("drop"), "blocks cleared:\n{rs}");
    assert!(
        rs.contains("queue"),
        "the sense (NFQUEUE) chain must survive reverse_all:\n{rs}"
    );
}

#[test]
fn block_two_then_unblock_one_leaves_the_other() {
    if reexec_in_netns("block_two_then_unblock_one_leaves_the_other") {
        return;
    }
    use familiar_linux::nft;
    use std::net::Ipv4Addr;

    nft::ensure_table().expect("table");
    let a: Ipv4Addr = "203.0.113.9".parse().unwrap();
    let b: Ipv4Addr = "198.51.100.4".parse().unwrap();
    nft::block_outbound(a, 443).expect("block a");
    nft::block_outbound(b, 8443).expect("block b");
    let rs = ruleset();
    assert!(
        rs.contains("203.0.113.9") && rs.contains("198.51.100.4"),
        "both blocked:\n{rs}"
    );

    nft::unblock_outbound(a, 443).expect("unblock a");
    let rs = ruleset();
    assert!(!rs.contains("203.0.113.9"), "a's drop must be gone:\n{rs}");
    assert!(rs.contains("198.51.100.4"), "b's drop must remain:\n{rs}");

    // Unblocking a non-existent rule is an error, not a silent success.
    assert!(
        nft::unblock_outbound(a, 443).is_err(),
        "second unblock has nothing to remove"
    );
}
