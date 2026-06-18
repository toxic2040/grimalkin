//! Freeze/thaw a child process in a cgroup we own (no root). Mirrors the Plan B
//! spike's cgroup_freeze leg.
use std::process::Command;
use std::time::Duration;

#[test]
fn freeze_then_thaw_a_child() {
    let unit = format!("fam-test-freeze-{}", std::process::id());
    let spawned = Command::new("systemd-run")
        .args([
            "--user",
            "--scope",
            &format!("--unit={unit}"),
            "sleep",
            "30",
        ])
        .spawn();
    let mut child = match spawned {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP freeze_then_thaw_a_child: systemd-run unavailable: {e}");
            return;
        }
    };
    std::thread::sleep(Duration::from_millis(600));

    let base = "/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice";
    let scope = format!("{base}/{unit}.scope");
    if !std::path::Path::new(&scope).exists() {
        eprintln!("SKIP freeze_then_thaw_a_child: scope cgroup not found at {scope}");
        let _ = child.kill();
        return;
    }

    let sleep_pid: u32 = String::from_utf8(
        Command::new("pgrep")
            .args(["-n", "-x", "sleep"])
            .output()
            .unwrap()
            .stdout,
    )
    .unwrap()
    .trim()
    .parse()
    .unwrap();

    let freezer = familiar_linux::cgroup::Freezer::new(&scope);
    let handle = freezer.freeze(sleep_pid).expect("freeze");
    let events = std::fs::read_to_string(format!("{handle}/cgroup.events")).unwrap();
    assert!(
        events.lines().any(|l| l == "frozen 1"),
        "should report frozen 1:\n{events}"
    );

    freezer.thaw(sleep_pid).expect("thaw");

    let _ = child.kill();
    let _ = child.wait();
    let _ = Command::new("systemctl")
        .args(["--user", "reset-failed", &format!("{unit}.scope")])
        .status();
}
