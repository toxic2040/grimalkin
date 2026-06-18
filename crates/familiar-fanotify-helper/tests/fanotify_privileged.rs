//! Privileged integration test for the fanotify helper's unsafe path. Skipped
//! unless FAMILIAR_PRIVILEGED_ACCEPTANCE=1 (set by scripts/run-privileged-acceptance.sh,
//! which provides CAP_SYS_ADMIN). Verifies that reading a watched file produces a
//! FileReadEvent on the helper's socket — the one code path that cannot run in
//! the unprivileged netns harness.
use std::io::{BufRead, BufReader};
use std::os::unix::net::UnixListener;
use std::time::Duration;

#[test]
fn helper_emits_fileread_for_a_watched_read() {
    if std::env::var("FAMILIAR_PRIVILEGED_ACCEPTANCE").is_err() {
        eprintln!(
            "SKIP helper_emits_fileread_for_a_watched_read: set FAMILIAR_PRIVILEGED_ACCEPTANCE=1 (needs CAP_SYS_ADMIN)"
        );
        return;
    }

    let base = std::env::temp_dir().join(format!("fam-fanotify-{}", std::process::id()));
    let watched = base.join("watched");
    std::fs::create_dir_all(&watched).unwrap();
    let secret = watched.join("secret");
    std::fs::write(&secret, b"sensitive").unwrap();
    let sock = base.join("helper.sock");
    let _ = std::fs::remove_file(&sock);

    // We play the daemon: bind the socket the helper connects to.
    let listener = UnixListener::bind(&sock).unwrap();

    // Spawn the helper (built by cargo for this crate).
    let bin = env!("CARGO_BIN_EXE_familiar-fanotify-helper");
    let mut helper = std::process::Command::new(bin)
        .arg(&sock)
        .arg(&watched)
        .spawn()
        .expect("spawn helper (run under sudo)");

    let (stream, _) = listener.accept().expect("helper connects");
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut reader = BufReader::new(stream);

    // Give the helper a moment to place its mark, then trigger a read.
    std::thread::sleep(Duration::from_millis(300));
    let _ = std::fs::read(&secret).unwrap();

    let mut line = String::new();
    let got = reader.read_line(&mut line);
    let _ = helper.kill();
    let _ = helper.wait();

    assert!(
        got.is_ok() && !line.trim().is_empty(),
        "expected a FileReadEvent line, got {got:?}"
    );
    let v: serde_json::Value = serde_json::from_str(line.trim()).expect("valid FileReadEvent json");
    assert!(
        v["path"].as_str().unwrap_or_default().contains("secret"),
        "event should name the read file: {line}"
    );
    assert!(
        v["pid"].as_u64().unwrap_or(0) > 0,
        "event should carry the accessing pid"
    );
}
