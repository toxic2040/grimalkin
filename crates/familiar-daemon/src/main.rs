#![forbid(unsafe_code)]
//! familiar-daemon — the least-privilege guardian process (CAP_NET_ADMIN only).
use familiar_daemon::config::DaemonConfig;
use familiar_daemon::run;
use std::path::PathBuf;

fn main() {
    let cfg = match std::env::args().nth(1).map(PathBuf::from) {
        Some(p) => DaemonConfig::load(&p).unwrap_or_else(|e| {
            eprintln!("[familiar] config load failed ({e}); using defaults");
            DaemonConfig::default()
        }),
        None => DaemonConfig::default(),
    };
    run::main_loop(cfg);
}
