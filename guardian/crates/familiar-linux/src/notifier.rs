//! Surfaces guardian activity. Writes a structured line to stderr (journald
//! captures it under systemd) and, optionally, a best-effort desktop
//! notification. v0.1 has no interactive prompt UI — that is Plan C; here a
//! permission request is logged and surfaced so a human can answer it through
//! the (future) control deck.
use familiar_core::permission::PermissionRequest;
use familiar_platform::Notifier;
use std::process::Command;

pub struct LinuxNotifier {
    desktop: bool,
}

impl LinuxNotifier {
    pub fn new(desktop: bool) -> Self {
        Self { desktop }
    }

    fn desktop_notify(&self, summary: &str, body: &str) {
        if self.desktop {
            // Best-effort; a missing notify-send must never break the daemon.
            let _ = Command::new("notify-send").arg(summary).arg(body).status();
        }
    }
}

impl Notifier for LinuxNotifier {
    fn notify(&mut self, message: &str) {
        eprintln!("[familiar] {message}");
        self.desktop_notify("Familiar", message);
    }

    fn request_permission(&mut self, request: &PermissionRequest) {
        let msg = format!(
            "permission needed (request {}): {} [{:?}] — answer within {} ms",
            request.id, request.detection.rationale, request.detection.proposed, request.timeout_ms
        );
        eprintln!("[familiar] {msg}");
        self.desktop_notify(
            "Familiar — action needs your approval",
            &request.detection.rationale,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use familiar_core::permission::PermissionRequest;
    use familiar_core::policy::{Confidence, Detection, DetectionKind, ProposedAction};

    #[test]
    fn notifier_constructs_and_handles_a_request_without_panicking() {
        let mut n = LinuxNotifier::new(false); // no desktop in tests
        n.notify("contained something");
        let req = PermissionRequest {
            id: 1,
            created_at: 1000,
            timeout_ms: 30_000,
            detection: Detection {
                at: 1000,
                kind: DetectionKind::ExfilSuspected,
                confidence: Confidence(50),
                proposed: ProposedAction::FreezeProcess { pid: 7 },
                rationale: "x".into(),
            },
        };
        n.request_permission(&req); // must not panic
    }
}
