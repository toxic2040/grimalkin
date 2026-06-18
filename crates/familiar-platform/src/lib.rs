#![forbid(unsafe_code)]
//! familiar-platform — the adapter seam.
//!
//! Traits the daemon implements per OS. v0.1 defines the seam and ships a
//! `testkit` fake adapter; the real Linux adapter is a follow-on plan. The core
//! never names this crate — the dependency flows platform -> core only.

use familiar_core::events::Event;
use familiar_core::permission::PermissionRequest;
use familiar_core::policy::ProposedAction;

#[cfg(feature = "testkit")]
pub mod testkit;

/// A source of normalized events. An adapter polls the OS and returns core
/// events; the core only ever sees `Event`.
pub trait Sensors {
    /// Return any events observed since the last poll (possibly empty).
    fn poll(&mut self) -> Vec<Event>;
}

/// The result of carrying out a reversible action.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActuationOutcome {
    /// A human-readable note for the audit/notify trail (e.g. the firewall rule
    /// handle that can later reverse this).
    pub note: String,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ActuationError {
    #[error("actuator does not support this action")]
    Unsupported,
    #[error("actuation failed: {0}")]
    Failed(String),
}

/// Carries out reversible containment actions. An error degrades to no-action
/// upstream — never to a silent unguarded pass.
pub trait Actuators {
    fn apply(&mut self, action: &ProposedAction) -> Result<ActuationOutcome, ActuationError>;

    /// Reverse a previously-applied action (remove a block, thaw a process).
    /// Reversal can only *reduce* containment; it never installs anything.
    /// Default: unsupported, so existing fakes that never reverse are unaffected.
    fn reverse(&mut self, _action: &ProposedAction) -> Result<ActuationOutcome, ActuationError> {
        Err(ActuationError::Unsupported)
    }
}

/// Surfaces notifications and permission prompts to the user (the UI in the
/// daemon; a capture buffer in tests).
pub trait Notifier {
    fn notify(&mut self, message: &str);
    fn request_permission(&mut self, request: &PermissionRequest);
}

#[cfg(test)]
mod tests {
    use super::*;

    struct OkActuators;
    impl Actuators for OkActuators {
        fn apply(&mut self, _a: &ProposedAction) -> Result<ActuationOutcome, ActuationError> {
            Ok(ActuationOutcome { note: "ok".into() })
        }
    }

    #[test]
    fn actuators_are_usable_as_trait_objects() {
        let mut a: Box<dyn Actuators> = Box::new(OkActuators);
        let outcome = a.apply(&ProposedAction::FreezeProcess { pid: 1 }).unwrap();
        assert_eq!(outcome.note, "ok");
    }
}
