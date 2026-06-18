//! Fake adapters for driving the runtime in tests without an OS.

use std::collections::VecDeque;

use familiar_core::events::Event;
use familiar_core::permission::PermissionRequest;
use familiar_core::policy::ProposedAction;

use crate::{ActuationError, ActuationOutcome, Actuators, Notifier, Sensors};

/// Sensors that replay scripted batches of events, one batch per poll.
#[derive(Clone, Debug, Default)]
pub struct FakeSensors {
    batches: VecDeque<Vec<Event>>,
}

impl FakeSensors {
    pub fn new(batches: Vec<Vec<Event>>) -> Self {
        Self {
            batches: batches.into(),
        }
    }
    /// True once every scripted batch has been polled.
    pub fn is_drained(&self) -> bool {
        self.batches.is_empty()
    }
}

impl Sensors for FakeSensors {
    fn poll(&mut self) -> Vec<Event> {
        self.batches.pop_front().unwrap_or_default()
    }
}

/// Actuators that record every applied action and can be forced to fail.
#[derive(Clone, Debug, Default)]
pub struct RecordingActuators {
    pub applied: Vec<ProposedAction>,
    pub reversed: Vec<ProposedAction>,
    pub fail: bool,
}

impl RecordingActuators {
    pub fn failing() -> Self {
        Self {
            applied: Vec::new(),
            fail: true,
            ..Default::default()
        }
    }
}

impl Actuators for RecordingActuators {
    fn apply(&mut self, action: &ProposedAction) -> Result<ActuationOutcome, ActuationError> {
        if self.fail {
            return Err(ActuationError::Failed("injected".into()));
        }
        self.applied.push(action.clone());
        Ok(ActuationOutcome {
            note: format!("applied {action:?}"),
        })
    }

    fn reverse(&mut self, action: &ProposedAction) -> Result<ActuationOutcome, ActuationError> {
        if self.fail {
            return Err(ActuationError::Failed("injected".into()));
        }
        self.reversed.push(action.clone());
        Ok(ActuationOutcome {
            note: format!("reversed {action:?}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recording_actuator_records_reversals_separately() {
        let mut a = RecordingActuators::default();
        let block = ProposedAction::BlockOutbound {
            process: familiar_core::events::ProcessRef {
                pid: 7,
                exe: "/x".into(),
            },
            dst_ip: "203.0.113.9".into(),
            dst_port: 443,
        };
        a.apply(&block).unwrap();
        a.reverse(&block).unwrap();
        assert_eq!(a.applied.len(), 1);
        assert_eq!(a.reversed.len(), 1);
        assert_eq!(a.reversed[0], block);
    }
}

/// Notifier that captures messages and permission requests.
#[derive(Clone, Debug, Default)]
pub struct CapturingNotifier {
    pub messages: Vec<String>,
    pub requests: Vec<PermissionRequest>,
}

impl Notifier for CapturingNotifier {
    fn notify(&mut self, message: &str) {
        self.messages.push(message.to_string());
    }
    fn request_permission(&mut self, request: &PermissionRequest) {
        self.requests.push(request.clone());
    }
}
