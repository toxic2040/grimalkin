use crate::Timestamp;
use crate::policy::Detection;
use std::collections::BTreeMap;

pub type RequestId = u64;

/// A pending request for the human to authorize an action.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PermissionRequest {
    pub id: RequestId,
    pub created_at: Timestamp,
    pub timeout_ms: u64,
    pub detection: Detection,
}

/// The resolution of a permission request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PermissionOutcome {
    Granted,
    Denied,
    TimedOut,
}

impl PermissionOutcome {
    /// Only an explicit grant permits the action. A timeout is a denial.
    pub fn permits_action(self) -> bool {
        matches!(self, PermissionOutcome::Granted)
    }
}

/// Tracks open permission requests. Deterministic: ids are a monotonic counter,
/// expiry is computed against a caller-supplied `now` (the core reads no clock).
#[derive(Clone, Debug)]
pub struct PermissionLedger {
    next_id: RequestId,
    open: BTreeMap<RequestId, PermissionRequest>,
}

impl Default for PermissionLedger {
    fn default() -> Self {
        Self::new()
    }
}

impl PermissionLedger {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            open: BTreeMap::new(),
        }
    }

    /// Open a request and return its id.
    pub fn open(
        &mut self,
        created_at: Timestamp,
        timeout_ms: u64,
        detection: Detection,
    ) -> RequestId {
        let id = self.next_id;
        self.next_id += 1;
        self.open.insert(
            id,
            PermissionRequest {
                id,
                created_at,
                timeout_ms,
                detection,
            },
        );
        id
    }

    pub fn is_open(&self, id: RequestId) -> bool {
        self.open.contains_key(&id)
    }

    /// Borrow an open request by id (the supervisor surfaces it to the user).
    pub fn get(&self, id: RequestId) -> Option<&PermissionRequest> {
        self.open.get(&id)
    }

    /// Borrow every still-open request, ordered by id (BTreeMap iteration). A
    /// read-only view for the control deck; it never mutates or expires.
    pub fn open_requests(&self) -> impl Iterator<Item = &PermissionRequest> {
        self.open.values()
    }

    /// Resolve an open request by explicit human decision. Returns the outcome
    /// and the request (so the caller can act on a grant), or None if the id is
    /// unknown or already resolved.
    pub fn resolve(
        &mut self,
        id: RequestId,
        granted: bool,
    ) -> Option<(PermissionOutcome, PermissionRequest)> {
        let req = self.open.remove(&id)?;
        let outcome = if granted {
            PermissionOutcome::Granted
        } else {
            PermissionOutcome::Denied
        };
        Some((outcome, req))
    }

    /// Expire every open request whose deadline has passed. Each expiry resolves
    /// to TimedOut (a denial). Returns the expired requests.
    pub fn expire_due(&mut self, now: Timestamp) -> Vec<(PermissionOutcome, PermissionRequest)> {
        let due: Vec<RequestId> = self
            .open
            .iter()
            .filter(|(_, r)| now.saturating_sub(r.created_at) >= r.timeout_ms)
            .map(|(id, _)| *id)
            .collect();
        due.into_iter()
            .map(|id| {
                (
                    PermissionOutcome::TimedOut,
                    self.open.remove(&id).expect("listed as due"),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::ProcessRef;
    use crate::policy::{Confidence, DetectionKind, ProposedAction};
    use proptest::prelude::*;

    fn sample(at: Timestamp) -> Detection {
        Detection {
            at,
            kind: DetectionKind::ExfilSuspected,
            confidence: Confidence(90),
            proposed: ProposedAction::BlockOutbound {
                process: ProcessRef {
                    pid: 7,
                    exe: "/usr/bin/curl".into(),
                },
                dst_ip: "203.0.113.9".into(),
                dst_port: 443,
            },
            rationale: "x".into(),
        }
    }

    #[test]
    fn only_a_grant_permits_action() {
        assert!(PermissionOutcome::Granted.permits_action());
        assert!(!PermissionOutcome::Denied.permits_action());
        assert!(!PermissionOutcome::TimedOut.permits_action());
    }

    #[test]
    fn open_then_resolve_returns_the_request() {
        let mut led = PermissionLedger::new();
        let id = led.open(100, 5_000, sample(100));
        assert!(led.is_open(id));
        let (outcome, req) = led.resolve(id, true).expect("open");
        assert_eq!(outcome, PermissionOutcome::Granted);
        assert_eq!(req.id, id);
        assert!(!led.is_open(id));
        assert!(led.resolve(id, true).is_none()); // already resolved
    }

    #[test]
    fn unknown_and_resolved_ids_resolve_and_get_to_none() {
        let mut led = PermissionLedger::new();
        assert!(
            led.resolve(999, true).is_none(),
            "unknown id cannot be resolved"
        );
        assert!(led.get(999).is_none(), "unknown id cannot be fetched");
        let id = led.open(100, 5_000, sample(100));
        assert!(led.get(id).is_some());
        led.resolve(id, true).expect("open");
        assert!(led.get(id).is_none(), "a resolved id is gone");
    }

    #[test]
    fn open_requests_lists_every_unresolved_request() {
        let mut led = PermissionLedger::new();
        let a = led.open(100, 5_000, sample(100));
        let b = led.open(200, 5_000, sample(200));
        led.resolve(a, true).expect("resolve a");
        let ids: Vec<RequestId> = led.open_requests().map(|r| r.id).collect();
        assert_eq!(ids, vec![b], "only the unresolved request remains");
    }

    proptest! {
        /// §8 invariant: a request never survives its deadline, and a timeout is
        /// a denial. Before the deadline it stays open; at/after, it is TimedOut.
        #[test]
        fn timeout_resolves_to_deny(
            created in 0u64..1_000_000,
            timeout in 1u64..100_000,
            delta in 0u64..200_000,
        ) {
            let mut led = PermissionLedger::new();
            let id = led.open(created, timeout, sample(created));
            let expired = led.expire_due(created + delta);
            if delta >= timeout {
                prop_assert_eq!(expired.len(), 1);
                prop_assert_eq!(expired[0].0, PermissionOutcome::TimedOut);
                prop_assert!(!expired[0].0.permits_action());
                prop_assert!(!led.is_open(id));
            } else {
                prop_assert!(expired.is_empty());
                prop_assert!(led.is_open(id));
            }
        }
    }
}
