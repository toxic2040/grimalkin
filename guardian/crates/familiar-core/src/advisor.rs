use crate::policy::{Detection, Disposition};

/// How much more cautious the advisor wants the harness to be. The advisor can
/// only ever move toward caution — never away. This is the structural form of
/// "the advisor never holds a gate" (spec §4.2, §7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Caution {
    /// No escalation.
    NoOpinion,
    /// Route an otherwise-autonomous action to the human instead.
    Heighten,
}

/// The advisor's contribution to a decision: a human-readable explanation and,
/// at most, a request to be more cautious.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Advice {
    pub explanation: String,
    pub caution: Caution,
}

impl Advice {
    /// The advice a null/absent advisor gives.
    pub fn none() -> Self {
        Advice {
            explanation: String::new(),
            caution: Caution::NoOpinion,
        }
    }
}

/// The interface the core may call for fuzzy judgment or explanation. An advisor
/// can inform a decision; it can never hold a gate.
pub trait Advisor {
    fn assess(&self, detection: &Detection) -> Advice;
}

/// Apply the advisor's caution to a rule-derived disposition. Monotonic toward
/// caution: the only move is `ActAutonomously -> RequirePermission`. The advisor
/// can never authorize an action and never unilaterally deny one.
pub fn apply_caution(disposition: Disposition, caution: Caution) -> Disposition {
    match (disposition, caution) {
        (Disposition::ActAutonomously, Caution::Heighten) => Disposition::RequirePermission,
        (d, _) => d,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn heighten_routes_an_autonomous_action_to_the_human() {
        assert_eq!(
            apply_caution(Disposition::ActAutonomously, Caution::Heighten),
            Disposition::RequirePermission
        );
    }

    #[test]
    fn no_opinion_changes_nothing() {
        assert_eq!(
            apply_caution(Disposition::ActAutonomously, Caution::NoOpinion),
            Disposition::ActAutonomously
        );
    }

    fn dispositions() -> impl Strategy<Value = Disposition> {
        prop_oneof![
            Just(Disposition::ActAutonomously),
            Just(Disposition::RequirePermission),
            Just(Disposition::Deny),
        ]
    }
    fn cautions() -> impl Strategy<Value = Caution> {
        prop_oneof![Just(Caution::NoOpinion), Just(Caution::Heighten)]
    }

    proptest! {
        /// The advisor never holds a gate: it can neither manufacture an
        /// autonomous action nor convert a human-ask into a unilateral deny, nor
        /// relax a deny. Its only move is Act -> RequirePermission.
        #[test]
        fn advisor_never_moves_a_gate(d in dispositions(), c in cautions()) {
            let out = apply_caution(d, c);
            if d != Disposition::ActAutonomously {
                prop_assert_ne!(out, Disposition::ActAutonomously);
            }
            if d == Disposition::RequirePermission {
                prop_assert_eq!(out, Disposition::RequirePermission);
            }
            if d == Disposition::Deny {
                prop_assert_eq!(out, Disposition::Deny);
            }
        }
    }
}
