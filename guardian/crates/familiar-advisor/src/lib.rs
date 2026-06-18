#![forbid(unsafe_code)]
//! familiar-advisor — implementations of the core `Advisor` trait.
//!
//! v0.1 ships only the null advisor: it explains nothing and never escalates.
//! The spine runs rule-only; this proves the seam without a model.

use familiar_core::advisor::{Advice, Advisor};
use familiar_core::policy::Detection;

/// An advisor that always abstains. The v0.1 default.
#[derive(Clone, Copy, Debug, Default)]
pub struct NullAdvisor;

impl Advisor for NullAdvisor {
    fn assess(&self, _detection: &Detection) -> Advice {
        Advice::none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use familiar_core::advisor::Caution;
    use familiar_core::events::ProcessRef;
    use familiar_core::policy::{Confidence, DetectionKind, ProposedAction};

    #[test]
    fn null_advisor_always_abstains() {
        let d = Detection {
            at: 1,
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
        };
        let advice = NullAdvisor.assess(&d);
        assert_eq!(advice.caution, Caution::NoOpinion);
        assert!(advice.explanation.is_empty());
    }
}
