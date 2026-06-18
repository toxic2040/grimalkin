//! The control deck's pure model: it holds the latest status the daemon sent and
//! turns user gestures into `ControlRequest`s. It contains no actuator types and
//! no socket I/O — the eframe App owns the client and feeds this model snapshots.
use familiar_core::capabilities::CapabilityId;
use familiar_ipc::{BlockDto, ControlRequest, StatusSnapshot};

#[derive(Default)]
pub struct DeckModel {
    pub status: Option<StatusSnapshot>,
    pub last_error: Option<String>,
}

impl DeckModel {
    pub fn toggle(&self, id: CapabilityId, enabled: bool) -> ControlRequest {
        ControlRequest::SetCapability { id, enabled }
    }
    pub fn answer(&self, id: u64, granted: bool) -> ControlRequest {
        ControlRequest::AnswerPrompt { id, granted }
    }
    pub fn unblock(&self, b: &BlockDto) -> ControlRequest {
        ControlRequest::Unblock {
            dst_ip: b.dst_ip.clone(),
            dst_port: b.dst_port,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gestures_map_to_the_narrow_requests() {
        let m = DeckModel::default();
        assert_eq!(
            m.toggle(CapabilityId::DetectorExfil, true),
            ControlRequest::SetCapability {
                id: CapabilityId::DetectorExfil,
                enabled: true
            }
        );
        assert_eq!(
            m.answer(5, false),
            ControlRequest::AnswerPrompt {
                id: 5,
                granted: false
            }
        );
        let b = BlockDto {
            dst_ip: "203.0.113.9".into(),
            dst_port: 443,
        };
        assert_eq!(
            m.unblock(&b),
            ControlRequest::Unblock {
                dst_ip: "203.0.113.9".into(),
                dst_port: 443
            }
        );
    }
}
