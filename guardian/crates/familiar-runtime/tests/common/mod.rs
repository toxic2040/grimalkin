#![allow(dead_code)]
use familiar_advisor::NullAdvisor;
use familiar_core::audit::AuditLog;
use familiar_core::capabilities::{CapabilityId, CapabilityRegistry};
use familiar_core::events::{Event, ProcessRef};
use familiar_core::policy::{Engine, ExfilConfig, ExfilDetector};
use familiar_platform::testkit::{CapturingNotifier, FakeSensors, RecordingActuators};
use familiar_runtime::Supervisor;

pub type TestSupervisor =
    Supervisor<FakeSensors, RecordingActuators, CapturingNotifier, NullAdvisor>;

pub fn proc(pid: u32) -> ProcessRef {
    ProcessRef {
        pid,
        exe: "/usr/bin/curl".into(),
    }
}
pub fn read(at: u64, pid: u32) -> Event {
    Event::FileRead {
        at,
        process: proc(pid),
        path: "/home/u/.ssh/id_ed25519".into(),
    }
}
pub fn out(at: u64, pid: u32, ip: &str) -> Event {
    Event::OutboundConn {
        at,
        process: proc(pid),
        dst_ip: ip.into(),
        dst_port: 443,
    }
}

/// Which capabilities to arm. Default: everything on.
pub struct Caps {
    pub sensor_read: bool,
    pub sensor_out: bool,
    pub detector: bool,
    pub actuator_block: bool,
}
impl Default for Caps {
    fn default() -> Self {
        Self {
            sensor_read: true,
            sensor_out: true,
            detector: true,
            actuator_block: true,
        }
    }
}

pub fn engine_with(caps: Caps) -> Engine {
    let mut reg = CapabilityRegistry::new();
    let mut throwaway = AuditLog::new();
    for (cap, on) in [
        (CapabilityId::SensorSensitiveRead, caps.sensor_read),
        (CapabilityId::SensorOutboundConn, caps.sensor_out),
        (CapabilityId::DetectorExfil, caps.detector),
        (CapabilityId::ActuatorBlockConn, caps.actuator_block),
    ] {
        if on {
            reg.set(cap, true, 0, &mut throwaway);
        }
    }
    let det = ExfilDetector::new(ExfilConfig {
        sensitive_prefixes: vec!["/home/u/.ssh".into()],
        ..ExfilConfig::default()
    });
    Engine::new(reg, det)
}

pub fn supervisor(
    engine: Engine,
    sensors: FakeSensors,
    actuators: RecordingActuators,
) -> TestSupervisor {
    Supervisor::new(
        engine,
        sensors,
        actuators,
        CapturingNotifier::default(),
        NullAdvisor,
        30_000,
    )
}
