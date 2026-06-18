//! Daemon configuration: JSON on disk, all paths and thresholds explicit.
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DaemonConfig {
    pub sensitive_prefixes: Vec<String>,
    pub established_dsts: Vec<String>,
    pub link_window_ms: u64,
    pub permission_timeout_ms: u64,
    pub queue_num: u16,
    pub tick_ms: u64,
    pub state_dir: PathBuf,
    pub cgroup_root: PathBuf,
    pub helper_socket: PathBuf,
    pub desktop_notify: bool,
    pub control_socket: PathBuf,
    pub operator_uid: u32,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            sensitive_prefixes: vec!["/home".into()], // narrowed by the operator
            established_dsts: Vec::new(),
            link_window_ms: 5_000,
            permission_timeout_ms: 30_000,
            queue_num: 0,
            tick_ms: 200,
            state_dir: PathBuf::from("/var/lib/familiar"),
            cgroup_root: PathBuf::from("/sys/fs/cgroup/familiar.slice"),
            helper_socket: PathBuf::from("/run/familiar/fileread.sock"),
            desktop_notify: false,
            control_socket: PathBuf::from("/run/familiar/control.sock"),
            operator_uid: 1000,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("read config: {0}")]
    Read(String),
    #[error("parse config: {0}")]
    Parse(String),
}

impl DaemonConfig {
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path).map_err(|e| ConfigError::Read(e.to_string()))?;
        serde_json::from_str(&text).map_err(|e| ConfigError::Parse(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_round_trips_through_json() {
        let c = DaemonConfig::default();
        let json = serde_json::to_string(&c).unwrap();
        let back: DaemonConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
        assert_eq!(back.operator_uid, 1000);
        assert_eq!(
            back.control_socket,
            PathBuf::from("/run/familiar/control.sock")
        );
    }
}
