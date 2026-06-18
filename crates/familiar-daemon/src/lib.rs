#![forbid(unsafe_code)]
//! familiar-daemon library surface: config, persistence, the file-read source,
//! and the run loop. The binary (`main.rs`) is a thin wrapper; the integration
//! tests drive these modules directly.

pub mod config;
pub mod control;
pub mod filereads;
pub mod persistence;
pub mod run;
