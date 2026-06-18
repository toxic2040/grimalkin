#![forbid(unsafe_code)]
//! familiar-ui — the local control deck. A thin egui front end over the daemon's
//! control socket. It can toggle capabilities, answer prompts, lift blocks, and
//! show the audit chain — nothing else: the protocol has no actuating verb and
//! this crate links no actuator code.
mod deck;

use deck::DeckModel;
use eframe::egui;
use familiar_core::capabilities::CapabilityId;
use familiar_ipc::client::ControlClient;
use familiar_ipc::{ControlRequest, ControlResponse};
use std::path::PathBuf;
use std::time::Duration;

struct App {
    client: Option<ControlClient>,
    socket: PathBuf,
    model: DeckModel,
    pending: Vec<ControlRequest>,
}

impl App {
    fn new(socket: PathBuf) -> Self {
        Self {
            client: ControlClient::connect(&socket).ok(),
            socket,
            model: DeckModel::default(),
            pending: Vec::new(),
        }
    }

    /// Issue one request, refreshing the connection if it dropped.
    fn send(&mut self, req: ControlRequest) -> Option<ControlResponse> {
        if self.client.is_none() {
            self.client = ControlClient::connect(&self.socket).ok();
        }
        let resp = self.client.as_mut()?.request(&req).ok();
        if resp.is_none() {
            self.client = None; // force reconnect next time
        }
        resp
    }

    fn refresh_status(&mut self) {
        match self.send(ControlRequest::GetStatus) {
            Some(ControlResponse::Status(s)) => {
                self.model.status = Some(s);
                self.model.last_error = None;
            }
            Some(ControlResponse::Error(e)) => self.model.last_error = Some(e),
            None => self.model.last_error = Some("daemon not reachable".into()),
            _ => {}
        }
    }
}

impl eframe::App for App {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll the daemon ~2x/sec.
        self.refresh_status();
        // Drain any requests queued during the last ui() call.
        let reqs: Vec<ControlRequest> = self.pending.drain(..).collect();
        for req in reqs {
            self.send(req);
        }
        ctx.request_repaint_after(Duration::from_millis(500));
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        ui.heading("Familiar — Control Deck");
        if let Some(err) = &self.model.last_error {
            ui.colored_label(egui::Color32::RED, format!("⚠ {err}"));
        }
        let status = self.model.status.clone();
        let Some(status) = status else {
            ui.label("waiting for the daemon…");
            return;
        };

        ui.separator();
        ui.label(egui::RichText::new("Capabilities").strong());
        for id in CapabilityId::ALL {
            let on = status
                .capabilities
                .states
                .get(&id)
                .copied()
                .unwrap_or(false);
            let mut v = on;
            if ui.checkbox(&mut v, format!("{id:?}")).changed() {
                self.pending.push(self.model.toggle(id, v));
            }
        }

        ui.separator();
        ui.label(egui::RichText::new("Pending approvals").strong());
        if status.prompts.is_empty() {
            ui.weak("none");
        }
        for p in &status.prompts {
            ui.horizontal(|ui| {
                ui.label(format!(
                    "#{} [{}] {} — {}",
                    p.id, p.confidence, p.proposed, p.rationale
                ));
                if ui.button("Allow").clicked() {
                    self.pending.push(self.model.answer(p.id, true));
                }
                if ui.button("Deny").clicked() {
                    self.pending.push(self.model.answer(p.id, false));
                }
            });
        }

        ui.separator();
        ui.label(egui::RichText::new("Active containment").strong());
        if status.active_blocks.is_empty() {
            ui.weak("none");
        }
        for b in &status.active_blocks {
            ui.horizontal(|ui| {
                ui.label(format!("{}:{}", b.dst_ip, b.dst_port));
                if ui.button("Lift").clicked() {
                    self.pending.push(self.model.unblock(b));
                }
            });
        }

        ui.separator();
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Sensors").strong());
            let chip = |ui: &mut egui::Ui, ok: bool, name: &str| {
                let (c, t) = if ok {
                    (egui::Color32::GREEN, "up")
                } else {
                    (egui::Color32::RED, "DOWN")
                };
                ui.colored_label(c, format!("{name}: {t}"));
            };
            chip(ui, status.network_sensor_ok, "network");
            chip(ui, status.file_sensor_ok, "file");
        });

        ui.separator();
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Audit chain").strong());
            if status.audit_ok {
                ui.colored_label(
                    egui::Color32::GREEN,
                    format!("✔ verified ({} records)", status.audit_len),
                );
            } else {
                ui.colored_label(egui::Color32::RED, "✘ TAMPERED");
            }
        });
        ui.monospace(format!("head {}", status.audit_head));
    }
}

fn main() -> eframe::Result<()> {
    let socket = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/run/familiar/control.sock"));
    eframe::run_native(
        "Familiar Control Deck",
        eframe::NativeOptions::default(),
        Box::new(move |_cc| Ok(Box::new(App::new(socket)))),
    )
}
