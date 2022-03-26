use std::collections::VecDeque;

use egui::plot::{HLine, Line, Plot, Value, Values};
use egui::{Color32, Ui};

pub struct Benchmark {
    capacity: usize,
    data: VecDeque<f64>,
    last_elapsed: f64,
}

impl Benchmark {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            data: VecDeque::with_capacity(capacity),
            last_elapsed: 0.0,
        }
    }

    pub fn draw(&self, ui: &mut Ui) {
        let iter = self
            .data
            .iter()
            .enumerate()
            .map(|(i, v)| Value::new(i as f64, *v * 1000.0));
        let curve = Line::new(Values::from_values_iter(iter)).color(Color32::BLUE);
        let target = HLine::new(1000.0 / 60.0).color(Color32::RED);

        ui.label(format!("Time in milliseconds that the gui took to draw: {:.4}", self.last_elapsed));
        Plot::new("plot")
            .view_aspect(2.0)
            .include_y(0)
            .show(ui, |plot_ui| {
                plot_ui.line(curve);
                plot_ui.hline(target)
            });
        ui.label("The red line marks the frametime target for drawing at 60 FPS.");
    }

    pub fn push(&mut self, v: f64) {
        if self.data.len() >= self.capacity {
            self.data.pop_front();
        }
        self.data.push_back(v);
        self.last_elapsed = v;
    }
}
