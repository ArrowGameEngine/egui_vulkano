use std::sync::Arc;

use vulkano::device::Device;
use vulkano::sync::{FenceSignalFuture, GpuFuture};
use vulkano::sync;

pub enum FrameEndFuture<F: GpuFuture + 'static> {
    FenceSignalFuture(FenceSignalFuture<F>),
    BoxedFuture(Box<dyn GpuFuture>),
}

impl<F: GpuFuture> FrameEndFuture<F> {
    pub fn now(device: Arc<Device>) -> Self {
        Self::BoxedFuture(sync::now(device).boxed())
    }

    pub fn get(self) -> Box<dyn GpuFuture> {
        match self {
            FrameEndFuture::FenceSignalFuture(f) => f.boxed(),
            FrameEndFuture::BoxedFuture(f) => f,
        }
    }
}

impl<F: GpuFuture> AsMut<dyn GpuFuture> for FrameEndFuture<F> {
    fn as_mut(&mut self) -> &mut (dyn GpuFuture + 'static) {
        match self {
            FrameEndFuture::FenceSignalFuture(f) => f,
            FrameEndFuture::BoxedFuture(f) => f,
        }
    }
}