// Copied from https://github.com/vulkano-rs/vulkano/blob/master/examples/src/bin/triangle.rs
/// Differences:
/// * Set the correct color format for the swapchain
/// * Second renderpass to draw the gui
use std::convert::TryInto;
use std::sync::Arc;
use std::time::Instant;

use egui_vulkano::UpdateTexturesResult;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceExtensions};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{ImageAccess, ImageUsage, SwapchainImage};
use vulkano::instance::Instance;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::render_pass::{Framebuffer, RenderPass, Subpass};
use vulkano::swapchain::{AcquireError, ColorSpace, Swapchain, PresentMode, SwapchainCreationError};
use vulkano::sync::{GpuFuture, FlushError};
use vulkano::{swapchain, Version};
use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Fullscreen, Window, WindowBuilder};

mod future;
use future::FrameEndFuture;

mod benchmark_widget;
use benchmark_widget::Benchmark;



#[derive(Default, Debug, Clone)]
struct Vertex {
    position: [f32; 2],
}
vulkano::impl_vertex!(Vertex, position);

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
            #version 450

            layout(location = 0) in vec2 position;

            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
            }
        "
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
            #version 450

            layout(location = 0) out vec4 f_color;

            void main() {
                f_color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        "
    }
}



fn main() {
    let required_extensions = vulkano_win::required_extensions();
    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        nv_shading_rate_image: true,
        ..DeviceExtensions::none()
    };

    let instance = Instance::new(None, Version::V1_1, &required_extensions, None).unwrap();

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .with_title("egui_vulkano demo")
        .with_fullscreen(Some(Fullscreen::Borderless(None)))
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let (physical, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions) )
        .filter_map(|p| p.queue_families()
                            .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false) )
                            .map(|q| (p, q))
        )
        .min_by_key(|(p, _)| match p.properties().device_type {
                                        PhysicalDeviceType::DiscreteGpu => 0,
                                        PhysicalDeviceType::IntegratedGpu => 1,
                                        PhysicalDeviceType::VirtualGpu => 2,
                                        PhysicalDeviceType::Cpu => 3,
                                        PhysicalDeviceType::Other => 4,
                                    }
        ).unwrap();

    println!(
        "Using device: {} (type: {:?})",
        physical.properties().device_name,
        physical.properties().device_type
    );

    // Solving conflict between attachment_variable_shading_rate and shading_rate_image
    let mut features = physical.supported_features().clone();
    features.shading_rate_image = false;

    let (device, mut queues) = Device::new(
        physical, 
        &features,                
        &physical.required_extensions().union(&device_extensions),
        [(queue_family, 0.5)].iter().cloned(),
    )
    .unwrap();

    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let caps = surface.capabilities(physical).unwrap();
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();

        // Set the swapchain format to Srgb to get correct colors for egui
        assert!(&caps.supported_formats
                     .contains(&(Format::B8G8R8A8_SRGB, ColorSpace::SrgbNonLinear)));
        let format = Format::B8G8R8A8_SRGB;
        let dimensions: [u32; 2] = surface.window().inner_size().into();

        // Select best present mode (optimization)
        let present_mode = if caps.present_modes.mailbox {
            println!("Swapchain mode: Mailbox");
            PresentMode::Mailbox
        } else if caps.present_modes.immediate {
            println!("Swapchain mode: Immediate");
            PresentMode::Immediate
        } else {
            println!("Swapchain mode: Fifo");
            PresentMode::Fifo
        };

        Swapchain::start(device.clone(), surface.clone())
            .num_images(caps.min_image_count)
            .format(format)
            .dimensions(dimensions)
            .usage(ImageUsage::color_attachment())
            .sharing_mode(&queue)
            .composite_alpha(alpha)
            .present_mode(present_mode)
            .build()
            .unwrap()
    };

    let vertex_buffer = {
        CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            [
                Vertex {
                    position: [-0.5, -0.25],
                },
                Vertex {
                    position: [0.0, 0.5],
                },
                Vertex {
                    position: [0.25, -0.1],
                },
            ]
            .iter()
            .cloned(),
        )
        .unwrap()
    };

    let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();

    let render_pass = vulkano::ordered_passes_renderpass!(
        device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.format(),
                samples: 1,
            }
        },
        passes: [
            { color: [color], depth_stencil: {}, input: [] },
            { color: [color], depth_stencil: {}, input: [] } // Create a second renderpass to draw egui
        ]
    )
    .unwrap();

    let pipeline = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .render_pass(Subpass::from(render_pass.clone().into(), 0).unwrap())
        .build(device.clone())
        .unwrap();

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };

    let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut viewport);

    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(FrameEndFuture::now(device.clone()));

    //Set up everything need to draw the gui
    let window = surface.window();
    let egui_ctx = egui::Context::default();
    let mut egui_winit = egui_winit::State::new(4096, window);

    let mut egui_painter = egui_vulkano::Painter::new(
        device.clone(),
        queue.clone(),
        Subpass::from(render_pass.clone(), 1).unwrap(),
    )
    .unwrap();

    //Set up some window to look at for the test

    let mut egui_test = egui_demo_lib::ColorTest::default();
    let mut demo_windows = egui_demo_lib::DemoWindows::default();
    let mut egui_bench = Benchmark::new(1000);
    let mut my_texture = egui_ctx.load_texture("my_texture", egui::ColorImage::example());

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                recreate_swapchain = true;
            }
            Event::WindowEvent { event, .. } => {
                let egui_consumed_event = egui_winit.on_event(&egui_ctx, &event);
                if !egui_consumed_event {
                    // do your own event handling here
                };
            }
            Event::RedrawEventsCleared => {
                previous_frame_end
                    .as_mut()
                    .unwrap()
                    .as_mut()
                    .cleanup_finished();

                if recreate_swapchain {
                    let dimensions: [u32; 2] = surface.window().inner_size().into();
                    let (new_swapchain, new_images) =
                        match swapchain.recreate().dimensions(dimensions).build() {
                            Ok(r) => r,
                            Err(SwapchainCreationError::UnsupportedDimensions) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                        };

                    swapchain = new_swapchain;
                    framebuffers = window_size_dependent_setup(
                        &new_images,
                        render_pass.clone(),
                        &mut viewport,
                    );
                    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];
                    recreate_swapchain = false;
                }

                let (image_num, suboptimal, acquire_future) =
                    match swapchain::acquire_next_image(swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("Failed to acquire next image: {:?}", e),
                    };

                if suboptimal {
                    recreate_swapchain = true;
                }

                let clear_values = vec![[0.0, 0.0, 1.0, 1.0].into()];
                let mut builder = AutoCommandBufferBuilder::primary(
                    device.clone(),
                    queue.family(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                let frame_start = Instant::now();
                egui_ctx.begin_frame(egui_winit.take_egui_input(surface.window()));
                demo_windows.ui(&egui_ctx);

                egui::Window::new("Color test")
                    .vscroll(true)
                    .show(&egui_ctx, |ui| {
                        egui_test.ui(ui);
                    });

                egui::Window::new("Settings").show(&egui_ctx, |ui| {
                    egui_ctx.settings_ui(ui);
                });

                egui::Window::new("Benchmark")
                    .default_height(600.0)
                    .show(&egui_ctx, |ui| {
                        egui_bench.draw(ui);
                    });

                egui::Window::new("Texture test").show(&egui_ctx, |ui| {
                    ui.image(my_texture.id(), (200.0, 200.0));
                    if ui.button("Reload texture").clicked() {
                        // previous TextureHandle is dropped, causing egui to free the texture:
                        my_texture = egui_ctx.load_texture("my_texture", egui::ColorImage::example());
                    }
                });

                // Get the shapes from egui
                let egui_output = egui_ctx.end_frame();
                let platform_output = egui_output.platform_output;
                egui_winit.handle_platform_output(surface.window(), &egui_ctx, platform_output);

                let result = egui_painter
                    .update_textures(egui_output.textures_delta, &mut builder)
                    .expect("egui texture error");

                let wait_for_last_frame = result == UpdateTexturesResult::Changed;

                // Do your usual rendering
                builder
                    .begin_render_pass(
                        framebuffers[image_num].clone(),
                        SubpassContents::Inline,
                        clear_values,
                    )
                    .unwrap()
                    .set_viewport(0, [viewport.clone()])
                    .bind_pipeline_graphics(pipeline.clone())
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .draw(vertex_buffer.len().try_into().unwrap(), 1, 0, 0)
                    .unwrap(); // Don't end the render pass yet

                // Build your gui

                // Automatically start the next render subpass and draw the gui
                let size = surface.window().inner_size();
                let sf: f32 = surface.window().scale_factor() as f32;
                egui_painter
                    .draw(
                        &mut builder,
                        [(size.width as f32) / sf, (size.height as f32) / sf],
                        &egui_ctx,
                        egui_output.shapes,
                    )
                    .unwrap();

                egui_bench.push(frame_start.elapsed().as_secs_f64());

                // End the render pass as usual
                builder.end_render_pass().unwrap();

                let command_buffer = builder.build().unwrap();

                if wait_for_last_frame {
                    if let Some(FrameEndFuture::FenceSignalFuture(ref mut f)) = previous_frame_end {
                        f.wait(None).unwrap();
                    }
                }

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .get()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(FrameEndFuture::FenceSignalFuture(future));
                    }
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(FrameEndFuture::now(device.clone()));
                    }
                    Err(e) => {
                        println!("Failed to flush future: {:?}", e);
                        previous_frame_end = Some(FrameEndFuture::now(device.clone()));
                    }
                }
            }
            _ => (),
        }
    });
}

fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> Vec<Arc<Framebuffer>> {
    let dimensions = images[0].dimensions().width_height();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

    let framebuffers = images
        .iter()
        .map(|image| {
            let view = ImageView::new(image.clone()).unwrap();
            Framebuffer::start(render_pass.clone())
                .add(view)
                .unwrap()
                .build()
                .unwrap()
        })
        .collect::<Vec<_>>();
    framebuffers
}