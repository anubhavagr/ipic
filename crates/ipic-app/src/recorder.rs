//! Microphone capture for spoken queries: cpal input stream collecting f32
//! frames into a shared buffer, converted to 16 kHz mono on stop.

use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};

pub struct AudioRecorder {
    stream: cpal::Stream,
    samples: Arc<Mutex<Vec<f32>>>,
    sample_rate: u32,
}

impl AudioRecorder {
    /// Starts recording from the default input device.
    pub fn start() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| anyhow!("no microphone available"))?;
        let supported = device.default_input_config()?;
        let sample_rate = supported.sample_rate();
        let channels = supported.channels().max(1) as usize;
        let sample_format = supported.sample_format();
        let config: cpal::StreamConfig = supported.into();
        let samples = Arc::new(Mutex::new(Vec::<f32>::new()));

        let collected = Arc::clone(&samples);
        let error_callback = |error| eprintln!("microphone error: {error}");
        let stream = match sample_format {
            cpal::SampleFormat::F32 => device.build_input_stream(
                config,
                move |data: &[f32], _| push_frames(&collected, data, channels),
                error_callback,
                None,
            )?,
            cpal::SampleFormat::I16 => device.build_input_stream(
                config,
                move |data: &[i16], _| {
                    let converted: Vec<f32> = data.iter().map(|sample| *sample as f32 / 32768.0).collect();
                    push_frames(&collected, &converted, channels)
                },
                error_callback,
                None,
            )?,
            other => return Err(anyhow!("unsupported microphone sample format: {other:?}")),
        };
        stream.play()?;
        Ok(Self { stream, samples, sample_rate })
    }

    /// Stops recording and returns 16 kHz mono PCM ready for whisper.
    pub fn stop(self) -> Vec<f32> {
        drop(self.stream);
        let mut samples = self.samples.lock().unwrap().clone();
        if self.sample_rate != 16_000 {
            samples = ipic_rag::extract::resample_linear(samples, self.sample_rate as usize, 16_000);
        }
        samples
    }

    pub fn captured_seconds(&self) -> f32 {
        self.samples.lock().unwrap().len() as f32 / self.sample_rate as f32
    }
}

fn push_frames(collected: &Mutex<Vec<f32>>, data: &[f32], channels: usize) {
    let mut buffer = collected.lock().unwrap();
    // Downmix to mono by averaging channels within each frame.
    buffer.extend(data.chunks(channels).map(|frame| frame.iter().sum::<f32>() / frame.len() as f32));
}
