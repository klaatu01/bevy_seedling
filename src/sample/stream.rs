//! Live PCM audio streaming support.
//!
//! This module provides types for creating live audio streams that can be fed
//! PCM samples at runtime. Suitable for voice chat, network audio, procedural
//! audio, microphone passthrough, and other real-time audio use cases.
//!
//! # Usage
//!
//! Create a [`LiveAudioStream`], add it as an [`AudioSample`] asset, and play
//! it with a [`SamplePlayer`]. Push PCM data from any Bevy system.
//!
//! ```no_run
//! use bevy::prelude::*;
//! use bevy_seedling::prelude::*;
//!
//! #[derive(Resource)]
//! struct MyStream(LiveAudioStream);
//!
//! fn setup(mut commands: Commands, mut assets: ResMut<Assets<AudioSample>>) {
//!     let stream = LiveAudioStream::new(StreamSettings::default());
//!     let handle = assets.add(stream.as_sample());
//!     commands.spawn((
//!         SamplePlayer::new(handle),
//!         PlaybackSettings::default().preserve(),
//!     ));
//!     commands.insert_resource(MyStream(stream));
//! }
//!
//! fn push_audio(state: Res<MyStream>) {
//!     // Generate or receive PCM samples...
//!     let samples = vec![0.0_f32; 480];
//!     state.0.push_mono(&samples);
//! }
//! ```
//!
//! [`AudioSample`]: super::AudioSample
//! [`SamplePlayer`]: super::SamplePlayer

use super::AudioSample;
use core::{
    cell::UnsafeCell,
    num::{NonZeroU32, NonZeroUsize},
    ops::Range,
};
use firewheel::sample_resource::{SampleResource, SampleResourceInfo};
use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
};

/// Number of audio channels for a live stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamChannels {
    /// Single channel. Push with [`LiveAudioStream::push_mono`].
    Mono,
    /// Two channels, interleaved. Push with [`LiveAudioStream::push_stereo_interleaved`].
    Stereo,
}

/// Settings for creating a [`LiveAudioStream`].
#[derive(Debug, Clone)]
pub struct StreamSettings {
    /// Sample rate of the input PCM data in Hz (default: 48000).
    pub sample_rate: u32,
    /// Number of channels (default: Mono).
    pub channels: StreamChannels,
    /// Maximum buffered duration in milliseconds. Older samples are dropped on overflow (default: 200).
    pub max_buffer_duration_ms: u32,
    /// Minimum buffered duration in milliseconds before playback starts (default: 40).
    pub start_threshold_ms: u32,
}

impl Default for StreamSettings {
    fn default() -> Self {
        Self {
            sample_rate: 48_000,
            channels: StreamChannels::Mono,
            max_buffer_duration_ms: 200,
            start_threshold_ms: 40,
        }
    }
}

/// Statistics about a live audio stream's buffer state.
#[derive(Debug, Clone, Copy)]
pub struct StreamStats {
    /// Number of audio frames currently buffered.
    pub buffered_frames: usize,
    /// Approximate milliseconds of audio currently buffered.
    pub buffered_ms: f64,
    /// Number of times the audio thread found the buffer empty (underflows).
    pub underflow_count: u64,
    /// Number of samples dropped because the buffer was full (overflows).
    pub overflow_count: u64,
}

// ---------------------------------------------------------------------------
// SharedStreamState
// ---------------------------------------------------------------------------

/// Shared state between the producer (Bevy systems) and consumer (audio thread).
pub(crate) struct SharedStreamState {
    buffer: Mutex<VecDeque<f32>>,
    max_buffer_samples: usize,
    start_threshold_samples: usize,
    sample_rate: u32,
    channel_count: usize,

    /// Set to true once enough samples are buffered to begin playback.
    started: AtomicBool,

    underflow_count: AtomicU64,
    overflow_count: AtomicU64,
    buffered_samples: AtomicUsize,
}

impl SharedStreamState {
    fn new(
        max_buffer_samples: usize,
        start_threshold_samples: usize,
        sample_rate: u32,
        channel_count: usize,
    ) -> Self {
        Self {
            buffer: Mutex::new(VecDeque::with_capacity(max_buffer_samples)),
            max_buffer_samples,
            start_threshold_samples,
            sample_rate,
            channel_count,
            started: AtomicBool::new(false),
            underflow_count: AtomicU64::new(0),
            overflow_count: AtomicU64::new(0),
            buffered_samples: AtomicUsize::new(0),
        }
    }

    fn push_samples(&self, samples: &[f32]) {
        let mut buffer = self.buffer.lock().expect("stream buffer poisoned");
        let available = self.max_buffer_samples.saturating_sub(buffer.len());

        if samples.len() > available {
            // Drop oldest samples to make room, aligned to frame boundaries.
            let to_drop = samples.len() - available;
            let frame_aligned =
                (to_drop + self.channel_count - 1) / self.channel_count * self.channel_count;
            let actual_drop = frame_aligned.min(buffer.len());
            buffer.drain(..actual_drop);
            self.overflow_count
                .fetch_add(actual_drop as u64, Ordering::Relaxed);
        }

        buffer.extend(samples);
        let len = buffer.len();
        self.buffered_samples.store(len, Ordering::Relaxed);

        if !self.started.load(Ordering::Relaxed) && len >= self.start_threshold_samples {
            self.started.store(true, Ordering::Release);
        }
    }

    /// Drain up to `max_samples` raw f32 samples from the shared buffer.
    /// Returns the number of samples actually drained.
    fn drain_into(&self, dest: &mut Vec<f32>, max_samples: usize) -> usize {
        let mut buffer = self.buffer.lock().expect("stream buffer poisoned");
        let n = max_samples.min(buffer.len());
        // Align to channel_count for stereo correctness.
        let n = n / self.channel_count * self.channel_count;
        dest.extend(buffer.drain(..n));
        let remaining = buffer.len();
        self.buffered_samples.store(remaining, Ordering::Relaxed);
        n
    }
}

// ---------------------------------------------------------------------------
// StreamingSampleResource
// ---------------------------------------------------------------------------

/// A [`SampleResource`] that pulls PCM data from a shared ringbuffer.
///
/// The sampler node calls [`fill_buffers`](SampleResource::fill_buffers) on
/// the audio thread each block. This implementation ignores the `start_frame`
/// parameter and reads sequentially from the ringbuffer, filling silence on
/// underflow.
pub(crate) struct StreamingSampleResource {
    shared: Arc<SharedStreamState>,
    channel_count: NonZeroUsize,
    sample_rate: NonZeroU32,

    // Local prefetch buffer, only accessed from the audio thread.
    // Safety: The sampler node guarantees single-threaded access to
    // `fill_buffers`. We use UnsafeCell to avoid requiring `&mut self`.
    local_buf: UnsafeCell<Vec<f32>>,
    local_pos: UnsafeCell<usize>,
}

// Safety: `local_buf` and `local_pos` are only accessed from `fill_buffers`,
// which the sampler node calls from a single audio thread.
unsafe impl Send for StreamingSampleResource {}
unsafe impl Sync for StreamingSampleResource {}

impl StreamingSampleResource {
    fn new(shared: Arc<SharedStreamState>) -> Self {
        let channel_count =
            NonZeroUsize::new(shared.channel_count).expect("channel_count must be > 0");
        let sample_rate = NonZeroU32::new(shared.sample_rate).expect("sample_rate must be > 0");

        Self {
            shared,
            channel_count,
            sample_rate,
            local_buf: UnsafeCell::new(Vec::new()),
            local_pos: UnsafeCell::new(0),
        }
    }

    /// Pre-fetch enough samples from the shared buffer for a block.
    ///
    /// # Safety
    ///
    /// Must only be called from the audio thread (single caller).
    unsafe fn prefetch(&self, frames_needed: usize) {
        let local_buf = unsafe { &mut *self.local_buf.get() };
        let local_pos = unsafe { &mut *self.local_pos.get() };

        // Compact: move unconsumed samples to front.
        if *local_pos > 0 {
            local_buf.drain(..*local_pos);
            *local_pos = 0;
        }

        let samples_needed = frames_needed * self.channel_count.get();
        let have = local_buf.len();
        if have < samples_needed {
            let need = samples_needed - have;
            self.shared.drain_into(local_buf, need);
        }
    }
}

impl SampleResourceInfo for StreamingSampleResource {
    fn num_channels(&self) -> NonZeroUsize {
        self.channel_count
    }

    fn len_frames(&self) -> u64 {
        // Unbounded — the sampler never considers this stream "finished".
        u64::MAX
    }

    fn sample_rate(&self) -> Option<NonZeroU32> {
        Some(self.sample_rate)
    }
}

impl SampleResource for StreamingSampleResource {
    fn fill_buffers(
        &self,
        buffers: &mut [&mut [f32]],
        buffer_range: Range<usize>,
        _start_frame: u64,
    ) {
        let frames = buffer_range.end - buffer_range.start;
        if frames == 0 {
            return;
        }

        let channels = self.channel_count.get();
        let started = self.shared.started.load(Ordering::Acquire);

        if !started {
            // Not enough data buffered yet — output silence.
            for buf in buffers.iter_mut().take(channels) {
                buf[buffer_range.clone()].fill(0.0);
            }
            return;
        }

        // Safety: only called from the audio thread.
        unsafe { self.prefetch(frames) };

        let local_buf = unsafe { &*self.local_buf.get() };
        let local_pos = unsafe { &mut *self.local_pos.get() };

        if channels == 1 {
            let buf = &mut buffers[0][buffer_range.clone()];
            for sample in buf.iter_mut() {
                if *local_pos < local_buf.len() {
                    *sample = local_buf[*local_pos];
                    *local_pos += 1;
                } else {
                    *sample = 0.0;
                    self.shared
                        .underflow_count
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
        } else if channels == 2 && buffers.len() >= 2 {
            let (first, rest) = buffers.split_first_mut().unwrap();
            let buf0 = &mut first[buffer_range.clone()];
            let buf1 = &mut rest[0][buffer_range.clone()];

            for (s0, s1) in buf0.iter_mut().zip(buf1.iter_mut()) {
                if *local_pos + 1 < local_buf.len() {
                    *s0 = local_buf[*local_pos];
                    *s1 = local_buf[*local_pos + 1];
                    *local_pos += 2;
                } else {
                    *s0 = 0.0;
                    *s1 = 0.0;
                    self.shared
                        .underflow_count
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
        } else {
            // Generic N-channel path.
            for frame_idx in buffer_range.clone() {
                let have_data = *local_pos + channels <= local_buf.len();
                for ch in 0..channels.min(buffers.len()) {
                    if have_data {
                        buffers[ch][frame_idx] = local_buf[*local_pos + ch];
                    } else {
                        buffers[ch][frame_idx] = 0.0;
                    }
                }
                if have_data {
                    *local_pos += channels;
                } else {
                    self.shared
                        .underflow_count
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// LiveAudioStream
// ---------------------------------------------------------------------------

/// A handle to a live audio stream.
///
/// Created with [`LiveAudioStream::new`], then converted to an [`AudioSample`]
/// with [`as_sample`](Self::as_sample) for playback via [`SamplePlayer`].
/// Push PCM samples with [`push_mono`](Self::push_mono) or
/// [`push_stereo_interleaved`](Self::push_stereo_interleaved).
///
/// This handle is cheaply cloneable and can be shared across systems.
///
/// [`AudioSample`]: super::AudioSample
/// [`SamplePlayer`]: super::SamplePlayer
#[derive(Clone)]
pub struct LiveAudioStream {
    shared: Arc<SharedStreamState>,
    settings: StreamSettings,
}

impl std::fmt::Debug for LiveAudioStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiveAudioStream")
            .field("settings", &self.settings)
            .finish_non_exhaustive()
    }
}

impl LiveAudioStream {
    /// Create a new live audio stream with the given settings.
    ///
    /// The stream does not produce audio until enough data has been pushed
    /// to meet the start threshold, and until the resulting [`AudioSample`]
    /// is played via a [`SamplePlayer`].
    ///
    /// [`AudioSample`]: super::AudioSample
    /// [`SamplePlayer`]: super::SamplePlayer
    pub fn new(settings: StreamSettings) -> Self {
        let channel_count = match settings.channels {
            StreamChannels::Mono => 1,
            StreamChannels::Stereo => 2,
        };
        let max_samples = (settings.sample_rate as usize
            * settings.max_buffer_duration_ms as usize)
            / 1000
            * channel_count;
        let start_threshold = (settings.sample_rate as usize
            * settings.start_threshold_ms as usize)
            / 1000
            * channel_count;

        let shared = Arc::new(SharedStreamState::new(
            max_samples,
            start_threshold,
            settings.sample_rate,
            channel_count,
        ));

        Self { shared, settings }
    }

    /// Push mono PCM samples into the stream buffer.
    ///
    /// If the buffer would overflow, the oldest samples are dropped.
    ///
    /// # Panics
    ///
    /// Panics if the stream was created with [`StreamChannels::Stereo`].
    pub fn push_mono(&self, samples: &[f32]) {
        assert!(
            matches!(self.settings.channels, StreamChannels::Mono),
            "push_mono called on a stereo stream"
        );
        self.shared.push_samples(samples);
    }

    /// Push interleaved stereo PCM samples `[L, R, L, R, ...]` into the stream buffer.
    ///
    /// If the buffer would overflow, the oldest samples are dropped.
    ///
    /// # Panics
    ///
    /// Panics if the stream was created with [`StreamChannels::Mono`] or if
    /// the sample count is not even.
    pub fn push_stereo_interleaved(&self, samples: &[f32]) {
        assert!(
            matches!(self.settings.channels, StreamChannels::Stereo),
            "push_stereo_interleaved called on a mono stream"
        );
        assert!(
            samples.len() % 2 == 0,
            "stereo samples must come in pairs"
        );
        self.shared.push_samples(samples);
    }

    /// Get current buffer statistics.
    pub fn stats(&self) -> StreamStats {
        let channel_count = match self.settings.channels {
            StreamChannels::Mono => 1,
            StreamChannels::Stereo => 2,
        };
        let buffered_samples = self.shared.buffered_samples.load(Ordering::Relaxed);
        let buffered_frames = buffered_samples / channel_count;
        let buffered_ms = buffered_frames as f64 / self.settings.sample_rate as f64 * 1000.0;
        StreamStats {
            buffered_frames,
            buffered_ms,
            underflow_count: self.shared.underflow_count.load(Ordering::Relaxed),
            overflow_count: self.shared.overflow_count.load(Ordering::Relaxed),
        }
    }

    /// Create an [`AudioSample`] asset from this stream.
    ///
    /// The returned sample can be added to [`Assets<AudioSample>`] and played
    /// with a [`SamplePlayer`]. The stream remains active as long as this
    /// handle (or any clone) is alive to push samples.
    ///
    /// [`Assets<AudioSample>`]: bevy_asset::Assets
    /// [`SamplePlayer`]: super::SamplePlayer
    pub fn as_sample(&self) -> AudioSample {
        let resource = StreamingSampleResource::new(self.shared.clone());
        AudioSample::new(
            resource,
            NonZeroU32::new(self.settings.sample_rate).expect("sample_rate must be > 0"),
        )
    }

    /// Return the stream's settings.
    pub fn settings(&self) -> &StreamSettings {
        &self.settings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_drain_mono() {
        let stream = LiveAudioStream::new(StreamSettings {
            sample_rate: 100,
            channels: StreamChannels::Mono,
            max_buffer_duration_ms: 1000,
            start_threshold_ms: 0,
            ..Default::default()
        });

        let input: Vec<f32> = (0..10).map(|i| i as f32).collect();
        stream.push_mono(&input);

        let stats = stream.stats();
        assert_eq!(stats.buffered_frames, 10);
        assert_eq!(stats.underflow_count, 0);
        assert_eq!(stats.overflow_count, 0);

        // Drain via the shared state directly.
        let mut out = Vec::new();
        let drained = stream.shared.drain_into(&mut out, 10);
        assert_eq!(drained, 10);
        assert_eq!(out, input);
    }

    #[test]
    fn push_and_drain_stereo() {
        let stream = LiveAudioStream::new(StreamSettings {
            sample_rate: 100,
            channels: StreamChannels::Stereo,
            max_buffer_duration_ms: 1000,
            start_threshold_ms: 0,
            ..Default::default()
        });

        // 4 samples = 2 stereo frames.
        let input = vec![1.0, 2.0, 3.0, 4.0];
        stream.push_stereo_interleaved(&input);

        let stats = stream.stats();
        assert_eq!(stats.buffered_frames, 2);

        let mut out = Vec::new();
        stream.shared.drain_into(&mut out, 4);
        assert_eq!(out, input);
    }

    #[test]
    fn overflow_drops_oldest() {
        let stream = LiveAudioStream::new(StreamSettings {
            sample_rate: 10,
            channels: StreamChannels::Mono,
            max_buffer_duration_ms: 1000, // 10 samples max
            start_threshold_ms: 0,
            ..Default::default()
        });

        // Fill to capacity.
        let first: Vec<f32> = (0..10).map(|i| i as f32).collect();
        stream.push_mono(&first);
        assert_eq!(stream.stats().overflow_count, 0);

        // Push 3 more — should drop the 3 oldest.
        let second = vec![10.0, 11.0, 12.0];
        stream.push_mono(&second);
        assert!(stream.stats().overflow_count > 0);

        let mut out = Vec::new();
        stream.shared.drain_into(&mut out, 20);
        // Should contain samples 3..=12 (oldest 3 dropped).
        assert_eq!(out.len(), 10);
        assert_eq!(*out.first().unwrap(), 3.0);
        assert_eq!(*out.last().unwrap(), 12.0);
    }

    #[test]
    fn start_threshold_gates_playback() {
        let stream = LiveAudioStream::new(StreamSettings {
            sample_rate: 100,
            channels: StreamChannels::Mono,
            max_buffer_duration_ms: 1000,
            start_threshold_ms: 50, // 5 samples at 100 Hz
            ..Default::default()
        });

        // Push fewer than threshold.
        stream.push_mono(&[1.0, 2.0, 3.0]);
        assert!(!stream.shared.started.load(Ordering::Relaxed));

        // Push past threshold.
        stream.push_mono(&[4.0, 5.0]);
        assert!(stream.shared.started.load(Ordering::Relaxed));
    }

    #[test]
    fn fill_buffers_produces_silence_on_underflow() {
        let stream = LiveAudioStream::new(StreamSettings {
            sample_rate: 48_000,
            channels: StreamChannels::Mono,
            max_buffer_duration_ms: 200,
            start_threshold_ms: 0,
            ..Default::default()
        });

        // Force started.
        stream.shared.started.store(true, Ordering::Relaxed);

        let resource = StreamingSampleResource::new(stream.shared.clone());
        let mut buf = vec![999.0_f32; 8];
        let mut bufs: Vec<&mut [f32]> = vec![&mut buf];
        resource.fill_buffers(&mut bufs, 0..8, 0);

        // All silence — nothing was pushed.
        assert!(buf.iter().all(|&s| s == 0.0));
        assert!(stream.stats().underflow_count > 0);
    }

    #[test]
    fn fill_buffers_reads_pushed_data() {
        let stream = LiveAudioStream::new(StreamSettings {
            sample_rate: 48_000,
            channels: StreamChannels::Mono,
            max_buffer_duration_ms: 200,
            start_threshold_ms: 0,
            ..Default::default()
        });

        stream.shared.started.store(true, Ordering::Relaxed);

        let input: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        stream.push_mono(&input);

        let resource = StreamingSampleResource::new(stream.shared.clone());
        let mut buf = vec![0.0_f32; 8];
        let mut bufs: Vec<&mut [f32]> = vec![&mut buf];
        resource.fill_buffers(&mut bufs, 0..8, 0);

        assert_eq!(buf, input);
        assert_eq!(stream.stats().underflow_count, 0);
    }

    #[test]
    fn fill_buffers_stereo() {
        let stream = LiveAudioStream::new(StreamSettings {
            sample_rate: 48_000,
            channels: StreamChannels::Stereo,
            max_buffer_duration_ms: 200,
            start_threshold_ms: 0,
            ..Default::default()
        });

        stream.shared.started.store(true, Ordering::Relaxed);

        // 4 frames of stereo: [L0, R0, L1, R1, L2, R2, L3, R3]
        let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
        stream.push_stereo_interleaved(&input);

        let resource = StreamingSampleResource::new(stream.shared.clone());
        let mut buf_l = vec![0.0_f32; 4];
        let mut buf_r = vec![0.0_f32; 4];
        let mut bufs: Vec<&mut [f32]> = vec![&mut buf_l, &mut buf_r];
        resource.fill_buffers(&mut bufs, 0..4, 0);

        assert_eq!(buf_l, vec![0.0, 2.0, 4.0, 6.0]); // L channels
        assert_eq!(buf_r, vec![1.0, 3.0, 5.0, 7.0]); // R channels
        assert_eq!(stream.stats().underflow_count, 0);
    }

    #[test]
    fn as_sample_produces_valid_audio_sample() {
        let stream = LiveAudioStream::new(StreamSettings::default());
        let sample = stream.as_sample();
        assert_eq!(sample.original_sample_rate().get(), 48_000);
    }

    #[test]
    #[should_panic(expected = "push_mono called on a stereo stream")]
    fn push_mono_on_stereo_panics() {
        let stream = LiveAudioStream::new(StreamSettings {
            channels: StreamChannels::Stereo,
            ..Default::default()
        });
        stream.push_mono(&[1.0]);
    }

    #[test]
    #[should_panic(expected = "push_stereo_interleaved called on a mono stream")]
    fn push_stereo_on_mono_panics() {
        let stream = LiveAudioStream::new(StreamSettings::default());
        stream.push_stereo_interleaved(&[1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "stereo samples must come in pairs")]
    fn push_stereo_odd_count_panics() {
        let stream = LiveAudioStream::new(StreamSettings {
            channels: StreamChannels::Stereo,
            ..Default::default()
        });
        stream.push_stereo_interleaved(&[1.0, 2.0, 3.0]);
    }
}
