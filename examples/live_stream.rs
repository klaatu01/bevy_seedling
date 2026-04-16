//! This example demonstrates live PCM audio streaming.
//!
//! A waveform is generated at runtime and pushed into a [`LiveAudioStream`].
//! Use the keyboard to change waveform, frequency, and amplitude, and view
//! buffer statistics in the HUD.
//!
//! Controls:
//! - Up/Down: change frequency (hold Shift for 10x)
//! - Left/Right: change amplitude
//! - Tab: cycle waveform

use bevy::prelude::*;
use bevy_seedling::prelude::*;
use std::f32::consts::TAU;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, SeedlingPlugins))
        .add_systems(Startup, setup)
        .add_systems(Update, (update_controls, push_samples, update_hud).chain())
        .run();
}

// ---------------------------------------------------------------------------
// Waveform types
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Waveform {
    Sine,
    Square,
    Sawtooth,
    Triangle,
}

impl Waveform {
    const ALL: [Waveform; 4] = [
        Waveform::Sine,
        Waveform::Square,
        Waveform::Sawtooth,
        Waveform::Triangle,
    ];

    fn next(self) -> Self {
        let i = Self::ALL.iter().position(|&w| w == self).unwrap();
        Self::ALL[(i + 1) % Self::ALL.len()]
    }

    fn label(self) -> &'static str {
        match self {
            Waveform::Sine => "Sine",
            Waveform::Square => "Square",
            Waveform::Sawtooth => "Sawtooth",
            Waveform::Triangle => "Triangle",
        }
    }

    fn sample(self, phase: f32) -> f32 {
        match self {
            Waveform::Sine => phase.sin(),
            Waveform::Square => {
                if phase < std::f32::consts::PI {
                    1.0
                } else {
                    -1.0
                }
            }
            Waveform::Sawtooth => 1.0 - (phase / std::f32::consts::PI),
            Waveform::Triangle => {
                let t = phase / TAU;
                if t < 0.5 {
                    4.0 * t - 1.0
                } else {
                    3.0 - 4.0 * t
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

#[derive(Resource)]
struct StreamState {
    stream: LiveAudioStream,
    phase: f32,
    frequency: f32,
    amplitude: f32,
    waveform: Waveform,
    time_elapsed: f64,
    samples_produced: u64,
}

#[derive(Component)]
struct HudText;

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

fn setup(mut commands: Commands, mut assets: ResMut<Assets<AudioSample>>) {
    let stream = LiveAudioStream::new(StreamSettings {
        sample_rate: 48_000,
        channels: StreamChannels::Mono,
        max_buffer_duration_ms: 200,
        start_threshold_ms: 20,
    });

    let handle = assets.add(stream.as_sample());
    commands.spawn((
        SamplePlayer::new(handle),
        PlaybackSettings::default().preserve(),
    ));

    commands.insert_resource(StreamState {
        stream,
        phase: 0.0,
        frequency: 440.0,
        amplitude: 0.3,
        waveform: Waveform::Sine,
        time_elapsed: 0.0,
        samples_produced: 0,
    });

    // Camera for UI
    commands.spawn(Camera2d);

    // HUD
    commands.spawn((
        Text::new(""),
        TextFont {
            font_size: 24.0,
            ..default()
        },
        Node {
            margin: UiRect::all(Val::Px(12.0)),
            ..default()
        },
        HudText,
    ));
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

fn update_controls(keys: Res<ButtonInput<KeyCode>>, mut state: ResMut<StreamState>) {
    let freq_step = if keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight) {
        50.0
    } else {
        5.0
    };

    if keys.pressed(KeyCode::ArrowUp) {
        state.frequency = (state.frequency + freq_step).min(8000.0);
    }
    if keys.pressed(KeyCode::ArrowDown) {
        state.frequency = (state.frequency - freq_step).max(20.0);
    }
    if keys.pressed(KeyCode::ArrowRight) {
        state.amplitude = (state.amplitude + 0.005).min(1.0);
    }
    if keys.pressed(KeyCode::ArrowLeft) {
        state.amplitude = (state.amplitude - 0.005).max(0.0);
    }
    if keys.just_pressed(KeyCode::Tab) {
        state.waveform = state.waveform.next();
    }
}

// ---------------------------------------------------------------------------
// Audio generation
// ---------------------------------------------------------------------------

fn push_samples(mut state: ResMut<StreamState>, time: Res<Time>) {
    let sample_rate = 48_000.0_f64;
    state.time_elapsed += time.delta_secs_f64();

    let total_needed = (state.time_elapsed * sample_rate) as u64;
    let to_generate = (total_needed - state.samples_produced) as usize;
    if to_generate == 0 {
        return;
    }

    let mut buf = Vec::with_capacity(to_generate);
    let phase_inc = state.frequency * TAU / sample_rate as f32;
    let waveform = state.waveform;
    let amplitude = state.amplitude;

    for _ in 0..to_generate {
        buf.push(waveform.sample(state.phase) * amplitude);
        state.phase += phase_inc;
        if state.phase > TAU {
            state.phase -= TAU;
        }
    }

    state.stream.push_mono(&buf);
    state.samples_produced += to_generate as u64;
}

// ---------------------------------------------------------------------------
// HUD
// ---------------------------------------------------------------------------

fn update_hud(state: Res<StreamState>, mut query: Query<&mut Text, With<HudText>>) {
    let stats = state.stream.stats();

    let text = format!(
        "\
Waveform:  {} (Tab to cycle)
Frequency: {:.0} Hz (Up/Down, +Shift 10x)
Amplitude: {:.0}% (Left/Right)
------
Buffer:     {:.1} ms ({} frames)
Underflows: {}
Overflows:  {}",
        state.waveform.label(),
        state.frequency,
        state.amplitude * 100.0,
        stats.buffered_ms,
        stats.buffered_frames,
        stats.underflow_count,
        stats.overflow_count,
    );

    for mut t in query.iter_mut() {
        **t = text.clone();
    }
}
