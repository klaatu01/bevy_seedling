//! This example demonstrates microphone passthrough using live audio streaming.
//!
//! Audio is captured from the system's default input device and pushed into a
//! [`LiveAudioStream`], which plays it back through the default output.
//!
//! **Warning:** Use headphones to avoid feedback loops!
//!
//! Controls:
//! - Space: toggle mute
//! - Escape: quit

use bevy::prelude::*;
use bevy_seedling::prelude::*;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, SeedlingPlugins))
        .add_systems(Startup, setup)
        .add_systems(Update, (toggle_mute, update_hud))
        .run();
}

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

#[derive(Resource)]
struct MicState {
    stream: LiveAudioStream,
    muted: Arc<AtomicBool>,
    // Keep the cpal stream alive.
    _input_stream: cpal::Stream,
}

#[derive(Component)]
struct HudText;

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

fn setup(mut commands: Commands, mut assets: ResMut<Assets<AudioSample>>) {
    let host = cpal::default_host();
    let input_device = host
        .default_input_device()
        .expect("no input device available");

    let input_config = input_device
        .default_input_config()
        .expect("no default input config");

    info!(
        "Using input device: {:?}",
        input_device.description()
    );
    info!("Input config: {:?}", input_config);

    let sample_rate = input_config.sample_rate();
    let channels = input_config.channels() as usize;

    let stream_channels = if channels >= 2 {
        StreamChannels::Stereo
    } else {
        StreamChannels::Mono
    };

    let stream = LiveAudioStream::new(StreamSettings {
        sample_rate,
        channels: stream_channels,
        max_buffer_duration_ms: 200,
        start_threshold_ms: 20,
    });

    let handle = assets.add(stream.as_sample());
    commands.spawn((
        SamplePlayer::new(handle),
        PlaybackSettings::default().preserve(),
    ));

    // Build cpal input stream that pushes captured audio into the LiveAudioStream.
    let stream_clone = stream.clone();
    let muted = Arc::new(AtomicBool::new(false));
    let muted_clone = muted.clone();

    let config = cpal::StreamConfig {
        channels: channels as u16,
        sample_rate,
        buffer_size: cpal::BufferSize::Default,
    };

    let input_stream = input_device
        .build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if muted_clone.load(Ordering::Relaxed) {
                    return;
                }
                match stream_channels {
                    StreamChannels::Mono => stream_clone.push_mono(data),
                    StreamChannels::Stereo => stream_clone.push_stereo_interleaved(data),
                }
            },
            |err| {
                eprintln!("input stream error: {err}");
            },
            None,
        )
        .expect("failed to build input stream");

    input_stream.play().expect("failed to start input stream");

    commands.insert_resource(MicState {
        stream,
        muted,
        _input_stream: input_stream,
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
// Systems
// ---------------------------------------------------------------------------

fn toggle_mute(keys: Res<ButtonInput<KeyCode>>, state: Res<MicState>) {
    if keys.just_pressed(KeyCode::Space) {
        let prev = state.muted.load(Ordering::Relaxed);
        state.muted.store(!prev, Ordering::Relaxed);
    }
}

fn update_hud(state: Res<MicState>, mut query: Query<&mut Text, With<HudText>>) {
    let stats = state.stream.stats();
    let muted = state.muted.load(Ordering::Relaxed);

    let text = format!(
        "\
Microphone Passthrough (use headphones!)
----------------------------------------
Status:     {} (Space to toggle)
Buffer:     {:.1} ms ({} frames)
Underflows: {}
Overflows:  {}",
        if muted { "MUTED" } else { "LIVE" },
        stats.buffered_ms,
        stats.buffered_frames,
        stats.underflow_count,
        stats.overflow_count,
    );

    for mut t in query.iter_mut() {
        **t = text.clone();
    }
}
