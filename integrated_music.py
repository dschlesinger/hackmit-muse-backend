"""
Integrated Neurofeedback Music System

Combines brain state detection, blink detection, and dynamic music generation
with Suno API to create an adaptive music experience.
"""

import numpy as np
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_byprop
import utils
from collections import deque
import time
import threading
import queue
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import os

# Import Suno utilities (assuming they're in the same directory)
from suno_utils.suno import generate_song, await_check_status, StatusCheck
from suno_utils.stream_audio import MediaPlayer

class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3

class BrainState(Enum):
    RELAXATION = "relaxation"
    CONCENTRATION = "concentration" 
    STRESS = "stress"
    UNKNOWN = "unknown"

@dataclass
class Song:
    id: str
    url: str
    brain_state: BrainState
    duration: float
    title: str = ""
    is_pregenerated: bool = False

class DoubleBlinlDetector:
    def __init__(self, max_interval=2.0, cooldown=3.0):
        self.max_interval = max_interval
        self.cooldown = cooldown
        self.blink_times = deque(maxlen=2)
        self.last_double_blink = 0
        
    def add_blink(self) -> bool:
        """Add a blink and check if it forms a double blink"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_double_blink < self.cooldown:
            return False
            
        self.blink_times.append(current_time)
        
        # Check for double blink
        if len(self.blink_times) == 2:
            time_diff = self.blink_times[1] - self.blink_times[0]
            if time_diff <= self.max_interval:
                self.last_double_blink = current_time
                print(f"👁️👁️ DOUBLE BLINK detected! ({time_diff:.2f}s interval)")
                return True
        
        return False

class BlinkDetector:
    def __init__(self, 
                 threshold_multiplier=4.0,
                 min_blink_interval=0.1,
                 frontal_channels=[1, 2],
                 baseline_duration=1,
                 blink_duration_range=(0.1, 0.5)):
        
        self.threshold_multiplier = threshold_multiplier
        self.min_blink_interval = min_blink_interval
        self.frontal_channels = frontal_channels
        self.baseline_duration = baseline_duration
        self.blink_duration_range = blink_duration_range
        
        # Baseline tracking
        self.baseline_buffer = deque(maxlen=int(baseline_duration * 256))
        self.baseline_established = False
        self.baseline_std = None
        self.baseline_mean = None
        
        # Blink tracking
        self.last_blink_time = 0
        self.potential_blink_start = None
        self.in_blink = False
        
        self.blink_callback = None
        
    def set_blink_callback(self, callback_function):
        self.blink_callback = callback_function
        
    def update_baseline(self, eeg_data):
        frontal_data = eeg_data[:, self.frontal_channels]
        
        for sample in frontal_data:
            self.baseline_buffer.append(np.mean(np.abs(sample)))
            
        if len(self.baseline_buffer) >= self.baseline_duration * 256 * 0.8:
            baseline_values = list(self.baseline_buffer)
            self.baseline_mean = np.mean(baseline_values)
            self.baseline_std = np.std(baseline_values)
            self.baseline_established = True
            
    def detect_blinks(self, eeg_data, timestamps):
        if not self.baseline_established:
            self.update_baseline(eeg_data)
            return False
            
        current_time = time.time()
        frontal_data = eeg_data[:, self.frontal_channels]
        amplitudes = np.mean(np.abs(frontal_data), axis=1)
        threshold = self.baseline_mean + (self.threshold_multiplier * self.baseline_std)
        
        for i, (amplitude, timestamp) in enumerate(zip(amplitudes, timestamps)):
            if amplitude > threshold:
                if not self.in_blink and (current_time - self.last_blink_time) > self.min_blink_interval:
                    self.potential_blink_start = {'start_time': current_time}
                    self.in_blink = True
            else:
                if self.in_blink and self.potential_blink_start:
                    blink_duration = current_time - self.potential_blink_start['start_time']
                    
                    if self.blink_duration_range[0] <= blink_duration <= self.blink_duration_range[1]:
                        self.last_blink_time = current_time
                        
                        if self.blink_callback:
                            self.blink_callback()
                        
                        self.in_blink = False
                        self.potential_blink_start = None
                        return True
                    
                    self.in_blink = False
                    self.potential_blink_start = None
                    
        return False

class NeurofeedbackProcessor:
    def __init__(self, baseline_duration=30, detection_window=5, change_threshold=0.3):
        self.baseline_duration = baseline_duration
        self.detection_window = detection_window
        self.change_threshold = change_threshold
        
        self.baseline_buffer = deque(maxlen=int(baseline_duration))
        self.detection_buffer = deque(maxlen=int(detection_window))
        
        self.baseline_established = False
        self.last_trigger_time = 0
        self.trigger_cooldown = 2.0
        
        self.alpha_baseline = 0
        self.beta_baseline = 0
        self.theta_baseline = 0
        
        # Track dominant brain state
        self.current_brain_state = BrainState.UNKNOWN
        self.brain_state_callback = None
        
    def set_brain_state_callback(self, callback_function):
        self.brain_state_callback = callback_function
        
    def add_sample(self, band_powers):
        alpha_metric = band_powers[Band.Alpha] / band_powers[Band.Delta]
        beta_metric = band_powers[Band.Beta] / band_powers[Band.Theta]
        theta_metric = band_powers[Band.Theta] / band_powers[Band.Alpha]
        
        sample = {
            'alpha': alpha_metric,
            'beta': beta_metric,
            'theta': theta_metric,
            'timestamp': time.time()
        }
        
        self.baseline_buffer.append(sample)
        self.detection_buffer.append(sample)
        
        if len(self.baseline_buffer) >= self.baseline_duration * 0.8:
            self._update_baselines()
            self.baseline_established = True
            
    def _update_baselines(self):
        if len(self.baseline_buffer) < 5:
            return
            
        alpha_values = [s['alpha'] for s in self.baseline_buffer]
        beta_values = [s['beta'] for s in self.baseline_buffer]
        theta_values = [s['theta'] for s in self.baseline_buffer]
        
        self.alpha_baseline = np.median(alpha_values)
        self.beta_baseline = np.median(beta_values)
        self.theta_baseline = np.median(theta_values)
        
    def detect_changes(self):
        if not self.baseline_established or len(self.detection_buffer) < 3:
            return {'trigger': False, 'reason': 'insufficient_data'}
        
        recent_samples = list(self.detection_buffer)[-min(3, len(self.detection_buffer)):]
        
        current_alpha = np.mean([s['alpha'] for s in recent_samples])
        current_beta = np.mean([s['beta'] for s in recent_samples])
        current_theta = np.mean([s['theta'] for s in recent_samples])
        
        alpha_change = (current_alpha - self.alpha_baseline) / self.alpha_baseline if self.alpha_baseline > 0 else 0
        beta_change = (current_beta - self.beta_baseline) / self.beta_baseline if self.beta_baseline > 0 else 0
        theta_change = (current_theta - self.theta_baseline) / self.theta_baseline if self.theta_baseline > 0 else 0
        
        current_time = time.time()
        if current_time - self.last_trigger_time < self.trigger_cooldown:
            return {
                'trigger': False, 
                'reason': 'cooldown',
                'alpha_change': alpha_change,
                'beta_change': beta_change,
                'theta_change': theta_change
            }
        
        result = {
            'trigger': False,
            'alpha_change': alpha_change,
            'beta_change': beta_change,
            'theta_change': theta_change,
            'current_alpha': current_alpha,
            'current_beta': current_beta,
            'current_theta': current_theta,
            'baselines': {
                'alpha': self.alpha_baseline,
                'beta': self.beta_baseline,
                'theta': self.theta_baseline
            }
        }
        
        # Determine brain state and trigger
        new_brain_state = None
        
        if alpha_change > self.change_threshold and beta_change < -self.change_threshold/2:
            result.update({
                'trigger': True,
                'trigger_type': 'relaxation_increase',
                'intensity': alpha_change,
                'reason': f'Alpha up {alpha_change:.2f}, Beta down {beta_change:.2f}'
            })
            new_brain_state = BrainState.RELAXATION
            self.last_trigger_time = current_time
            
        elif beta_change > self.change_threshold and theta_change < -self.change_threshold/2:
            result.update({
                'trigger': True,
                'trigger_type': 'concentration_increase',
                'intensity': beta_change,
                'reason': f'Beta up {beta_change:.2f}, Theta down {theta_change:.2f}'
            })
            new_brain_state = BrainState.CONCENTRATION
            self.last_trigger_time = current_time
            
        elif theta_change > self.change_threshold * 1.5:
            result.update({
                'trigger': True,
                'trigger_type': 'stress_increase',
                'intensity': theta_change,
                'reason': f'Theta up {theta_change:.2f}'
            })
            new_brain_state = BrainState.STRESS
            self.last_trigger_time = current_time
        
        # Update brain state and notify callback
        if new_brain_state and new_brain_state != self.current_brain_state:
            self.current_brain_state = new_brain_state
            if self.brain_state_callback:
                self.brain_state_callback(new_brain_state)
        
        return result

class MusicQueueManager:
    def __init__(self):
        self.queue = deque()
        self.current_player = None
        self.generation_queue = queue.Queue()
        self.is_generating = False
        self.generation_thread = None
        
        # Pre-generated songs (placeholder URLs)
        self.pregenerated_songs = {
            BrainState.RELAXATION: [
                Song("pre_relax_1", "https://example.com/calm1.mp3", BrainState.RELAXATION, 180, "Calm Waters", True),
                Song("pre_relax_2", "https://example.com/calm2.mp3", BrainState.RELAXATION, 200, "Peaceful Mind", True),
            ],
            BrainState.CONCENTRATION: [
                Song("pre_focus_1", "https://example.com/focus1.mp3", BrainState.CONCENTRATION, 190, "Deep Focus", True),
                Song("pre_focus_2", "https://example.com/focus2.mp3", BrainState.CONCENTRATION, 210, "Mental Clarity", True),
            ],
            BrainState.STRESS: [
                Song("pre_stress_1", "https://example.com/calm3.mp3", BrainState.STRESS, 170, "Stress Relief", True),
                Song("pre_stress_2", "https://example.com/calm4.mp3", BrainState.STRESS, 195, "Anxiety Ease", True),
            ]
        }
        
        # Counters for batch management
        self.songs_played_in_batch = 0
        self.initial_batch_generated = False
        
    def get_pregenerated_song(self, brain_state: BrainState) -> Optional[Song]:
        """Get a random pregenerated song for the given brain state"""
        if brain_state in self.pregenerated_songs and self.pregenerated_songs[brain_state]:
            import random
            return random.choice(self.pregenerated_songs[brain_state])
        return None
        
    def generate_song_for_state(self, brain_state: BrainState) -> Optional[Song]:
        """Generate a single song for the given brain state"""
        try:
            # Map brain state to music prompt
            prompts = {
                BrainState.RELAXATION: "calm ambient meditation music for deep relaxation",
                BrainState.CONCENTRATION: "upbeat energetic electronic music for focus and productivity",
                BrainState.STRESS: "soothing peaceful music for stress relief and anxiety reduction"
            }
            
            tags = {
                BrainState.RELAXATION: "ambient, calm, meditative, peaceful",
                BrainState.CONCENTRATION: "electronic, upbeat, energetic, focus",
                BrainState.STRESS: "soothing, peaceful, calming, healing"
            }
            
            topic = prompts.get(brain_state, "relaxing ambient music")
            tag = tags.get(brain_state, "ambient, calm")
            
            print(f"🎵 Generating {brain_state.value} music...")
            
            # Generate song
            clip_id = generate_song(topic=topic, tags=tag, make_instrumental=True)
            
            # Wait for completion
            clip_result = await_check_status(clip_id=clip_id, poll_interval=3.0, poll_timeout=180.0)
            
            if clip_result.audio_url:
                song = Song(
                    id=clip_id,
                    url=clip_result.audio_url,
                    brain_state=brain_state,
                    duration=clip_result.song_length,
                    title=f"Generated {brain_state.value.title()} Music"
                )
                print(f"✅ Generated song for {brain_state.value}: {song.duration:.1f}s")
                return song
            
        except Exception as e:
            print(f"❌ Failed to generate song for {brain_state.value}: {e}")
            # Retry once
            try:
                print("🔄 Retrying song generation...")
                clip_id = generate_song(topic=topic, tags=tag, make_instrumental=True)
                clip_result = await_check_status(clip_id=clip_id, poll_interval=3.0, poll_timeout=180.0)
                
                if clip_result.audio_url:
                    song = Song(
                        id=clip_id,
                        url=clip_result.audio_url,
                        brain_state=brain_state,
                        duration=clip_result.song_length,
                        title=f"Generated {brain_state.value.title()} Music (Retry)"
                    )
                    print(f"✅ Retry successful for {brain_state.value}: {song.duration:.1f}s")
                    return song
            except Exception as retry_e:
                print(f"❌ Retry also failed: {retry_e}")
                
        return None
        
    def generate_initial_batch(self, brain_state: BrainState):
        """Generate initial batch of 5 songs for the given brain state"""
        print(f"🎼 Generating initial batch of 5 songs for {brain_state.value}...")
        
        def generate_batch():
            songs = []
            for i in range(5):
                song = self.generate_song_for_state(brain_state)
                if song:
                    songs.append(song)
                    print(f"Generated song {i+1}/5 for batch")
                else:
                    print(f"Failed to generate song {i+1}/5, using pregenerated fallback")
                    fallback = self.get_pregenerated_song(brain_state)
                    if fallback:
                        songs.append(fallback)
            
            # Add songs to queue
            for song in songs:
                self.queue.append(song)
            
            self.initial_batch_generated = True
            self.songs_played_in_batch = 0
            print(f"✅ Initial batch complete! Queue size: {len(self.queue)}")
            
        # Run in background thread
        self.generation_thread = threading.Thread(target=generate_batch, daemon=True)
        self.generation_thread.start()
        
    def generate_background_song(self, brain_state: BrainState):
        """Generate a single song in the background"""
        if self.is_generating:
            return  # Already generating
            
        def generate_single():
            self.is_generating = True
            song = self.generate_song_for_state(brain_state)
            if song:
                self.queue.append(song)
                print(f"🎵 Background generation complete. Queue size: {len(self.queue)}")
            self.is_generating = False
            
        threading.Thread(target=generate_single, daemon=True).start()
        
    def play_next_song(self):
        """Play the next song in the queue"""
        if not self.queue:
            # No songs available, play pregenerated
            song = self.get_pregenerated_song(BrainState.RELAXATION)  # Default fallback
            if song:
                print(f"🎵 No songs in queue, playing pregenerated: {song.title}")
                self.current_player = MediaPlayer(song.url)
                self.current_player.play()
                return
        
        song = self.queue.popleft()
        print(f"🎵 Now playing: {song.title} ({song.brain_state.value}) - {song.duration:.1f}s")
        
        try:
            self.current_player = MediaPlayer(song.url)
            self.current_player.play()
            
            # Track batch progress
            if song.is_pregenerated:
                pass  # Don't count pregenerated songs in batch
            else:
                self.songs_played_in_batch += 1
                
                # Check if we need to start background generation
                if self.initial_batch_generated and self.songs_played_in_batch == 4:
                    print("🎼 Starting background generation (4th song playing)...")
                    # Generate for current dominant brain state (this would come from processor)
                    self.generate_background_song(BrainState.RELAXATION)  # Default for now
                    
        except Exception as e:
            print(f"❌ Failed to play song: {e}")
            # Try next song or fallback
            if self.queue:
                self.play_next_song()
            
    def skip_current_song(self):
        """Skip the current song and play next"""
        if self.current_player:
            print("⏭️ Skipping current song...")
            try:
                self.current_player.player.stop()
            except:
                pass
                
        self.play_next_song()
        
    def is_current_song_playing(self) -> bool:
        """Check if current song is still playing"""
        if self.current_player:
            return self.current_player.is_playing()
        return False
        
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        return {
            'queue_length': len(self.queue),
            'songs_played_in_batch': self.songs_played_in_batch,
            'initial_batch_generated': self.initial_batch_generated,
            'is_generating': self.is_generating,
            'current_playing': self.current_player is not None and self.is_current_song_playing()
        }

class IntegratedNeurofeedbackSystem:
    def __init__(self):
        # Initialize all components
        self.processor = NeurofeedbackProcessor(
            baseline_duration=30,
            detection_window=30, 
            change_threshold=0.5
        )
        
        self.blink_detector = BlinkDetector(
            threshold_multiplier=3.0,
            min_blink_interval=0.3,
            frontal_channels=[1, 2]
        )
        
        self.double_blink_detector = DoubleBlinlDetector(
            max_interval=2.0,
            cooldown=3.0
        )
        
        self.music_manager = MusicQueueManager()
        
        # Set up callbacks
        self.processor.set_brain_state_callback(self.on_brain_state_change)
        self.blink_detector.set_blink_callback(self.on_blink_detected)
        
        # System state
        self.calibration_complete = False
        self.initial_batch_triggered = False
        
    def on_brain_state_change(self, new_brain_state: BrainState):
        """Handle brain state changes"""
        print(f"🧠 Brain state changed to: {new_brain_state.value}")
        
        # If we haven't triggered initial batch generation yet, do it now
        if not self.initial_batch_triggered and self.calibration_complete:
            self.music_manager.generate_initial_batch(new_brain_state)
            self.initial_batch_triggered = True
        
        # For future enhancements: could influence ongoing generation
        
    def on_blink_detected(self):
        """Handle single blink detection"""
        is_double_blink = self.double_blink_detector.add_blink()
        
        if is_double_blink:
            self.music_manager.skip_current_song()
        else:
            print("👁️ Single blink detected")
            
    def run_music_monitoring(self):
        """Background thread to monitor music playback"""
        def monitor():
            while True:
                if self.calibration_complete and not self.music_manager.is_current_song_playing():
                    if self.music_manager.initial_batch_generated or len(self.music_manager.queue) > 0:
                        print("🎵 Song ended, playing next...")
                        self.music_manager.play_next_song()
                        
                        # Generate replacement song if queue is established
                        if self.music_manager.initial_batch_generated and len(self.music_manager.queue) <= 3:
                            # Generate for current brain state
                            current_state = self.processor.current_brain_state
                            if current_state != BrainState.UNKNOWN:
                                self.music_manager.generate_background_song(current_state)
                
                time.sleep(2)  # Check every 2 seconds
                
        threading.Thread(target=monitor, daemon=True).start()

# Main execution
if __name__ == "__main__":
    """ EXPERIMENTAL PARAMETERS """
    BUFFER_LENGTH = 5
    EPOCH_LENGTH = 1
    OVERLAP_LENGTH = 0.8
    SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH
    INDEX_CHANNELS = [0, 1, 2, 3, 4]

    # Initialize the integrated system
    system = IntegratedNeurofeedbackSystem()
    
    # Start music monitoring
    system.run_music_monitoring()

    """ 1. CONNECT TO EEG STREAM """
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    info = inlet.info()
    fs = int(info.nominal_srate())

    """ 2. INITIALIZE BUFFERS """
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), len(INDEX_CHANNELS)))
    filter_state = None

    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) / SHIFT_LENGTH + 1))
    band_buffer = np.zeros((n_win_test, 4))

    """ 3. MAIN LOOP """
    print('Press Ctrl-C in the console to break the while loop.')
    print('🎼 INTEGRATED NEUROFEEDBACK MUSIC SYSTEM')
    print('=' * 50)
    print('Phase 1: Calibration with pregenerated music')
    print('Phase 2: Brain state detection → Initial batch generation')
    print('Phase 3: Queue management with background generation')
    print('Phase 4: Steady state - one song generated as current ends')
    print('👁️👁️ Double blink within 2 seconds = Skip song')
    print('=' * 50)

    # Play initial pregenerated music
    initial_song = system.music_manager.get_pregenerated_song(BrainState.RELAXATION)
    if initial_song:
        print(f"🎵 Starting with pregenerated music: {initial_song.title}")
        system.music_manager.current_player = MediaPlayer(initial_song.url)
        system.music_manager.current_player.play()

    try:
        while True:
            """ 3.1 ACQUIRE DATA """
            eeg_data, timestamps = inlet.pull_chunk(
                timeout=1, max_samples=int(SHIFT_LENGTH * fs))

            if len(eeg_data) > 0:
                ch_data = np.array(eeg_data)[:, INDEX_CHANNELS]
                
                eeg_buffer, filter_state = utils.update_buffer(
                    eeg_buffer, ch_data, notch=True, filter_state=filter_state)

                """ 3.2 BLINK DETECTION """
                system.blink_detector.detect_blinks(ch_data, timestamps)

                """ 3.3 COMPUTE BAND POWERS """
                single_channel_buffer = eeg_buffer[:, [0]]
                data_epoch = utils.get_last_data(single_channel_buffer, EPOCH_LENGTH * fs)
                band_powers = utils.compute_band_powers(data_epoch, fs)
                band_buffer, _ = utils.update_buffer(band_buffer, np.asarray([band_powers]))
                
                smooth_band_powers = np.mean(band_buffer, axis=0)

                """ 3.4 BRAIN STATE PROCESSING """
                system.processor.add_sample(smooth_band_powers)
                change_info = system.processor.detect_changes()
                
                # Check if calibration is complete
                if not system.calibration_complete:
                    brain_ready = system.processor.baseline_established
                    blink_ready = system.blink_detector.baseline_established
                    
                    if brain_ready and blink_ready:
                        system.calibration_complete = True
                        print("\n✅ CALIBRATION COMPLETE!")
                        print("Now monitoring for brain state changes to trigger music generation...")
                        print("Double blink to skip songs is now active.")
                        print("-" * 50)

                """ 3.5 STATUS REPORTING """
                if hasattr(system, '_status_counter'):
                    system._status_counter += 1
                else:
                    system._status_counter = 0
                    
                if system._status_counter % 50 == 0:
                    if system.calibration_complete:
                        queue_status = system.music_manager.get_queue_status()
                        print(f"Status - Queue: {queue_status['queue_length']}, "
                              f"Batch: {queue_status['songs_played_in_batch']}/5, "
                              f"Brain: {system.processor.current_brain_state.value}, "
                              f"Playing: {queue_status['current_playing']}")
                    else:
                        brain_progress = len(system.processor.baseline_buffer) / system.processor.baseline_duration
                        blink_progress = len(system.blink_detector.baseline_buffer) / (system.blink_detector.baseline_duration * 256)
                        print(f"Calibration - Brain: {brain_progress:.1%}, "
                              f"Blink: {blink_progress:.1%}")

    except KeyboardInterrupt:
        print('\n🛑 Shutting down system...')
        if system.music_manager.current_player:
            try:
                system.music_manager.current_player.player.stop()
            except:
                pass
        print('✅ System stopped cleanly!')