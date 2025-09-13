"""
Enhanced Neurofeedback with Change Detection and Blink Detection

Simplified version that only detects whether a blink occurred, without tracking history.
"""

import numpy as np
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_byprop
import utils
from collections import deque
import time

class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3

class BlinkDetector:
    def __init__(self, 
                 threshold_multiplier=4.0,
                 min_blink_interval=0.05,
                 frontal_channels=[1, 3],
                 baseline_duration=1,
                 blink_duration_range=(0.05, 0.5)):
        
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
        
        # Blink tracking - simplified
        self.last_blink_time = 0
        self.potential_blink_start = None
        self.in_blink = False
        
        self.blink_callback = None
        
    def set_blink_callback(self, callback_function):
        """Set a callback function to be called when a blink is detected"""
        self.blink_callback = callback_function
        
    def update_baseline(self, eeg_data):
        """Update the baseline statistics from clean EEG data"""
        # Use frontal channels for baseline calculation
        frontal_data = eeg_data[:, self.frontal_channels]
        
        # Add to baseline buffer
        for sample in frontal_data:
            self.baseline_buffer.append(np.mean(np.abs(sample)))  # Mean absolute value
            
        # Update baseline statistics if we have enough data
        if len(self.baseline_buffer) >= self.baseline_duration * 256 * 0.8:
            baseline_values = list(self.baseline_buffer)
            self.baseline_mean = np.mean(baseline_values)
            self.baseline_std = np.std(baseline_values)
            self.baseline_established = True
            
    def detect_blinks(self, eeg_data, timestamps):
        """
        Detect blinks in the current EEG data chunk
        
        Returns:
            bool: True if a blink was detected, False otherwise
        """
        if not self.baseline_established:
            self.update_baseline(eeg_data)
            return False
            
        current_time = time.time()
        
        # Focus on frontal channels where blinks are most prominent
        frontal_data = eeg_data[:, self.frontal_channels]
        
        # Calculate the amplitude for each sample (mean of frontal channels)
        amplitudes = np.mean(np.abs(frontal_data), axis=1)
        
        # Dynamic threshold based on baseline
        threshold = self.baseline_mean + (self.threshold_multiplier * self.baseline_std)
        
        for i, (amplitude, timestamp) in enumerate(zip(amplitudes, timestamps)):
            # Check if amplitude exceeds threshold
            if amplitude > threshold:
                if not self.in_blink and (current_time - self.last_blink_time) > self.min_blink_interval:
                    # Start of potential blink
                    self.potential_blink_start = {
                        'start_time': current_time,
                    }
                    self.in_blink = True
                    
            else:
                # Below threshold - end of blink?
                if self.in_blink and self.potential_blink_start:
                    blink_duration = current_time - self.potential_blink_start['start_time']
                    
                    # Validate blink duration
                    if self.blink_duration_range[0] <= blink_duration <= self.blink_duration_range[1]:
                        # Valid blink detected!
                        self.last_blink_time = current_time
                        
                        # Call callback if set
                        if self.blink_callback:
                            self.blink_callback()
                        
                        # Reset blink detection state
                        self.in_blink = False
                        self.potential_blink_start = None
                        return True
                    
                    # Reset if not a valid blink
                    self.in_blink = False
                    self.potential_blink_start = None
                    
        return False

class NeurofeedbackProcessor:
    def __init__(self, baseline_duration=30, detection_window=5, change_threshold=0.3):
        """
        Initialize the neurofeedback processor with change detection capabilities.
        
        Args:
            baseline_duration: Duration in seconds for baseline calculation
            detection_window: Duration in seconds for change detection window
            change_threshold: Minimum change ratio to trigger an event (e.g., 0.3 = 30% change)
        """
        self.baseline_duration = baseline_duration
        self.detection_window = detection_window
        self.change_threshold = change_threshold
        
        # Buffers for different time scales
        self.baseline_buffer = deque(maxlen=int(baseline_duration))  # Long-term baseline
        self.detection_buffer = deque(maxlen=int(detection_window))  # Short-term for change detection
        
        # State tracking
        self.baseline_established = False
        self.last_trigger_time = 0
        self.trigger_cooldown = 2.0  # Minimum seconds between triggers
        
        # Current baselines
        self.alpha_baseline = 0
        self.beta_baseline = 0
        self.theta_baseline = 0
        
    def add_sample(self, band_powers):
        """Add a new sample and update baselines"""
        # Calculate current metrics
        alpha_metric = band_powers[Band.Alpha] / band_powers[Band.Delta]
        beta_metric = band_powers[Band.Beta] / band_powers[Band.Theta]
        theta_metric = band_powers[Band.Theta] / band_powers[Band.Alpha]
        
        sample = {
            'alpha': alpha_metric,
            'beta': beta_metric,
            'theta': theta_metric,
            'timestamp': time.time()
        }
        
        # Add to both buffers
        self.baseline_buffer.append(sample)
        self.detection_buffer.append(sample)
        
        # Update baselines if we have enough data
        if len(self.baseline_buffer) >= self.baseline_duration * 0.8:
            self._update_baselines()
            self.baseline_established = True
            
    def _update_baselines(self):
        """Update the baseline values from the long-term buffer"""
        if len(self.baseline_buffer) < 5:
            return
            
        alpha_values = [s['alpha'] for s in self.baseline_buffer]
        beta_values = [s['beta'] for s in self.baseline_buffer]
        theta_values = [s['theta'] for s in self.baseline_buffer]
        
        # Use median for robustness against outliers
        self.alpha_baseline = np.median(alpha_values)
        self.beta_baseline = np.median(beta_values)
        self.theta_baseline = np.median(theta_values)
        
    def detect_changes(self):
        """
        Detect significant changes from baseline and return trigger events
        
        Returns:
            dict: Contains change information and trigger recommendations
        """
        if not self.baseline_established or len(self.detection_buffer) < 3:
            return {'trigger': False, 'reason': 'insufficient_data'}
        
        # Calculate current averages from detection window
        recent_samples = list(self.detection_buffer)[-min(3, len(self.detection_buffer)):]
        
        current_alpha = np.mean([s['alpha'] for s in recent_samples])
        current_beta = np.mean([s['beta'] for s in recent_samples])
        current_theta = np.mean([s['theta'] for s in recent_samples])
        
        # Calculate relative changes
        alpha_change = (current_alpha - self.alpha_baseline) / self.alpha_baseline if self.alpha_baseline > 0 else 0
        beta_change = (current_beta - self.beta_baseline) / self.beta_baseline if self.beta_baseline > 0 else 0
        theta_change = (current_theta - self.theta_baseline) / self.theta_baseline if self.theta_baseline > 0 else 0
        
        # Check for significant changes and cooldown
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
        
        # Detect relaxation increase (alpha up, beta down)
        if alpha_change > self.change_threshold and beta_change < -self.change_threshold/2:
            result.update({
                'trigger': True,
                'trigger_type': 'relaxation_increase',
                'intensity': alpha_change,
                'reason': f'Alpha up {alpha_change:.2f}, Beta down {beta_change:.2f}'
            })
            self.last_trigger_time = current_time
            
        # Detect concentration increase (beta up, theta down)
        elif beta_change > self.change_threshold and theta_change < -self.change_threshold/2:
            result.update({
                'trigger': True,
                'trigger_type': 'concentration_increase',
                'intensity': beta_change,
                'reason': f'Beta up {beta_change:.2f}, Theta down {theta_change:.2f}'
            })
            self.last_trigger_time = current_time
            
        # Detect stress/anxiety (theta up significantly)
        elif theta_change > self.change_threshold * 1.5:
            result.update({
                'trigger': True,
                'trigger_type': 'stress_increase',
                'intensity': theta_change,
                'reason': f'Theta up {theta_change:.2f}'
            })
            self.last_trigger_time = current_time
            
        return result

""" EXPERIMENTAL PARAMETERS """
BUFFER_LENGTH = 5
EPOCH_LENGTH = 1
OVERLAP_LENGTH = 0.8
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH
INDEX_CHANNELS = [0, 1, 2, 3, 4]  # All channels for blink detection

# Initialize processors
processor = NeurofeedbackProcessor(
    baseline_duration=30,      # 30 seconds to establish baseline
    detection_window=30,       # 30 seconds for change detection
    change_threshold=0.5       # 50% change threshold
)

blink_detector = BlinkDetector(
    threshold_multiplier=3.0,   # Sensitivity for blink detection
    min_blink_interval=0.3,     # Prevent double-detection
    frontal_channels=[1, 2]     # AF7 and AF8 channels
)

def music_control_callback(change_info):
    """
    Callback function to handle music control based on brain state changes.
    Replace this with your actual music manipulation code.
    """
    if change_info['trigger']:
        trigger_type = change_info['trigger_type']
        intensity = change_info['intensity']
        
        print(f"\n🎵 BRAIN STATE TRIGGER: {trigger_type}")
        print(f"   Intensity: {intensity:.2f}")
        print(f"   Reason: {change_info['reason']}")
        
        if trigger_type == 'relaxation_increase':
            print("   → Suggestion: Softer, slower music")
        elif trigger_type == 'concentration_increase':
            print("   → Suggestion: Energetic, rhythmic music")
        elif trigger_type == 'stress_increase':
            print("   → Suggestion: Calming, ambient music")

def blink_control_callback():
    """
    Simplified callback function - just indicates a blink occurred
    """
    print("\n👁️  BLINK DETECTED - Performing action!")

# Set up the blink callback
blink_detector.set_blink_callback(blink_control_callback)

if __name__ == "__main__":
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
    # Modified to use all channels for blink detection
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), len(INDEX_CHANNELS)))
    filter_state = None

    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) / SHIFT_LENGTH + 1))
    band_buffer = np.zeros((n_win_test, 4))

    """ 3. MAIN LOOP """
    print('Press Ctrl-C in the console to break the while loop.')
    print('Establishing baselines... (this may take 30 seconds)')
    print('Brain state baseline + Blink detection baseline')
    print('\nControls available:')
    print('- Brain state changes: Automatic music mood adjustment')
    print('- Eye blinks: Manual music control')

    try:
        while True:
            """ 3.1 ACQUIRE DATA """
            eeg_data, timestamps = inlet.pull_chunk(
                timeout=1, max_samples=int(SHIFT_LENGTH * fs))

            if len(eeg_data) > 0:
                # Get all channel data for blink detection
                ch_data = np.array(eeg_data)[:, INDEX_CHANNELS]
                
                # Update EEG buffer with all channels
                eeg_buffer, filter_state = utils.update_buffer(
                    eeg_buffer, ch_data, notch=True, filter_state=filter_state)

                """ 3.2 BLINK DETECTION """
                # Detect blinks using the current chunk
                blink_detected = blink_detector.detect_blinks(ch_data, timestamps)

                """ 3.3 COMPUTE BAND POWERS (using single channel as in original) """
                # Use only the first channel for band power analysis (as in original)
                single_channel_buffer = eeg_buffer[:, [0]]  # Just TP9 channel
                data_epoch = utils.get_last_data(single_channel_buffer, EPOCH_LENGTH * fs)
                band_powers = utils.compute_band_powers(data_epoch, fs)
                band_buffer, _ = utils.update_buffer(band_buffer, np.asarray([band_powers]))
                
                # Use the smoothed band powers
                smooth_band_powers = np.mean(band_buffer, axis=0)

                """ 3.4 ENHANCED CHANGE DETECTION """
                # Add sample to processor
                processor.add_sample(smooth_band_powers)
                
                # Detect changes
                change_info = processor.detect_changes()
                
                # Handle music control based on brain state
                music_control_callback(change_info)
                
                # Print status (less frequent than before to reduce noise)
                if not change_info['trigger']:
                    # Only print baseline status occasionally
                    if hasattr(processor, '_status_counter'):
                        processor._status_counter += 1
                    else:
                        processor._status_counter = 0
                        
                    if processor._status_counter % 50 == 0:  # Every 50 iterations
                        brain_ready = processor.baseline_established
                        blink_ready = blink_detector.baseline_established
                        
                        if brain_ready and blink_ready:
                            print(f"Monitoring... Brain: α:{change_info['alpha_change']:+.2f} "
                                  f"β:{change_info['beta_change']:+.2f} "
                                  f"θ:{change_info['theta_change']:+.2f}")
                        else:
                            brain_progress = len(processor.baseline_buffer) / processor.baseline_duration if not brain_ready else 1.0
                            blink_progress = len(blink_detector.baseline_buffer) / (blink_detector.baseline_duration * 256) if not blink_ready else 1.0
                            print(f"Baselines - Brain: {brain_progress:.1%}, Blink: {blink_progress:.1%}")

    except KeyboardInterrupt:
        print('\nClosing!')