# -*- coding: utf-8 -*-
"""
Blink Detection and Logging for Muse EEG

This module detects eye blinks from EEG data and logs them for use in other functions.
Blinks typically appear as large amplitude artifacts in frontal channels (AF7, AF8).
"""

import numpy as np
import time
from collections import deque
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

class BlinkDetector:
    def __init__(self, 
                 threshold_multiplier=4.0,     # How many standard deviations above mean
                 min_blink_interval=0.2,       # Minimum time between blinks (seconds)
                 frontal_channels=[1, 2],      # AF7=1, AF8=2 (0-indexed: TP9=0, AF7=1, AF8=2, TP10=3, AUX=4)
                 baseline_duration=10,         # Seconds to establish noise baseline
                 blink_duration_range=(0.1, 0.5)):  # Expected blink duration range
        
        self.threshold_multiplier = threshold_multiplier
        self.min_blink_interval = min_blink_interval
        self.frontal_channels = frontal_channels
        self.baseline_duration = baseline_duration
        self.blink_duration_range = blink_duration_range
        
        # Baseline tracking
        self.baseline_buffer = deque(maxlen=int(baseline_duration * 256))  # 256Hz sampling rate
        self.baseline_established = False
        self.baseline_std = None
        self.baseline_mean = None
        
        # Blink tracking
        self.last_blink_time = 0
        self.blink_log = []
        self.potential_blink_start = None
        self.in_blink = False
        
        # Statistics
        self.total_blinks = 0
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
        
        Args:
            eeg_data: EEG data array (samples x channels)
            timestamps: Corresponding timestamps
            
        Returns:
            list: Detected blinks with timing and characteristics
        """
        if not self.baseline_established:
            self.update_baseline(eeg_data)
            return []
            
        detected_blinks = []
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
                        'timestamp': timestamp,
                        'start_time': current_time,
                        'max_amplitude': amplitude,
                        'start_sample': i,
                        'frontal_values': frontal_data[i].copy()
                    }
                    self.in_blink = True
                    
                elif self.in_blink and self.potential_blink_start:
                    # Update max amplitude during blink
                    if amplitude > self.potential_blink_start['max_amplitude']:
                        self.potential_blink_start['max_amplitude'] = amplitude
                        self.potential_blink_start['peak_timestamp'] = timestamp
                        self.potential_blink_start['peak_frontal_values'] = frontal_data[i].copy()
                        
            else:
                # Below threshold - end of blink?
                if self.in_blink and self.potential_blink_start:
                    blink_duration = current_time - self.potential_blink_start['start_time']
                    
                    # Validate blink duration
                    if self.blink_duration_range[0] <= blink_duration <= self.blink_duration_range[1]:
                        # Valid blink detected!
                        blink_info = {
                            'id': self.total_blinks + 1,
                            'timestamp': self.potential_blink_start.get('peak_timestamp', 
                                                                     self.potential_blink_start['timestamp']),
                            'duration': blink_duration,
                            'max_amplitude': self.potential_blink_start['max_amplitude'],
                            'amplitude_ratio': self.potential_blink_start['max_amplitude'] / self.baseline_mean,
                            'frontal_af7': self.potential_blink_start.get('peak_frontal_values', 
                                                                        self.potential_blink_start['frontal_values'])[0],
                            'frontal_af8': self.potential_blink_start.get('peak_frontal_values', 
                                                                        self.potential_blink_start['frontal_values'])[1],
                            'detection_time': current_time
                        }
                        
                        detected_blinks.append(blink_info)
                        self.blink_log.append(blink_info)
                        self.total_blinks += 1
                        self.last_blink_time = current_time
                        
                        # Call callback if set
                        if self.blink_callback:
                            self.blink_callback(blink_info)
                            
                        print(f"👁️ BLINK #{blink_info['id']} detected! "
                              f"Amplitude: {blink_info['max_amplitude']:.1f} "
                              f"({blink_info['amplitude_ratio']:.1f}x baseline) "
                              f"Duration: {blink_info['duration']:.3f}s")
                    
                    # Reset blink detection state
                    self.in_blink = False
                    self.potential_blink_start = None
                    
        return detected_blinks
    
    def get_recent_blinks(self, seconds_back=10):
        """Get blinks from the last N seconds"""
        current_time = time.time()
        return [blink for blink in self.blink_log 
                if (current_time - blink['detection_time']) <= seconds_back]
    
    def get_blink_rate(self, seconds_back=60):
        """Calculate blinks per minute over the last N seconds"""
        recent_blinks = self.get_recent_blinks(seconds_back)
        if not recent_blinks:
            return 0
        return len(recent_blinks) * (60 / seconds_back)
    
    def clear_log(self):
        """Clear the blink log"""
        self.blink_log.clear()
        self.total_blinks = 0
        
    def get_statistics(self):
        """Get blink detection statistics"""
        if not self.blink_log:
            return {"message": "No blinks detected yet"}
            
        amplitudes = [b['amplitude_ratio'] for b in self.blink_log]
        durations = [b['duration'] for b in self.blink_log]
        
        return {
            'total_blinks': len(self.blink_log),
            'blink_rate_1min': self.get_blink_rate(60),
            'avg_amplitude_ratio': np.mean(amplitudes),
            'avg_duration': np.mean(durations),
            'baseline_established': self.baseline_established,
            'current_threshold': self.baseline_mean + (self.threshold_multiplier * self.baseline_std) if self.baseline_established else None
        }

# Example callback functions for different uses
def music_blink_callback(blink_info):
    """Example: Use blinks to control music"""
    print(f"🎵 Music Control: Blink detected - could trigger track change!")
    
    # Example logic:
    if blink_info['amplitude_ratio'] > 6:  # Strong blink
        print("   → Strong blink: Skip to next song")
    elif blink_info['duration'] > 0.3:     # Long blink
        print("   → Long blink: Pause/Play toggle")
    else:
        print("   → Regular blink: Volume adjustment")

def interaction_blink_callback(blink_info):
    """Example: Use blinks for UI interaction"""
    print(f"🖱️ UI Interaction: Blink #{blink_info['id']} could trigger menu selection")

# Integration example with the main neurofeedback loop
def integrate_with_neurofeedback():
    """
    Example of how to integrate blink detection with your existing neurofeedback code
    """
    from pylsl import StreamInlet, resolve_byprop
    import utils
    
    # Initialize blink detector
    blink_detector = BlinkDetector(
        threshold_multiplier=4.0,    # Adjust sensitivity
        min_blink_interval=0.3,      # Prevent double-detection
        frontal_channels=[1, 2]      # AF7 and AF8
    )
    
    # Set up your callback
    blink_detector.set_blink_callback(music_blink_callback)
    
    # Your existing setup code...
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')
        
    inlet = StreamInlet(streams[0], max_chunklen=12)
    info = inlet.info()
    fs = int(info.nominal_srate())
    
    # Buffers
    BUFFER_LENGTH = 5
    EPOCH_LENGTH = 1
    SHIFT_LENGTH = 0.2
    INDEX_CHANNELS = [0, 1, 2, 3, 4]  # All channels for blink detection
    
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), len(INDEX_CHANNELS)))
    filter_state = None
    
    print("Starting combined neurofeedback and blink detection...")
    print("Establishing baselines...")
    
    try:
        while True:
            # Get data
            eeg_data, timestamps = inlet.pull_chunk(
                timeout=1, max_samples=int(SHIFT_LENGTH * fs))
            
            if len(eeg_data) > 0:
                ch_data = np.array(eeg_data)[:, INDEX_CHANNELS]
                
                # Update EEG buffer
                eeg_buffer, filter_state = utils.update_buffer(
                    eeg_buffer, ch_data, notch=True, filter_state=filter_state)
                
                # Detect blinks in the new data
                blink_detector.detect_blinks(ch_data, timestamps)
                
                # Your existing neurofeedback processing...
                # (band power analysis, change detection, etc.)
                
                # Print statistics occasionally
                if blink_detector.total_blinks > 0 and blink_detector.total_blinks % 5 == 0:
                    stats = blink_detector.get_statistics()
                    print(f"Blink Stats: {stats['total_blinks']} total, "
                          f"{stats['blink_rate_1min']:.1f}/min")
                    
    except KeyboardInterrupt:
        print("Stopping...")
        final_stats = blink_detector.get_statistics()
        print(f"Final blink statistics: {final_stats}")

if __name__ == "__main__":
    # Run the integration example
    integrate_with_neurofeedback()