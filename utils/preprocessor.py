import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional, Dict, Any
import scipy.signal

class SeismicPreprocessor:
    """
    Preprocessor for seismic waveform data and velocity maps
    Handles normalization, filtering, and augmentation
    """
    
    def __init__(self):
        self.seismic_scaler = None
        self.velocity_scaler = None
        self.seismic_stats = {}
        self.velocity_stats = {}
        self.fitted = False
    
    def fit(self, seismic_samples: np.ndarray, velocity_samples: np.ndarray):
        """
        Fit preprocessing parameters on training data
        
        Args:
            seismic_samples: Array of seismic data samples
            velocity_samples: Array of velocity map samples
        """
        # Compute statistics for seismic data
        self.seismic_stats = {
            'mean': np.mean(seismic_samples),
            'std': np.std(seismic_samples),
            'min': np.min(seismic_samples),
            'max': np.max(seismic_samples),
            'percentile_1': np.percentile(seismic_samples, 1),
            'percentile_99': np.percentile(seismic_samples, 99)
        }
        
        # Compute statistics for velocity data
        self.velocity_stats = {
            'mean': np.mean(velocity_samples),
            'std': np.std(velocity_samples),
            'min': np.min(velocity_samples),
            'max': np.max(velocity_samples),
            'percentile_1': np.percentile(velocity_samples, 1),
            'percentile_99': np.percentile(velocity_samples, 99)
        }
        
        self.fitted = True
    
    def preprocess_seismic_data(self, seismic_data: np.ndarray, apply_filters: bool = True) -> np.ndarray:
        """
        Preprocess seismic waveform data
        
        Args:
            seismic_data: Raw seismic data
            apply_filters: Whether to apply signal processing filters
        
        Returns:
            Preprocessed seismic data
        """
        processed = seismic_data.copy()
        
        # Apply bandpass filter to remove noise
        if apply_filters and len(processed.shape) >= 3:
            processed = self._apply_bandpass_filter(processed)
        
        # Normalize amplitude
        processed = self._normalize_seismic_amplitude(processed)
        
        # Clip outliers
        processed = self._clip_outliers(processed, 'seismic')
        
        return processed
    
    def preprocess_velocity_data(self, velocity_data: np.ndarray) -> np.ndarray:
        """
        Preprocess velocity map data
        
        Args:
            velocity_data: Raw velocity maps
        
        Returns:
            Preprocessed velocity data
        """
        processed = velocity_data.copy()
        
        # Ensure positive velocities
        processed = np.maximum(processed, 1000.0)  # Minimum realistic velocity
        
        # Apply smoothing to reduce noise
        processed = self._apply_gaussian_smoothing(processed)
        
        # Normalize to standard range
        processed = self._normalize_velocity_range(processed)
        
        return processed
    
    def _apply_bandpass_filter(self, seismic_data: np.ndarray, lowcut: float = 2.0, highcut: float = 50.0, fs: float = 100.0) -> np.ndarray:
        """Apply bandpass filter to seismic traces"""
        # Design Butterworth bandpass filter
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        
        try:
            b, a = scipy.signal.butter(4, [low, high], btype='band')
            
            # Apply filter along time axis
            filtered = seismic_data.copy()
            
            if len(seismic_data.shape) == 4:  # (batch, sources, time, receivers)
                for i in range(seismic_data.shape[0]):
                    for j in range(seismic_data.shape[1]):
                        for k in range(seismic_data.shape[3]):
                            filtered[i, j, :, k] = scipy.signal.filtfilt(b, a, seismic_data[i, j, :, k])
            elif len(seismic_data.shape) == 3:  # (sources, time, receivers)
                for j in range(seismic_data.shape[0]):
                    for k in range(seismic_data.shape[2]):
                        filtered[j, :, k] = scipy.signal.filtfilt(b, a, seismic_data[j, :, k])
            
            return filtered
        except Exception:
            # Return original data if filtering fails
            return seismic_data
    
    def _normalize_seismic_amplitude(self, seismic_data: np.ndarray) -> np.ndarray:
        """Normalize seismic amplitude using z-score normalization"""
        if self.fitted and 'mean' in self.seismic_stats:
            # Use fitted statistics
            mean = self.seismic_stats['mean']
            std = self.seismic_stats['std']
        else:
            # Compute on-the-fly
            mean = np.mean(seismic_data)
            std = np.std(seismic_data)
        
        if std > 0:
            normalized = (seismic_data - mean) / std
        else:
            normalized = seismic_data - mean
        
        return normalized
    
    def _normalize_velocity_range(self, velocity_data: np.ndarray) -> np.ndarray:
        """Normalize velocity to [0, 1] range"""
        if self.fitted and 'min' in self.velocity_stats:
            # Use fitted statistics
            v_min = self.velocity_stats['min']
            v_max = self.velocity_stats['max']
        else:
            # Compute on-the-fly
            v_min = np.min(velocity_data)
            v_max = np.max(velocity_data)
        
        if v_max > v_min:
            normalized = (velocity_data - v_min) / (v_max - v_min)
        else:
            normalized = velocity_data - v_min
        
        return normalized
    
    def _apply_gaussian_smoothing(self, velocity_data: np.ndarray, sigma: float = 0.5) -> np.ndarray:
        """Apply Gaussian smoothing to velocity maps"""
        try:
            from scipy import ndimage
            smoothed = velocity_data.copy()
            
            if len(velocity_data.shape) == 3:  # (batch, height, width)
                for i in range(velocity_data.shape[0]):
                    smoothed[i] = ndimage.gaussian_filter(velocity_data[i], sigma=sigma)
            elif len(velocity_data.shape) == 2:  # (height, width)
                smoothed = ndimage.gaussian_filter(velocity_data, sigma=sigma)
            
            return smoothed
        except ImportError:
            # Return original if scipy not available
            return velocity_data
    
    def _clip_outliers(self, data: np.ndarray, data_type: str) -> np.ndarray:
        """Clip extreme outliers based on percentiles"""
        if self.fitted:
            if data_type == 'seismic':
                lower_bound = self.seismic_stats['percentile_1']
                upper_bound = self.seismic_stats['percentile_99']
            else:
                lower_bound = self.velocity_stats['percentile_1']
                upper_bound = self.velocity_stats['percentile_99']
        else:
            lower_bound = np.percentile(data, 1)
            upper_bound = np.percentile(data, 99)
        
        return np.clip(data, lower_bound, upper_bound)
    
    def denormalize_velocity(self, normalized_velocity: np.ndarray) -> np.ndarray:
        """Convert normalized velocity back to original scale"""
        if not self.fitted:
            return normalized_velocity
        
        v_min = self.velocity_stats['min']
        v_max = self.velocity_stats['max']
        
        return normalized_velocity * (v_max - v_min) + v_min
    
    def augment_seismic_data(self, seismic_data: np.ndarray, velocity_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation to seismic data and velocity maps
        
        Args:
            seismic_data: Input seismic data
            velocity_data: Corresponding velocity maps
        
        Returns:
            Augmented (seismic, velocity) tuple
        """
        aug_seismic = seismic_data.copy()
        aug_velocity = velocity_data.copy()
        
        # Random noise addition
        if np.random.random() < 0.3:
            noise_level = 0.05 * np.std(aug_seismic)
            noise = np.random.normal(0, noise_level, aug_seismic.shape)
            aug_seismic += noise
        
        # Random amplitude scaling
        if np.random.random() < 0.3:
            scale_factor = np.random.uniform(0.8, 1.2)
            aug_seismic *= scale_factor
        
        # Random time shift (circular shift)
        if np.random.random() < 0.2 and len(aug_seismic.shape) >= 3:
            time_axis = -2 if len(aug_seismic.shape) == 4 else 1
            shift = np.random.randint(-5, 6)
            aug_seismic = np.roll(aug_seismic, shift, axis=time_axis)
        
        # Horizontal flip (for 2D velocity maps)
        if np.random.random() < 0.5:
            aug_velocity = np.flip(aug_velocity, axis=-1)
            # Also flip seismic data receiver axis
            if len(aug_seismic.shape) >= 3:
                aug_seismic = np.flip(aug_seismic, axis=-1)
        
        return aug_seismic, aug_velocity
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get preprocessing statistics"""
        return {
            'seismic': self.seismic_stats,
            'velocity': self.velocity_stats
        }
