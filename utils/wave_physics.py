"""
Wave physics utilities for FWI
Based on OpenFWI tutorial concepts and wave traveltime calculations
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import scipy.signal

class WavePhysicsCalculator:
    """Calculate wave traveltimes and physics-based constraints"""
    
    def __init__(self):
        self.dt = 0.001  # Time sampling (1ms)
        self.dg = 10     # Geophone spacing (10m)
        self.dx = 10     # Spatial sampling (10m)
        self.dz = 10     # Depth sampling (10m)
    
    def calculate_direct_wave_traveltime(self, distance: float, velocity: float) -> float:
        """
        Calculate direct wave traveltime: t_direct = d/c
        
        Args:
            distance: Distance between source and receiver (m)
            velocity: P-wave velocity (m/s)
        
        Returns:
            Travel time in seconds
        """
        return distance / velocity
    
    def calculate_reflection_traveltime(self, offset: float, depth: float, velocity: float) -> float:
        """
        Calculate reflection wave traveltime: t_refl = sqrt(d^2 + 4h^2) / c
        
        Args:
            offset: Source-receiver offset (m)
            depth: Reflector depth (m)
            velocity: P-wave velocity (m/s)
        
        Returns:
            Travel time in seconds
        """
        distance = np.sqrt(offset**2 + 4 * depth**2)
        return distance / velocity
    
    def generate_synthetic_seismogram(
        self, 
        velocity_map: np.ndarray, 
        source_positions: List[int],
        nt: int = 1000,
        ng: int = 70
    ) -> np.ndarray:
        """
        Generate synthetic seismic data based on velocity map
        
        Args:
            velocity_map: 2D velocity model (nz x nx)
            source_positions: List of source positions
            nt: Number of time samples
            ng: Number of geophones
        
        Returns:
            Seismic data array (ns x nt x ng)
        """
        nz, nx = velocity_map.shape
        ns = len(source_positions)
        seismic_data = np.zeros((ns, nt, ng))
        
        # Time axis
        time_axis = np.arange(nt) * self.dt
        
        for s_idx, source_pos in enumerate(source_positions):
            for g_idx in range(ng):
                # Calculate offset
                offset = abs(g_idx * self.dg - source_pos * self.dx)
                
                # Generate synthetic trace based on velocity structure
                trace = self._generate_trace(velocity_map, offset, time_axis, source_pos)
                seismic_data[s_idx, :, g_idx] = trace
        
        return seismic_data
    
    def _generate_trace(
        self, 
        velocity_map: np.ndarray, 
        offset: float, 
        time_axis: np.ndarray,
        source_pos: int
    ) -> np.ndarray:
        """Generate synthetic trace for given offset and velocity model"""
        nz, nx = velocity_map.shape
        trace = np.zeros_like(time_axis)
        
        # Add direct wave
        if offset > 0:
            surface_velocity = velocity_map[0, source_pos]
            direct_time = self.calculate_direct_wave_traveltime(offset, surface_velocity)
            direct_amplitude = 0.5 * np.exp(-offset / 200.0)  # Distance attenuation
            
            # Add ricker wavelet at direct arrival time
            trace += self._add_ricker_wavelet(time_axis, direct_time, direct_amplitude, 25.0)
        
        # Add reflections from velocity interfaces
        for z_idx in range(1, nz):
            # Check for velocity contrast (interface)
            velocity_above = velocity_map[z_idx-1, source_pos]
            velocity_below = velocity_map[z_idx, source_pos]
            
            if abs(velocity_below - velocity_above) > 200:  # Significant contrast
                depth = z_idx * self.dz
                avg_velocity = (velocity_above + velocity_below) / 2
                
                reflection_time = self.calculate_reflection_traveltime(offset, depth, avg_velocity)
                
                # Reflection coefficient (simplified)
                refl_coeff = (velocity_below - velocity_above) / (velocity_below + velocity_above)
                refl_amplitude = 0.3 * abs(refl_coeff) * np.exp(-offset / 300.0)
                
                # Add reflection
                trace += self._add_ricker_wavelet(time_axis, reflection_time, refl_amplitude, 20.0)
        
        # Add noise
        noise_level = 0.05
        trace += noise_level * np.random.randn(len(time_axis))
        
        return trace
    
    def _add_ricker_wavelet(
        self, 
        time_axis: np.ndarray, 
        arrival_time: float, 
        amplitude: float, 
        frequency: float
    ) -> np.ndarray:
        """Add Ricker wavelet at specified arrival time"""
        # Find closest time sample
        time_idx = np.argmin(np.abs(time_axis - arrival_time))
        
        # Generate Ricker wavelet
        wavelet_length = 61  # samples
        t_wavelet = np.arange(wavelet_length) * self.dt
        t_center = t_wavelet[wavelet_length // 2]
        
        # Ricker wavelet formula
        a = np.pi * frequency * (t_wavelet - t_center)
        ricker = amplitude * (1 - 2 * a**2) * np.exp(-a**2)
        
        # Add to trace
        trace = np.zeros_like(time_axis)
        start_idx = max(0, time_idx - wavelet_length // 2)
        end_idx = min(len(time_axis), time_idx + wavelet_length // 2 + 1)
        
        wavelet_start = max(0, wavelet_length // 2 - time_idx)
        wavelet_end = wavelet_start + (end_idx - start_idx)
        
        if wavelet_end <= len(ricker):
            trace[start_idx:end_idx] = ricker[wavelet_start:wavelet_end]
        
        return trace
    
    def analyze_velocity_quality(self, velocity_map: np.ndarray) -> Dict:
        """Analyze velocity map quality and physical constraints"""
        stats = {
            'min_velocity': float(np.min(velocity_map)),
            'max_velocity': float(np.max(velocity_map)),
            'mean_velocity': float(np.mean(velocity_map)),
            'std_velocity': float(np.std(velocity_map)),
            'velocity_range': float(np.max(velocity_map) - np.min(velocity_map))
        }
        
        # Check physical constraints
        constraints = {
            'within_physical_range': 1500 <= stats['min_velocity'] and stats['max_velocity'] <= 6000,
            'reasonable_gradient': self._check_velocity_gradient(velocity_map),
            'smoothness_score': self._calculate_smoothness(velocity_map)
        }
        
        return {**stats, **constraints}
    
    def _check_velocity_gradient(self, velocity_map: np.ndarray) -> bool:
        """Check if velocity generally increases with depth"""
        nz = velocity_map.shape[0]
        depth_avg = np.mean(velocity_map, axis=1)  # Average velocity at each depth
        
        # Check if velocity generally increases with depth
        gradient_violations = 0
        for z in range(1, nz):
            if depth_avg[z] < depth_avg[z-1] - 100:  # Allow some fluctuation
                gradient_violations += 1
        
        return gradient_violations < nz * 0.3  # Less than 30% violations
    
    def _calculate_smoothness(self, velocity_map: np.ndarray) -> float:
        """Calculate spatial smoothness of velocity map"""
        # Calculate gradients
        grad_x = np.gradient(velocity_map, axis=1)
        grad_z = np.gradient(velocity_map, axis=0)
        
        # RMS gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_z**2)
        smoothness = 1.0 / (1.0 + np.mean(gradient_magnitude) / 100.0)  # Normalize
        
        return float(smoothness)

class SeismicDataAnalyzer:
    """Analyze seismic data characteristics"""
    
    def __init__(self):
        self.dt = 0.001
        self.dg = 10
    
    def analyze_seismic_data(self, seismic_data: np.ndarray) -> Dict:
        """Analyze seismic data characteristics"""
        # Expected shape: (ns, nt, ng) or (nt, ng)
        if len(seismic_data.shape) == 3:
            ns, nt, ng = seismic_data.shape
            data_2d = seismic_data[0]  # Use first shot
        else:
            nt, ng = seismic_data.shape
            ns = 1
            data_2d = seismic_data
        
        stats = {
            'shape': seismic_data.shape,
            'num_sources': ns if len(seismic_data.shape) == 3 else 1,
            'num_time_samples': nt,
            'num_geophones': ng,
            'time_duration': nt * self.dt,
            'geophone_aperture': ng * self.dg,
            'amplitude_range': (float(np.min(seismic_data)), float(np.max(seismic_data))),
            'rms_amplitude': float(np.sqrt(np.mean(seismic_data**2))),
            'dominant_frequency': self._estimate_dominant_frequency(data_2d)
        }
        
        return stats
    
    def _estimate_dominant_frequency(self, data_2d: np.ndarray) -> float:
        """Estimate dominant frequency of seismic data"""
        # Take FFT of central trace
        central_trace = data_2d[:, data_2d.shape[1] // 2]
        
        # Apply window to reduce edge effects
        windowed = central_trace * np.hanning(len(central_trace))
        
        # FFT
        fft_data = np.fft.fft(windowed)
        freqs = np.fft.fftfreq(len(windowed), self.dt)
        
        # Find peak frequency (positive frequencies only)
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = np.abs(fft_data[:len(fft_data)//2])
        
        # Find dominant frequency
        peak_idx = np.argmax(positive_fft[1:]) + 1  # Skip DC component
        dominant_freq = positive_freqs[peak_idx]
        
        return float(dominant_freq)

def generate_physics_tutorial_data() -> Tuple[np.ndarray, np.ndarray]:
    """Generate tutorial data demonstrating wave physics concepts"""
    # Create layered velocity model (similar to OpenFWI tutorial)
    nz, nx = 70, 70
    velocity_map = np.zeros((nz, nx))
    
    # Layer 1: Water/sediment (shallow, low velocity)
    velocity_map[0:10, :] = 1500 + np.random.normal(0, 50, (10, nx))
    
    # Layer 2: Sedimentary rock
    velocity_map[10:25, :] = 2200 + np.random.normal(0, 100, (15, nx))
    
    # Layer 3: Limestone
    velocity_map[25:40, :] = 3500 + np.random.normal(0, 150, (15, nx))
    
    # Layer 4: Sandstone
    velocity_map[40:55, :] = 4200 + np.random.normal(0, 200, (15, nx))
    
    # Layer 5: Basement (deep, high velocity)
    velocity_map[55:70, :] = 5500 + np.random.normal(0, 250, (15, nx))
    
    # Add some lateral variation
    for x in range(nx):
        velocity_map[:, x] += 200 * np.sin(2 * np.pi * x / nx)
    
    # Ensure physical constraints
    velocity_map = np.clip(velocity_map, 1500, 6000)
    
    # Generate corresponding seismic data
    calculator = WavePhysicsCalculator()
    source_positions = [10, 20, 30, 40, 50]  # 5 source positions
    seismic_data = calculator.generate_synthetic_seismogram(
        velocity_map, source_positions, nt=1000, ng=70
    )
    
    return velocity_map, seismic_data