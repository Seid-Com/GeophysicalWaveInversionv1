import os
import glob
import numpy as np
from pathlib import Path
import random
from typing import Dict, List, Tuple, Optional, Union

class SeismicDataLoader:
    """
    Data loader for seismic waveform data and velocity maps
    Handles Vel, Fault, and Style dataset families
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.families = self._detect_families()
        
    def _detect_families(self) -> Dict[str, Dict]:
        """Detect available dataset families and their file patterns"""
        families = {}
        
        # Check for Vel and Style families (data/*.npy, model/*.npy)
        data_dir = self.data_dir / "data"
        model_dir = self.data_dir / "model"
        
        if data_dir.exists() and model_dir.exists():
            data_files = list(data_dir.glob("*.npy"))
            model_files = list(model_dir.glob("*.npy"))
            
            if data_files and model_files:
                # Determine if Vel or Style based on file names or content
                families["Vel_Style"] = {
                    "pattern": "data/*.npy + model/*.npy",
                    "data_files": data_files,
                    "model_files": model_files,
                    "file_count": len(data_files)
                }
        
        # Check for Fault family (seis_*.npy, vel_*.npy)
        fault_seis_files = list(self.data_dir.glob("seis_*.npy"))
        fault_vel_files = list(self.data_dir.glob("vel_*.npy"))
        
        if fault_seis_files and fault_vel_files:
            families["Fault"] = {
                "pattern": "seis_*.npy + vel_*.npy",
                "data_files": fault_seis_files,
                "model_files": fault_vel_files,
                "file_count": len(fault_seis_files)
            }
        
        return families
    
    def get_data_statistics(self) -> Dict:
        """Get statistics about available data"""
        total_files = sum(family["file_count"] for family in self.families.values())
        estimated_samples = total_files * 500  # Each file contains 500 samples
        
        stats = {
            "total_files": total_files,
            "estimated_samples": estimated_samples,
            "families": {}
        }
        
        for name, family in self.families.items():
            stats["families"][name] = {
                "file_count": family["file_count"],
                "pattern": family["pattern"],
                "sample_files": [str(f.name) for f in family["data_files"][:5]]
            }
        
        return stats
    
    def load_sample_data(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load a sample seismic data and velocity map for visualization"""
        if not self.families:
            return None
        
        # Get first available family
        family_name, family_info = next(iter(self.families.items()))
        
        # Load first data file and corresponding model file
        data_file = family_info["data_files"][0]
        
        # Find corresponding model file
        if family_name == "Fault":
            # For fault family: seis_10_1_0.npy -> vel_10_1_0.npy
            base_name = data_file.name.replace("seis_", "vel_")
            model_file = data_file.parent / base_name
        else:
            # For Vel/Style families: data1.npy -> model1.npy
            base_name = data_file.name.replace("data", "model")
            model_file = self.data_dir / "model" / base_name
        
        try:
            seismic_data = np.load(data_file)
            velocity_map = np.load(model_file)
            return seismic_data, velocity_map
        except Exception:
            return None
    
    def get_file_pairs(self, family_names: List[str], max_files_per_family: int = None) -> List[Tuple[Path, Path]]:
        """
        Get paired data and model files for specified families
        
        Args:
            family_names: List of family names to include
            max_files_per_family: Maximum number of files per family
        
        Returns:
            List of (data_file, model_file) tuples
        """
        file_pairs = []
        
        for family_name in family_names:
            if family_name not in self.families:
                continue
            
            family_info = self.families[family_name]
            data_files = family_info["data_files"]
            
            # Limit files if specified
            if max_files_per_family:
                data_files = data_files[:max_files_per_family]
            
            for data_file in data_files:
                # Find corresponding model file
                if family_name == "Fault":
                    base_name = data_file.name.replace("seis_", "vel_")
                    model_file = data_file.parent / base_name
                else:
                    base_name = data_file.name.replace("data", "model")
                    model_file = self.data_dir / "model" / base_name
                
                if model_file.exists():
                    file_pairs.append((data_file, model_file))
        
        return file_pairs
    
    def load_batch_data(
        self, 
        file_pairs: List[Tuple[Path, Path]], 
        batch_size: int,
        shuffle: bool = True
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Load batches of seismic data and velocity maps
        
        Args:
            file_pairs: List of (data_file, model_file) tuples
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle file order
        
        Returns:
            List of (seismic_batch, velocity_batch) tuples
        """
        if shuffle:
            file_pairs = file_pairs.copy()
            random.shuffle(file_pairs)
        
        batches = []
        current_seismic = []
        current_velocity = []
        current_batch_size = 0
        
        for data_file, model_file in file_pairs:
            try:
                # Load data
                seismic_data = np.load(data_file)  # Shape: (500, sources, time, receivers)
                velocity_data = np.load(model_file)  # Shape: (500, height, width)
                
                # Add samples to current batch
                for i in range(seismic_data.shape[0]):
                    current_seismic.append(seismic_data[i])
                    current_velocity.append(velocity_data[i])
                    current_batch_size += 1
                    
                    # Create batch when full
                    if current_batch_size >= batch_size:
                        batch_seismic = np.stack(current_seismic[:batch_size])
                        batch_velocity = np.stack(current_velocity[:batch_size])
                        batches.append((batch_seismic, batch_velocity))
                        
                        # Reset for next batch
                        current_seismic = current_seismic[batch_size:]
                        current_velocity = current_velocity[batch_size:]
                        current_batch_size = len(current_seismic)
                        
            except Exception as e:
                print(f"Error loading {data_file}: {e}")
                continue
        
        # Add remaining samples as final batch
        if current_seismic:
            batch_seismic = np.stack(current_seismic)
            batch_velocity = np.stack(current_velocity)
            batches.append((batch_seismic, batch_velocity))
        
        return batches
    
    def create_train_val_split(
        self, 
        family_names: List[str], 
        max_files_per_family: int = None,
        validation_split: float = 0.2,
        random_seed: int = 42
    ) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
        """
        Create train/validation split
        
        Args:
            family_names: Dataset families to include
            max_files_per_family: Maximum files per family
            validation_split: Fraction for validation
            random_seed: Random seed for reproducibility
        
        Returns:
            (train_pairs, val_pairs) tuple
        """
        random.seed(random_seed)
        
        all_pairs = self.get_file_pairs(family_names, max_files_per_family)
        random.shuffle(all_pairs)
        
        val_size = int(len(all_pairs) * validation_split)
        
        val_pairs = all_pairs[:val_size]
        train_pairs = all_pairs[val_size:]
        
        return train_pairs, val_pairs


class TestDataLoader:
    """
    Data loader for test data (single .npy files without ground truth)
    """
    
    def __init__(self, test_dir: str):
        self.test_dir = Path(test_dir)
        self.test_files = list(self.test_dir.glob("*.npy"))
    
    def get_test_files(self) -> List[Path]:
        """Get list of test files"""
        return self.test_files
    
    def load_test_file(self, file_path: Path) -> np.ndarray:
        """Load a single test file"""
        return np.load(file_path)
    
    def get_file_id(self, file_path: Path) -> str:
        """Extract file ID from path"""
        return file_path.stem
