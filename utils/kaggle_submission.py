"""
Kaggle submission utilities for FWI competition
Handles proper formatting and validation for competition requirements
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import os
from pathlib import Path
import gc

class KaggleSubmissionGenerator:
    """Generate competition-ready submission files"""
    
    def __init__(self):
        self.required_columns = ['oid_ypos'] + [f'x_{i}' for i in range(1, 70, 2)]  # Only odd columns
        self.expected_height = 70
        self.expected_width = 70
    
    def create_submission_from_predictions(
        self, 
        predictions: Dict[str, np.ndarray], 
        output_path: str = "submission.csv",
        chunk_size: int = 10000
    ) -> pd.DataFrame:
        """
        Create submission file from model predictions
        
        Args:
            predictions: Dict mapping file_id -> prediction array (2D velocity maps)
            output_path: Path to save submission file
            chunk_size: Number of rows to process in chunks for memory efficiency
        
        Returns:
            DataFrame in competition format
        """
        total_files = len(predictions)
        total_rows = total_files * self.expected_height
        
        print(f"Processing {total_files} files -> {total_rows:,} rows...")
        
        # Process in chunks for memory efficiency
        chunk_dfs = []
        current_chunk = []
        
        for file_idx, (file_id, prediction) in enumerate(predictions.items()):
            # Ensure prediction is 2D with correct shape
            processed_pred = self._process_prediction(prediction)
            
            # Create rows for each y position (0 to 69)
            for y_pos in range(self.expected_height):
                row_id = f"{file_id}_y_{y_pos}"
                
                # Extract values for odd columns only (x_1, x_3, x_5, ..., x_69)
                row_values = [row_id]
                for x_pos in range(1, self.expected_width, 2):
                    value = processed_pred[y_pos, x_pos]
                    row_values.append(float(value))
                
                current_chunk.append(row_values)
                
                # Process chunk when it reaches chunk_size
                if len(current_chunk) >= chunk_size:
                    chunk_df = pd.DataFrame(current_chunk, columns=self.required_columns)
                    chunk_dfs.append(chunk_df)
                    current_chunk = []
            
            # Progress indicator for large datasets
            if (file_idx + 1) % 100 == 0 or file_idx == total_files - 1:
                print(f"Processed {file_idx + 1}/{total_files} files ({((file_idx + 1) / total_files * 100):.1f}%)")
        
        # Process remaining rows
        if current_chunk:
            chunk_df = pd.DataFrame(current_chunk, columns=self.required_columns)
            chunk_dfs.append(chunk_df)
        
        # Combine all chunks efficiently
        print("Combining chunks...")
        submission_df = pd.concat(chunk_dfs, ignore_index=True)
        
        # Sort by oid_ypos for consistency
        print("Sorting rows...")
        submission_df = submission_df.sort_values('oid_ypos').reset_index(drop=True)
        
        # Validate submission format
        print("Validating format...")
        validation_results = self._validate_submission(submission_df)
        if not validation_results['is_valid']:
            raise ValueError(f"Invalid submission format: {validation_results['errors']}")
        
        # Save to file with compression for large files
        print(f"Saving to {output_path}...")
        if total_rows > 100000:  # Use compression for large files
            submission_df.to_csv(output_path, index=False, compression='gzip')
            output_path = output_path.replace('.csv', '.csv.gz')
        else:
            submission_df.to_csv(output_path, index=False)
        
        print(f"Submission saved to {output_path}")
        print(f"Submission shape: {submission_df.shape}")
        print(f"Files included: {len(set(row.split('_y_')[0] for row in submission_df['oid_ypos']))}")
        print(f"File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
        
        return submission_df
    
    def _process_prediction(self, prediction: np.ndarray) -> np.ndarray:
        """Process prediction to match competition requirements"""
        # Handle different input shapes
        if len(prediction.shape) == 4:  # (batch, channels, height, width)
            pred_2d = prediction[0, 0]
        elif len(prediction.shape) == 3:  # (batch, height, width) or (channels, height, width)
            pred_2d = prediction[0]
        elif len(prediction.shape) == 2:  # (height, width)
            pred_2d = prediction
        else:
            raise ValueError(f"Unexpected prediction shape: {prediction.shape}")
        
        # Resize if necessary
        if pred_2d.shape != (self.expected_height, self.expected_width):
            pred_2d = self._resize_to_target(pred_2d, self.expected_height, self.expected_width)
        
        # Apply velocity constraints (typical seismic velocities)
        pred_2d = np.clip(pred_2d, 1500.0, 6000.0)
        
        # Handle any NaN or infinite values
        pred_2d = np.where(np.isfinite(pred_2d), pred_2d, 3000.0)
        
        return pred_2d
    
    def _resize_to_target(self, prediction: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
        """Resize prediction to target dimensions"""
        try:
            from scipy import ndimage
            zoom_factors = (target_height / prediction.shape[0], target_width / prediction.shape[1])
            return ndimage.zoom(prediction, zoom_factors, order=1)
        except ImportError:
            # Fallback to simple interpolation
            return self._simple_resize(prediction, target_height, target_width)
    
    def _simple_resize(self, prediction: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
        """Simple nearest neighbor resize"""
        h_old, w_old = prediction.shape
        h_scale = h_old / target_height
        w_scale = w_old / target_width
        
        resized = np.zeros((target_height, target_width))
        for i in range(target_height):
            for j in range(target_width):
                old_i = min(int(i * h_scale), h_old - 1)
                old_j = min(int(j * w_scale), w_old - 1)
                resized[i, j] = prediction[old_i, old_j]
        
        return resized
    
    def _validate_submission(self, submission_df: pd.DataFrame) -> Dict:
        """Validate submission format against Kaggle competition requirements for MAE evaluation"""
        errors = []
        warnings = []
        
        # CRITICAL: Check column names - only odd x positions allowed
        expected_columns = ['oid_ypos'] + [f'x_{i}' for i in range(1, 70, 2)]  # 35 odd columns
        actual_columns = list(submission_df.columns)
        
        if actual_columns != expected_columns:
            errors.append(f"Incorrect columns for MAE evaluation")
            errors.append(f"Expected {len(expected_columns)} columns: oid_ypos, x_1, x_3, x_5, ..., x_69")
            errors.append(f"Got {len(actual_columns)} columns: {actual_columns}")
        
        # CRITICAL: Check for missing values (will break MAE calculation)
        if submission_df.isnull().any().any():
            null_count = submission_df.isnull().sum().sum()
            errors.append(f"Contains {null_count} missing values - MAE calculation requires all values")
        
        # CRITICAL: Check oid_ypos format for proper stacking
        import re
        if len(submission_df) > 0:
            sample_oid = submission_df['oid_ypos'].iloc[0]
            pattern = r'^[a-f0-9]+_y_\d+$'
            if not re.match(pattern, sample_oid):
                errors.append(f"Invalid oid_ypos format: '{sample_oid}' (should be 'fileID_y_rowNumber')")
        
        # CRITICAL: Check value types (must be numeric for MAE)
        numeric_cols = [col for col in submission_df.columns if col != 'oid_ypos']
        if numeric_cols:
            for col in numeric_cols:
                if not pd.api.types.is_numeric_dtype(submission_df[col]):
                    errors.append(f"Column '{col}' is not numeric - required for MAE calculation")
            
            # Check value ranges
            min_val = submission_df[numeric_cols].min().min()
            max_val = submission_df[numeric_cols].max().max()
            
            if min_val < 1000 or max_val > 10000:
                warnings.append(f"Velocity values {min_val:.1f} to {max_val:.1f} outside typical seismic range (1500-6000 m/s)")
            
            # Check for infinite values
            if not np.isfinite(submission_df[numeric_cols].values).all():
                errors.append("Contains infinite values - not allowed for MAE evaluation")
        
        # CRITICAL: Check rows per file - must be exactly 70 (all y positions)
        if len(submission_df) > 0:
            file_groups = submission_df['oid_ypos'].apply(lambda x: x.split('_y_')[0])
            file_counts = file_groups.value_counts()
            
            if not all(count == 70 for count in file_counts.values):
                errors.append("Each file must have exactly 70 rows (y_0 to y_69) for proper MAE evaluation")
                incorrect_files = file_counts[file_counts != 70]
                errors.append(f"Files with wrong row count: {dict(incorrect_files.head())}")
            
            # Check y positions are sequential 0-69
            for file_id in file_groups.unique():
                file_mask = file_groups == file_id
                file_y_positions = submission_df.loc[file_mask, 'oid_ypos'].apply(
                    lambda x: int(x.split('_y_')[1])
                ).sort_values()
                
                expected_positions = list(range(70))
                if list(file_y_positions) != expected_positions:
                    missing = set(expected_positions) - set(file_y_positions)
                    if missing:
                        errors.append(f"File {file_id} missing y positions: {sorted(missing)[:10]}")
        
        # Overall MAE readiness
        mae_ready = len(errors) == 0 and len(submission_df) > 0
        
        return {
            'is_valid': len(errors) == 0,
            'mae_ready': mae_ready,
            'errors': errors,
            'warnings': warnings,
            'total_rows': len(submission_df),
            'total_files': len(submission_df['oid_ypos'].apply(lambda x: x.split('_y_')[0]).unique()) if len(submission_df) > 0 else 0
        }
    
    def create_sample_submission(self, test_file_ids: List[str], output_path: str = "sample_submission.csv", 
                               chunk_size: int = 10000) -> pd.DataFrame:
        """Create a sample submission with default values"""
        total_files = len(test_file_ids)
        total_rows = total_files * self.expected_height
        default_velocity = 3000.0
        
        print(f"Creating sample submission for {total_files} files -> {total_rows:,} rows...")
        
        # Process in chunks for large datasets
        chunk_dfs = []
        current_chunk = []
        
        for file_idx, file_id in enumerate(test_file_ids):
            for y_pos in range(self.expected_height):
                row_id = f"{file_id}_y_{y_pos}"
                row_values = [row_id] + [default_velocity] * 35  # 35 odd columns
                current_chunk.append(row_values)
                
                # Process chunk when it reaches chunk_size
                if len(current_chunk) >= chunk_size:
                    chunk_df = pd.DataFrame(current_chunk, columns=self.required_columns)
                    chunk_dfs.append(chunk_df)
                    current_chunk = []
            
            # Progress indicator for large datasets
            if (file_idx + 1) % 1000 == 0 or file_idx == total_files - 1:
                print(f"Processed {file_idx + 1}/{total_files} files")
        
        # Process remaining rows
        if current_chunk:
            chunk_df = pd.DataFrame(current_chunk, columns=self.required_columns)
            chunk_dfs.append(chunk_df)
        
        # Combine chunks
        submission_df = pd.concat(chunk_dfs, ignore_index=True)
        
        # Save with compression for large files
        if total_rows > 100000:
            submission_df.to_csv(output_path, index=False, compression='gzip')
            output_path = output_path.replace('.csv', '.csv.gz')
        else:
            submission_df.to_csv(output_path, index=False)
        
        print(f"Sample submission saved: {output_path}")
        return submission_df
    
    def load_test_file_ids(self, test_dir: str) -> List[str]:
        """Extract file IDs from test directory"""
        test_path = Path(test_dir)
        test_files = list(test_path.glob("*.npy"))
        
        # Extract file IDs (remove .npy extension)
        file_ids = [f.stem for f in test_files]
        
        return sorted(file_ids)
    
    def preview_submission(self, submission_df: pd.DataFrame, num_rows: int = 10):
        """Preview submission format"""
        print(f"Submission Preview (first {num_rows} rows):")
        print(submission_df.head(num_rows))
        print(f"\nTotal rows: {len(submission_df)}")
        print(f"Total columns: {len(submission_df.columns)}")
        
        # Count unique files
        unique_files = len(set(row.split('_y_')[0] for row in submission_df['oid_ypos']))
        print(f"Unique files: {unique_files}")
        
        # Value statistics
        numeric_cols = [col for col in submission_df.columns if col != 'oid_ypos']
        print(f"Value range: {submission_df[numeric_cols].min().min():.2f} to {submission_df[numeric_cols].max().max():.2f}")
        
        return {
            'total_rows': len(submission_df),
            'total_columns': len(submission_df.columns),
            'unique_files': unique_files,
            'value_range': (submission_df[numeric_cols].min().min(), submission_df[numeric_cols].max().max())
        }

def generate_kaggle_submission_code():
    """Generate example code for creating Kaggle submissions"""
    code = '''
# Example: Creating a Kaggle submission from FWI model predictions

from utils.kaggle_submission import KaggleSubmissionGenerator
import numpy as np

# Initialize submission generator
submission_gen = KaggleSubmissionGenerator()

# Example: Load test file IDs
test_file_ids = submission_gen.load_test_file_ids("sample_data/test")
print(f"Found {len(test_file_ids)} test files: {test_file_ids}")

# Example predictions (replace with your model predictions)
predictions = {}
for file_id in test_file_ids:
    # Create synthetic prediction for demonstration (70x70 velocity map)
    velocity_map = np.random.normal(3000, 500, (70, 70))
    velocity_map = np.clip(velocity_map, 1500, 6000)  # Realistic velocity range
    predictions[file_id] = velocity_map

# Create submission file
submission_df = submission_gen.create_submission_from_predictions(
    predictions=predictions,
    output_path="my_submission.csv"
)

# Preview submission
stats = submission_gen.preview_submission(submission_df)

# Validate submission format
validation = submission_gen._validate_submission(submission_df)
print(f"Submission valid: {validation['is_valid']}")
if validation['warnings']:
    print(f"Warnings: {validation['warnings']}")
'''
    return code