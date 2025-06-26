import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import re

class SubmissionFormatter:
    """
    Format model predictions for Kaggle competition submission
    Handles the specific requirements of the FWI competition
    """
    
    def __init__(self):
        pass
    
    def format_predictions(
        self, 
        predictions: Dict[str, np.ndarray], 
        output_height: int = 70, 
        output_width: int = 70
    ) -> pd.DataFrame:
        """
        Format predictions into competition submission format
        
        Args:
            predictions: Dictionary mapping file_id -> prediction array
            output_height: Expected height of velocity maps
            output_width: Expected width of velocity maps
        
        Returns:
            DataFrame in submission format
        """
        submission_rows = []
        
        for file_id, prediction in predictions.items():
            # Ensure prediction has correct shape
            processed_pred = self._process_prediction(prediction, output_height, output_width)
            
            # Create rows for each y position
            for y_pos in range(output_height):
                row_id = f"{file_id}_y_{y_pos}"
                
                # Extract odd-column values (x_1, x_3, x_5, ..., x_69)
                row_values = [row_id]
                for x_pos in range(1, output_width, 2):  # Odd positions only
                    value = processed_pred[y_pos, x_pos]
                    row_values.append(float(value))
                
                submission_rows.append(row_values)
        
        # Create column names
        columns = ['oid_ypos']
        for x_pos in range(1, output_width, 2):
            columns.append(f'x_{x_pos}')
        
        # Create DataFrame
        submission_df = pd.DataFrame(submission_rows, columns=columns)
        
        # Sort by oid_ypos for consistency
        submission_df = submission_df.sort_values('oid_ypos').reset_index(drop=True)
        
        return submission_df
    
    def _process_prediction(self, prediction: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
        """
        Process prediction array to match required output dimensions
        
        Args:
            prediction: Raw prediction array
            target_height: Required height
            target_width: Required width
        
        Returns:
            Processed 2D array
        """
        # Handle different input shapes
        if len(prediction.shape) == 4:  # (batch, channels, height, width)
            pred_2d = prediction[0, 0]  # Take first batch and channel
        elif len(prediction.shape) == 3:  # (batch, height, width) or (channels, height, width)
            pred_2d = prediction[0]  # Take first dimension
        elif len(prediction.shape) == 2:  # (height, width)
            pred_2d = prediction
        else:
            raise ValueError(f"Unexpected prediction shape: {prediction.shape}")
        
        # Resize if necessary
        if pred_2d.shape != (target_height, target_width):
            pred_2d = self._resize_prediction(pred_2d, target_height, target_width)
        
        # Ensure realistic velocity values
        pred_2d = self._apply_velocity_constraints(pred_2d)
        
        return pred_2d
    
    def _resize_prediction(self, prediction: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
        """Resize prediction to target dimensions using interpolation"""
        try:
            from scipy import ndimage
            # Use scipy for high-quality interpolation
            zoom_factors = (target_height / prediction.shape[0], target_width / prediction.shape[1])
            resized = ndimage.zoom(prediction, zoom_factors, order=1)
            return resized
        except ImportError:
            # Fallback to simple nearest neighbor interpolation
            return self._simple_resize(prediction, target_height, target_width)
    
    def _simple_resize(self, prediction: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
        """Simple resize using nearest neighbor interpolation"""
        h_old, w_old = prediction.shape
        h_scale = h_old / target_height
        w_scale = w_old / target_width
        
        resized = np.zeros((target_height, target_width))
        
        for i in range(target_height):
            for j in range(target_width):
                old_i = int(i * h_scale)
                old_j = int(j * w_scale)
                old_i = min(old_i, h_old - 1)
                old_j = min(old_j, w_old - 1)
                resized[i, j] = prediction[old_i, old_j]
        
        return resized
    
    def _apply_velocity_constraints(self, prediction: np.ndarray, min_vel: float = 1500.0, max_vel: float = 6000.0) -> np.ndarray:
        """Apply realistic velocity constraints"""
        # Clip to reasonable velocity range
        constrained = np.clip(prediction, min_vel, max_vel)
        
        # Replace any NaN or infinite values with default velocity
        default_velocity = 3000.0
        constrained = np.where(np.isfinite(constrained), constrained, default_velocity)
        
        return constrained
    
    def validate_submission(self, submission_df: pd.DataFrame, expected_files: List[str] = None) -> Dict[str, bool]:
        """
        Validate submission format against Kaggle competition requirements
        
        Args:
            submission_df: Submission DataFrame
            expected_files: List of expected file IDs (optional)
        
        Returns:
            Dictionary of validation results with detailed checks
        """
        validation_results = {}
        errors = []
        warnings = []
        
        # Check column format - CRITICAL: Only odd positions x_1, x_3, x_5, ..., x_69
        expected_cols = ['oid_ypos'] + [f'x_{i}' for i in range(1, 70, 2)]  # 35 odd columns
        actual_cols = list(submission_df.columns)
        validation_results['correct_columns'] = actual_cols == expected_cols
        if not validation_results['correct_columns']:
            errors.append(f"Incorrect columns. Expected {len(expected_cols)} cols: {expected_cols[:3]}...{expected_cols[-2:]}")
            errors.append(f"Got {len(actual_cols)} cols: {actual_cols[:3]}...{actual_cols[-2:] if len(actual_cols) >= 2 else actual_cols}")
        
        # Check for missing values - CRITICAL: No NaN allowed
        validation_results['no_missing_values'] = not submission_df.isnull().any().any()
        if not validation_results['no_missing_values']:
            missing_count = submission_df.isnull().sum().sum()
            errors.append(f"Contains {missing_count} missing values - not allowed")
        
        # Check value ranges for MAE evaluation
        numeric_cols = [col for col in submission_df.columns if col != 'oid_ypos']
        if numeric_cols:
            min_val = submission_df[numeric_cols].min().min()
            max_val = submission_df[numeric_cols].max().max()
            validation_results['reasonable_velocity_range'] = (1000 <= min_val <= max_val <= 10000)
            if not validation_results['reasonable_velocity_range']:
                warnings.append(f"Velocity range {min_val:.1f}-{max_val:.1f} outside typical 1500-6000 m/s")
        
        # Check oid_ypos format - CRITICAL: fileID_y_rowNumber
        import re
        valid_format = True
        if len(submission_df) > 0:
            sample_oid = submission_df['oid_ypos'].iloc[0]
            pattern = r'^[a-f0-9]+_y_\d+$'
            valid_format = bool(re.match(pattern, sample_oid))
            if not valid_format:
                errors.append(f"Invalid oid_ypos format. Expected 'fileID_y_rowNum', got '{sample_oid}'")
        validation_results['valid_oid_format'] = valid_format
        
        # Check rows per file - CRITICAL: Exactly 70 rows (y_0 to y_69) per file
        if len(submission_df) > 0:
            file_groups = submission_df['oid_ypos'].apply(lambda x: x.split('_y_')[0])
            rows_per_file = file_groups.value_counts()
            validation_results['correct_rows_per_file'] = all(count == 70 for count in rows_per_file.values)
            if not validation_results['correct_rows_per_file']:
                incorrect_files = rows_per_file[rows_per_file != 70]
                errors.append(f"Files with incorrect row count: {dict(incorrect_files)}")
            
            # Check y_ positions are 0-69
            y_positions = submission_df['oid_ypos'].apply(lambda x: int(x.split('_y_')[1]))
            expected_y_range = set(range(70))
            for file_id in file_groups.unique():
                file_rows = submission_df[file_groups == file_id]
                file_y_positions = set(file_rows['oid_ypos'].apply(lambda x: int(x.split('_y_')[1])))
                if file_y_positions != expected_y_range:
                    missing_y = expected_y_range - file_y_positions
                    extra_y = file_y_positions - expected_y_range
                    if missing_y:
                        errors.append(f"File {file_id} missing y positions: {sorted(missing_y)[:5]}")
                    if extra_y:
                        errors.append(f"File {file_id} has extra y positions: {sorted(extra_y)[:5]}")
        
        # Check expected files are present
        if expected_files:
            unique_files = set(row.split('_y_')[0] for row in submission_df['oid_ypos'])
            expected_file_set = set(expected_files)
            validation_results['all_files_present'] = unique_files == expected_file_set
            if not validation_results['all_files_present']:
                missing_files = expected_file_set - unique_files
                extra_files = unique_files - expected_file_set
                if missing_files:
                    errors.append(f"Missing files: {list(missing_files)[:5]}")
                if extra_files:
                    errors.append(f"Extra files: {list(extra_files)[:5]}")
        
        # MAE evaluation readiness check
        validation_results['mae_ready'] = (
            validation_results['correct_columns'] and 
            validation_results['no_missing_values'] and 
            validation_results['valid_oid_format'] and
            validation_results['correct_rows_per_file']
        )
        
        validation_results['errors'] = errors
        validation_results['warnings'] = warnings
        validation_results['is_valid'] = len(errors) == 0
        
        return validation_results
    
    def create_sample_submission(self, test_files: List[str], output_height: int = 70, output_width: int = 70) -> pd.DataFrame:
        """
        Create a sample submission with default values
        
        Args:
            test_files: List of test file names
            output_height: Height of velocity maps
            output_width: Width of velocity maps
        
        Returns:
            Sample submission DataFrame
        """
        submission_rows = []
        default_velocity = 3000.0
        
        for file_name in test_files:
            # Extract file ID (remove .npy extension)
            file_id = file_name.replace('.npy', '')
            
            for y_pos in range(output_height):
                row_id = f"{file_id}_y_{y_pos}"
                
                # Create row with default values for odd columns
                row_values = [row_id]
                for x_pos in range(1, output_width, 2):
                    row_values.append(default_velocity)
                
                submission_rows.append(row_values)
        
        # Create column names
        columns = ['oid_ypos']
        for x_pos in range(1, output_width, 2):
            columns.append(f'x_{x_pos}')
        
        return pd.DataFrame(submission_rows, columns=columns)
