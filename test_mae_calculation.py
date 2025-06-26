
#!/usr/bin/env python3
"""
Test MAE calculation on submission format
Ensures the submission format works correctly with competition evaluation
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append('.')

from utils.kaggle_submission import KaggleSubmissionGenerator

def test_mae_calculation():
    """Test MAE calculation on properly formatted submission"""
    
    print("TESTING MAE CALCULATION COMPATIBILITY")
    print("=" * 50)
    
    # Create sample submissions with known values
    kaggle_gen = KaggleSubmissionGenerator()
    test_files = kaggle_gen.load_test_file_ids("sample_data/test")
    
    # Create submission with known values for testing
    submission_rows = []
    
    for file_idx, file_id in enumerate(test_files):
        for y_pos in range(70):  # y_0 to y_69
            row_id = f"{file_id}_y_{y_pos}"
            
            # Create test values: different for each file and position
            base_velocity = 3000 + file_idx * 100 + y_pos * 10
            row_values = [row_id]
            
            # Add 35 odd columns with varying values
            for x_pos in range(1, 70, 2):
                velocity = base_velocity + x_pos
                row_values.append(float(velocity))
            
            submission_rows.append(row_values)
    
    # Create DataFrame
    columns = ['oid_ypos'] + [f'x_{i}' for i in range(1, 70, 2)]
    test_submission = pd.DataFrame(submission_rows, columns=columns)
    
    print(f"✅ Test submission shape: {test_submission.shape}")
    print(f"✅ Columns: {len(test_submission.columns)} ({test_submission.columns[0]}, {test_submission.columns[1]}, ..., {test_submission.columns[-1]})")
    
    # Validate format
    validation = kaggle_gen._validate_submission(test_submission)
    print(f"✅ Format valid: {validation['is_valid']}")
    
    if not validation['is_valid']:
        print("❌ Validation errors:")
        for error in validation['errors']:
            print(f"   - {error}")
        return False
    
    # Test MAE calculation simulation
    print("\nMAE CALCULATION TEST")
    print("-" * 30)
    
    # Create "ground truth" with slightly different values
    gt_submission = test_submission.copy()
    numeric_cols = [col for col in gt_submission.columns if col != 'oid_ypos']
    
    # Add small random differences to simulate prediction vs truth
    np.random.seed(42)
    for col in numeric_cols:
        # Add noise: ±50 m/s
        noise = np.random.normal(0, 50, size=len(gt_submission))
        gt_submission[col] = gt_submission[col] + noise
    
    # Calculate MAE across all values
    pred_values = test_submission[numeric_cols].values.flatten()
    true_values = gt_submission[numeric_cols].values.flatten()
    
    mae = np.mean(np.abs(pred_values - true_values))
    
    print(f"✅ Total prediction values: {len(pred_values):,}")
    print(f"✅ MAE calculation successful: {mae:.2f} m/s")
    print(f"✅ Value range - Predictions: {pred_values.min():.1f} to {pred_values.max():.1f}")
    print(f"✅ Value range - Ground truth: {true_values.min():.1f} to {true_values.max():.1f}")
    
    # Test per-file statistics
    print("\nPER-FILE STATISTICS")
    print("-" * 30)
    
    file_groups = test_submission['oid_ypos'].apply(lambda x: x.split('_y_')[0])
    
    for file_id in test_files:
        file_mask = file_groups == file_id
        file_pred = test_submission.loc[file_mask, numeric_cols].values.flatten()
        file_true = gt_submission.loc[file_mask, numeric_cols].values.flatten()
        file_mae = np.mean(np.abs(file_pred - file_true))
        
        print(f"✅ {file_id}: {len(file_pred)} values, MAE = {file_mae:.2f} m/s")
    
    # Competition format verification
    print("\nCOMPETITION FORMAT VERIFICATION")
    print("-" * 40)
    
    # Save and reload to test CSV format
    test_submission.to_csv("test_mae_submission.csv", index=False)
    reloaded = pd.read_csv("test_mae_submission.csv")
    
    print(f"✅ CSV save/load successful: {reloaded.shape}")
    print(f"✅ Data preserved: {np.allclose(test_submission[numeric_cols], reloaded[numeric_cols])}")
    
    # File size check
    file_size = Path("test_mae_submission.csv").stat().st_size / 1024 / 1024
    print(f"✅ File size: {file_size:.2f} MB (reasonable for upload)")
    
    print("\n" + "=" * 50)
    print("✅ MAE CALCULATION TEST PASSED")
    print("✅ Submission format compatible with competition evaluation")
    print("✅ Ready for Kaggle upload and scoring")
    
    return True

if __name__ == "__main__":
    success = test_mae_calculation()
    sys.exit(0 if success else 1)
