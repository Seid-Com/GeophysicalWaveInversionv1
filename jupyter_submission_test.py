
#!/usr/bin/env python3
"""
Simplified submission test for Jupyter notebook environment
This script creates a working submission example even without actual test files
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Add current directory to path
sys.path.append('.')
sys.path.append('..')

def create_jupyter_submission_test():
    """Create a test submission that works in Jupyter environment"""
    
    print("=== Jupyter FWI Submission Test ===\n")
    
    # Import after adding to path
    try:
        from utils.kaggle_submission import KaggleSubmissionGenerator
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running from the correct directory")
        return None
    
    # Initialize submission generator
    kaggle_gen = KaggleSubmissionGenerator()
    
    # Create sample test file IDs (using actual files from your directory structure)
    test_file_ids = ["000030dca2", "000031dca3", "000032dca4", "000033dca5", "000034dca6"]
    print(f"Using test files: {test_file_ids}\n")
    
    # Create sample predictions (70x70 velocity maps)
    print("Creating sample velocity predictions...")
    predictions = {}
    
    for file_id in test_file_ids:
        # Create realistic velocity map (70x70)
        velocity_map = np.random.normal(3000, 500, (70, 70))
        velocity_map = np.clip(velocity_map, 1500, 6000)  # Realistic seismic velocities
        predictions[file_id] = velocity_map
        print(f"  Created prediction for {file_id}: shape {velocity_map.shape}")
    
    print(f"\nGenerated {len(predictions)} predictions")
    
    # Create submission file
    print("\nCreating Kaggle submission...")
    try:
        submission_df = kaggle_gen.create_submission_from_predictions(
            predictions=predictions,
            output_path="jupyter_test_submission.csv"
        )
        
        print("\n=== Submission Statistics ===")
        print(f"Shape: {submission_df.shape}")
        print(f"Columns: {list(submission_df.columns)}")
        print(f"Files: {len(set(row.split('_y_')[0] for row in submission_df['oid_ypos']))}")
        
        # Validate submission
        validation = kaggle_gen._validate_submission(submission_df)
        print(f"\nValidation:")
        print(f"  Valid: {validation['is_valid']}")
        print(f"  MAE Ready: {validation['mae_ready']}")
        if validation['errors']:
            print(f"  Errors: {validation['errors']}")
        if validation['warnings']:
            print(f"  Warnings: {validation['warnings']}")
        
        # Preview first few rows
        print("\n=== First 5 Rows ===")
        print(submission_df.head())
        
        print(f"\n[SUCCESS] Submission saved as 'jupyter_test_submission.csv'")
        print(f"File size: {os.path.getsize('jupyter_test_submission.csv') / 1024:.1f} KB")
        
        return submission_df
        
    except Exception as e:
        print(f"[ERROR] Error creating submission: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_sample_submission_only():
    """Create just a sample submission with default values"""
    
    print("=== Creating Sample Submission ===\n")
    
    try:
        from utils.kaggle_submission import KaggleSubmissionGenerator
        
        kaggle_gen = KaggleSubmissionGenerator()
        test_file_ids = ["000030dca2", "000031dca3", "000032dca4", "000033dca5", "000034dca6"]
        
        # Create sample submission
        sample_df = kaggle_gen.create_sample_submission(
            test_file_ids=test_file_ids,
            output_path="jupyter_sample_submission.csv"
        )
        
        print("[SUCCESS] Sample submission created successfully!")
        print(f"Shape: {sample_df.shape}")
        print("\nFirst 5 rows:")
        print(sample_df.head())
        
        return sample_df
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run the test
    result = create_jupyter_submission_test()
    
    if result is None:
        print("\nTrying sample submission instead...")
        create_sample_submission_only()
