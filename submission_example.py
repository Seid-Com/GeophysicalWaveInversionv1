#!/usr/bin/env python3
"""
Example script for creating Kaggle submissions for FWI competition
This shows how to create submissions outside of the Streamlit app
"""

import numpy as np
import torch
from pathlib import Path
import sys
sys.path.append('.')

from utils.kaggle_submission import KaggleSubmissionGenerator
from utils.large_dataset_utils import LargeDatasetProcessor, optimize_submission_workflow
from models.physics_guided_network import PhysicsGuidedFWI
from utils.preprocessor import SeismicPreprocessor

def create_submission_example():
    """Example of creating a submission file for Kaggle"""
    
    print("=== FWI Kaggle Submission Example ===\n")
    
    # Initialize submission generator
    kaggle_gen = KaggleSubmissionGenerator()
    
    # 1. Load test file IDs - try multiple possible paths
    possible_test_dirs = [
        "sample_data/test",
        "sample_data/sample_data/test", 
        "Downloads/GeophysicalWaveInversion/GeophysicalWaveInversion/sample_data/test",
        "sample_data/sample_data/test"
    ]
    
    test_file_ids = []
    test_dir = None
    
    for dir_path in possible_test_dirs:
        try:
            test_file_ids = kaggle_gen.load_test_file_ids(dir_path)
            if test_file_ids:
                test_dir = dir_path
                break
        except:
            continue
    
    if not test_file_ids:
        print("No test files found. Creating sample test files for demonstration...")
        # Create sample test file IDs for demo
        test_file_ids = ["000030dca2", "000031dca3", "000032dca4", "000033dca5", "000034dca6"]
        test_dir = "sample_data/sample_data/test"  # Use the actual path from directory structure
    
    print(f"Found {len(test_file_ids)} test files: {test_file_ids}\n")
    
    # 2. Initialize model (example - you would load your trained model)
    print("Initializing physics-guided FWI model...")
    model = PhysicsGuidedFWI(
        input_channels=1,
        output_channels=1,
        encoder_channels=[64, 128, 256],
        decoder_channels=[256, 128, 64]
    )
    model.eval()
    
    # 3. Generate predictions for each test file
    print("Generating predictions...")
    predictions = {}
    preprocessor = SeismicPreprocessor()
    
    for i, file_id in enumerate(test_file_ids):
        print(f"Processing {file_id} ({i+1}/{len(test_file_ids)})")
        
        # Try to load test data, create dummy data if file doesn't exist
        test_file = Path(test_dir) / f"{file_id}.npy"
        
        if test_file.exists():
            seismic_data = np.load(test_file)
        else:
            print(f"  Creating dummy seismic data for {file_id}")
            # Create dummy seismic data (typical dimensions for FWI)
            seismic_data = np.random.normal(0, 1, (1, 50, 70))  # (channels, time, receivers)
        
        # Preprocess (if you have a fitted preprocessor)
        # seismic_data = preprocessor.preprocess_seismic_data(seismic_data)
        
        # Convert to tensor and add batch dimension
        if len(seismic_data.shape) == 3:
            seismic_data = seismic_data[np.newaxis, :]
        
        seismic_tensor = torch.FloatTensor(seismic_data)
        
        # Generate prediction
        with torch.no_grad():
            prediction = model(seismic_tensor)
            prediction_np = prediction.cpu().numpy()
        
        # Store prediction
        predictions[file_id] = prediction_np
    
    print(f"\nGenerated predictions for {len(predictions)} files")
    
    # Validate we have predictions
    if not predictions:
        print("ERROR: No predictions generated. Cannot create submission.")
        return None
    
    # 4. Optimize submission workflow based on dataset size
    workflow_config = optimize_submission_workflow(len(test_file_ids))
    print(f"Using {workflow_config['method']} method for {len(test_file_ids)} files")
    
    # 4. Create Kaggle submission file
    print("Creating Kaggle submission file...")
    
    if workflow_config['streaming']:
        # Use streaming approach for very large datasets
        processor = LargeDatasetProcessor()
        result = processor.create_streaming_submission(
            predictions=predictions,
            output_path="my_kaggle_submission.csv"
        )
        print(f"Streaming submission completed: {result}")
        submission_df = None  # Don't load into memory for large datasets
    else:
        # Use standard approach with optimized chunk size
        submission_df = kaggle_gen.create_submission_from_predictions(
            predictions=predictions,
            output_path="my_kaggle_submission.csv",
            chunk_size=workflow_config['chunk_size']
        )
    
    # 5. Validate submission (skip for streaming method)
    if submission_df is not None:
        validation = kaggle_gen._validate_submission(submission_df)
        print(f"\nSubmission validation:")
        print(f"Valid: {validation['is_valid']}")
        if validation['errors']:
            print(f"Errors: {validation['errors']}")
        if validation['warnings']:
            print(f"Warnings: {validation['warnings']}")
        
        # 6. Preview submission
        print("\n=== Submission Preview ===")
        stats = kaggle_gen.preview_submission(submission_df, num_rows=5)
    else:
        print("\nLarge dataset processed using streaming method")
        print("Validation performed during streaming process")
    
    print(f"\nSubmission ready! Upload 'my_kaggle_submission.csv' to:")
    print("https://www.kaggle.com/competitions/waveform-inversion/submissions")
    
    return submission_df

def create_sample_submission_only():
    """Create a sample submission with default values"""
    
    print("=== Creating Sample Submission ===\n")
    
    kaggle_gen = KaggleSubmissionGenerator()
    
    # Get test file IDs
    test_dir = "sample_data/test"
    test_file_ids = kaggle_gen.load_test_file_ids(test_dir)
    
    # Create sample submission with default values
    sample_df = kaggle_gen.create_sample_submission(
        test_file_ids=test_file_ids,
        output_path="sample_submission.csv"
    )
    
    print("Sample submission created: sample_submission.csv")
    print(f"Shape: {sample_df.shape}")
    print(f"Files: {len(test_file_ids)}")
    
    return sample_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create FWI Kaggle submission")
    parser.add_argument("--sample-only", action="store_true", 
                       help="Create sample submission with default values")
    
    args = parser.parse_args()
    
    if args.sample_only:
        create_sample_submission_only()
    else:
        create_submission_example()