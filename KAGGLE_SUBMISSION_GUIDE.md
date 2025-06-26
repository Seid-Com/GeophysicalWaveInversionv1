# Kaggle Submission Guide for FWI Competition

## Competition URL
https://www.kaggle.com/competitions/waveform-inversion/submissions

## Quick Start - Using the Streamlit App

### Method 1: Complete Workflow in App
1. **Navigate to "Data Overview"** - Load sample data
2. **Go to "Model Configuration"** - Initialize your physics-guided model
3. **Visit "Training"** - Train the model on seismic data
4. **Open "Prediction"** - Generate predictions for test files
5. **Access "Submission"** - Create and download Kaggle-ready CSV

### Method 2: Generate Sample Submission
1. Go to the "Submission" page
2. Click "Generate Sample Submission" 
3. Download the CSV file
4. Upload to Kaggle for baseline score

## Submission Format Requirements

### File Structure
```
oid_ypos,x_1,x_3,x_5,x_7,...,x_69
000030dca2_y_0,3000.0,3100.0,2950.0,...
000030dca2_y_1,3000.0,3100.0,2950.0,...
...
000030dca2_y_69,3000.0,3100.0,2950.0,...
000031dca3_y_0,3000.0,3100.0,2950.0,...
```

### Key Requirements
- **Columns**: 36 total (1 ID column + 35 velocity columns)
- **ID Format**: `{file_id}_y_{row_number}` (e.g., "000030dca2_y_0")
- **Velocity Columns**: Only odd positions (x_1, x_3, x_5, ..., x_69)
- **Rows per File**: Exactly 70 rows (y_0 to y_69)
- **Values**: Velocity predictions in m/s (typically 1500-6000)

## Command Line Usage

### Create Sample Submission
```bash
!python jupyter_submission_test.py --sample-only
```

### Generate Real Predictions
```bash
!python jupyter_submission_test.py
```

## Large Dataset Handling

### Memory-Optimized Processing
The system automatically detects dataset size and optimizes processing:

- **Small datasets** (<1,000 files): Standard processing
- **Medium datasets** (1,000-10,000 files): Chunked processing with compression
- **Large datasets** (>10,000 files): Streaming processing with minimal memory usage

### Dataset Size Analysis
```python
from utils.large_dataset_utils import LargeDatasetProcessor

processor = LargeDatasetProcessor()
estimates = processor.estimate_memory_usage(num_files=50000)

print(f"Total rows: {estimates['total_rows']:,}")
print(f"Memory needed: {estimates['estimated_memory_mb']:.1f} MB")
print(f"File size: {estimates['estimated_csv_size_mb']:.1f} MB")
```

## Manual Code Example

```python
from utils.kaggle_submission import KaggleSubmissionGenerator
import numpy as np

# Initialize
kaggle_gen = KaggleSubmissionGenerator()

# Get test files
test_files = kaggle_gen.load_test_file_ids("sample_data/test")

# Create predictions (replace with your model)
predictions = {}
for file_id in test_files:
    # Your model prediction here (70x70 velocity map)
    velocity_map = PhysicsGuidedFWI.predict(test_data)
    predictions[file_id] = velocity_map

# Generate submission
submission_df = kaggle_gen.create_submission_from_predictions(
    predictions=predictions,
    output_path="my_submission.csv"
)

# Upload my_submission.csv to Kaggle
```

## Validation Checklist

Before uploading to Kaggle, verify:
- [ ] File has 36 columns (oid_ypos + 35 velocity columns)
- [ ] Column names match: oid_ypos, x_1, x_3, x_5, ..., x_69
- [ ] Each test file has exactly 70 rows (y_0 to y_69)
- [ ] No missing values (NaN or empty cells)
- [ ] Velocity values in reasonable range (1500-6000 m/s)
- [ ] File size is reasonable (should be a few MB)

## Upload Steps

1. **Go to Competition**: https://www.kaggle.com/competitions/waveform-inversion/submissions
2. **Click "Submit Predictions"**
3. **Upload CSV file**
4. **Add Description**: Describe your model/approach
5. **Submit**: Wait for scoring results

## Model Description Examples

### Basic Submission
"Physics-guided neural network with encoder-decoder architecture. Uses wave equation constraints and boundary conditions for realistic velocity predictions."

### Advanced Submission
"Custom physics-informed neural network combining:
- 3D CNN encoder for seismic waveform processing
- Physics constraints (wave equation, smoothness, boundaries)
- Data augmentation with noise injection and temporal shifts
- Trained on Vel/Fault/Style dataset families
- Preprocessing: bandpass filtering + z-score normalization"

## Expected Performance

### Sample Submission Baseline
- Uses constant 3000 m/s velocity
- Provides baseline score to compare against

### Trained Model
- Physics-guided models typically achieve better scores
- Training for more epochs generally improves performance
- Data preprocessing significantly impacts results

## Troubleshooting

### Common Issues
1. **Wrong Format**: Check column names and row counts
2. **Missing Files**: Ensure all test files have predictions
3. **Value Range**: Keep velocities between 1500-6000 m/s
4. **File Size**: Large files may indicate format issues

### Validation in App
The Streamlit app automatically validates submissions and shows:
- Format correctness
- Missing values
- Value ranges
- Row/column counts

## Competition Tips

1. **Start Simple**: Upload sample submission first
2. **Iterate Quickly**: Try different model configurations
3. **Track Experiments**: Use the Experiments page to compare models
4. **Physics Matters**: Models with physics constraints perform better
5. **Data Quality**: Good preprocessing improves results significantly