"""
Utilities for handling large datasets efficiently
Memory-optimized processing for million+ row submissions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Iterator, Tuple
import os
import gc
from pathlib import Path

class LargeDatasetProcessor:
    """Handle large datasets with memory optimization"""
    
    def __init__(self, chunk_size: int = 50000):
        self.chunk_size = chunk_size
    
    def process_predictions_in_batches(
        self, 
        predictions: Dict[str, np.ndarray], 
        batch_size: int = 100
    ) -> Iterator[List[Tuple[str, np.ndarray]]]:
        """Process predictions in batches to manage memory"""
        prediction_items = list(predictions.items())
        
        for i in range(0, len(prediction_items), batch_size):
            batch = prediction_items[i:i + batch_size]
            yield batch
            # Force garbage collection after each batch
            gc.collect()
    
    def estimate_memory_usage(self, num_files: int, rows_per_file: int = 70, cols: int = 36) -> Dict:
        """Estimate memory requirements for submission generation"""
        total_rows = num_files * rows_per_file
        
        # Estimate DataFrame memory usage
        # Each float64 value takes 8 bytes, string takes ~50 bytes average
        memory_per_row = (cols - 1) * 8 + 50  # velocity values + oid_ypos string
        total_memory_mb = (total_rows * memory_per_row) / (1024 * 1024)
        
        # CSV file size estimate (text format is larger)
        csv_size_mb = total_memory_mb * 2.5  # Text format expansion factor
        
        return {
            'total_rows': total_rows,
            'estimated_memory_mb': total_memory_mb,
            'estimated_csv_size_mb': csv_size_mb,
            'recommended_chunk_size': min(50000, max(1000, int(100000000 / memory_per_row)))
        }
    
    def create_streaming_submission(
        self, 
        predictions: Dict[str, np.ndarray], 
        output_path: str,
        progress_callback=None
    ) -> Dict:
        """Create submission using streaming approach for very large datasets"""
        
        total_files = len(predictions)
        estimates = self.estimate_memory_usage(total_files)
        
        print(f"Processing {total_files:,} files -> {estimates['total_rows']:,} rows")
        print(f"Estimated memory: {estimates['estimated_memory_mb']:.1f} MB")
        print(f"Estimated file size: {estimates['estimated_csv_size_mb']:.1f} MB")
        
        # Write header
        columns = ['oid_ypos'] + [f'x_{i}' for i in range(1, 70, 2)]
        
        with open(output_path, 'w') as f:
            # Write header
            f.write(','.join(columns) + '\n')
            
            rows_written = 0
            batch_size = 100 if estimates['estimated_memory_mb'] > 1000 else 500
            
            for batch_idx, batch in enumerate(self.process_predictions_in_batches(predictions, batch_size)):
                batch_rows = []
                
                for file_id, prediction in batch:
                    # Process prediction
                    processed_pred = self._process_prediction_optimized(prediction)
                    
                    # Create rows for this file
                    for y_pos in range(70):
                        row_id = f"{file_id}_y_{y_pos}"
                        
                        # Extract odd columns
                        values = [str(processed_pred[y_pos, x_pos]) for x_pos in range(1, 70, 2)]
                        row_line = f"{row_id}," + ','.join(values) + '\n'
                        batch_rows.append(row_line)
                        rows_written += 1
                
                # Write batch to file
                f.writelines(batch_rows)
                
                if progress_callback:
                    progress_callback(batch_idx + 1, len(list(predictions.keys())) // batch_size + 1)
                
                # Progress update
                if (batch_idx + 1) % 10 == 0:
                    print(f"Written {rows_written:,} rows...")
                
                # Clear memory
                del batch_rows
                gc.collect()
        
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Submission complete: {output_path}")
        print(f"Final size: {file_size_mb:.1f} MB")
        
        return {
            'output_path': output_path,
            'total_rows': rows_written,
            'file_size_mb': file_size_mb,
            'total_files': total_files
        }
    
    def _process_prediction_optimized(self, prediction: np.ndarray) -> np.ndarray:
        """Memory-optimized prediction processing"""
        # Handle different input shapes efficiently
        if len(prediction.shape) == 4:
            pred_2d = prediction[0, 0]
        elif len(prediction.shape) == 3:
            pred_2d = prediction[0]
        elif len(prediction.shape) == 2:
            pred_2d = prediction
        else:
            raise ValueError(f"Unexpected prediction shape: {prediction.shape}")
        
        # Resize if necessary (in-place when possible)
        if pred_2d.shape != (70, 70):
            pred_2d = self._resize_optimized(pred_2d, 70, 70)
        
        # Apply constraints in-place
        np.clip(pred_2d, 1500.0, 6000.0, out=pred_2d)
        
        # Handle NaN/inf values
        mask = ~np.isfinite(pred_2d)
        pred_2d[mask] = 3000.0
        
        return pred_2d
    
    def _resize_optimized(self, prediction: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Optimized resize for large arrays"""
        h_old, w_old = prediction.shape
        
        if h_old == target_h and w_old == target_w:
            return prediction
        
        # Use linear interpolation indices
        h_indices = np.linspace(0, h_old - 1, target_h).astype(int)
        w_indices = np.linspace(0, w_old - 1, target_w).astype(int)
        
        # Index-based resize (faster than scipy for large arrays)
        return prediction[np.ix_(h_indices, w_indices)]

def optimize_submission_workflow(num_files: int) -> Dict:
    """Recommend optimal settings based on dataset size"""
    processor = LargeDatasetProcessor()
    estimates = processor.estimate_memory_usage(num_files)
    
    if estimates['total_rows'] < 100000:
        return {
            'method': 'standard',
            'chunk_size': 10000,
            'batch_size': 1000,
            'use_compression': False,
            'streaming': False
        }
    elif estimates['total_rows'] < 1000000:
        return {
            'method': 'chunked',
            'chunk_size': 50000,
            'batch_size': 500,
            'use_compression': True,
            'streaming': False
        }
    else:
        return {
            'method': 'streaming',
            'chunk_size': 100000,
            'batch_size': 100,
            'use_compression': True,
            'streaming': True
        }