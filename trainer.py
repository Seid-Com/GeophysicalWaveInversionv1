import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
import time
from pathlib import Path

from models.physics_guided_network import FWILoss
from utils.data_loader import SeismicDataLoader
from utils.preprocessor import SeismicPreprocessor

class SeismicDataset(Dataset):
    """PyTorch Dataset for seismic data"""
    
    def __init__(
        self, 
        file_pairs: List[Tuple[Path, Path]], 
        preprocessor: SeismicPreprocessor,
        use_augmentation: bool = False,
        max_samples_per_file: int = None
    ):
        self.file_pairs = file_pairs
        self.preprocessor = preprocessor
        self.use_augmentation = use_augmentation
        self.max_samples_per_file = max_samples_per_file
        
        # Load all data into memory (for small datasets)
        self.seismic_data = []
        self.velocity_data = []
        self._load_all_data()
    
    def _load_all_data(self):
        """Load all data from files"""
        for data_file, model_file in self.file_pairs:
            try:
                seismic = np.load(data_file)
                velocity = np.load(model_file)
                
                # Limit samples per file if specified
                if self.max_samples_per_file:
                    n_samples = min(self.max_samples_per_file, seismic.shape[0])
                    seismic = seismic[:n_samples]
                    velocity = velocity[:n_samples]
                
                # Preprocess data
                seismic = self.preprocessor.preprocess_seismic_data(seismic)
                velocity = self.preprocessor.preprocess_velocity_data(velocity)
                
                # Add to dataset
                for i in range(seismic.shape[0]):
                    self.seismic_data.append(seismic[i])
                    self.velocity_data.append(velocity[i])
                    
            except Exception as e:
                print(f"Error loading {data_file}: {e}")
                continue
    
    def __len__(self):
        return len(self.seismic_data)
    
    def __getitem__(self, idx):
        seismic = self.seismic_data[idx]
        velocity = self.velocity_data[idx]
        
        # Apply augmentation if enabled
        if self.use_augmentation:
            seismic, velocity = self.preprocessor.augment_seismic_data(seismic, velocity)
        
        # Convert to tensors
        seismic_tensor = torch.FloatTensor(seismic)
        velocity_tensor = torch.FloatTensor(velocity)
        
        return seismic_tensor, velocity_tensor


class FWITrainer:
    """Trainer for Physics-Guided FWI Model"""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        batch_size: int = 8,
        num_epochs: int = 100,
        validation_split: float = 0.2,
        device: str = None
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.validation_split = validation_split
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        self.criterion = FWILoss(data_weight=1.0, physics_weight=self.model.physics_weight)
        
        # Training history
        self.history = []
        self.best_val_loss = float('inf')
        
    def train(
        self,
        data_loader: SeismicDataLoader,
        selected_families: List[str],
        max_samples_per_family: int,
        preprocessor: SeismicPreprocessor,
        use_augmentation: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> List[Dict]:
        """
        Train the model
        
        Args:
            data_loader: SeismicDataLoader instance
            selected_families: List of dataset families to use
            max_samples_per_family: Maximum samples per family
            preprocessor: Data preprocessor
            use_augmentation: Whether to use data augmentation
            progress_callback: Callback function for progress updates
        
        Returns:
            Training history
        """
        print(f"Training on device: {self.device}")
        
        # Create train/validation split
        train_pairs, val_pairs = data_loader.create_train_val_split(
            selected_families, 
            max_samples_per_family, 
            self.validation_split
        )
        
        print(f"Training files: {len(train_pairs)}, Validation files: {len(val_pairs)}")
        
        # Fit preprocessor on training data
        self._fit_preprocessor(train_pairs, preprocessor)
        
        # Create datasets
        train_dataset = SeismicDataset(
            train_pairs, 
            preprocessor, 
            use_augmentation=use_augmentation,
            max_samples_per_file=100  # Limit for faster training
        )
        val_dataset = SeismicDataset(
            val_pairs, 
            preprocessor, 
            use_augmentation=False,
            max_samples_per_file=100
        )
        
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0  # Set to 0 for compatibility
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        # Training loop
        for epoch in range(self.num_epochs):
            start_time = time.time()
            
            # Train phase
            train_metrics = self._train_epoch(train_loader)
            
            # Validation phase
            val_metrics = self._validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_metrics['val_loss'])
            
            # Record metrics
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_mae': train_metrics['mae'],
                'val_loss': val_metrics['val_loss'],
                'val_mae': val_metrics['val_mae'],
                'lr': self.optimizer.param_groups[0]['lr'],
                'epoch_time': time.time() - start_time
            }
            
            self.history.append(epoch_metrics)
            
            # Save best model
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self._save_checkpoint(epoch + 1, is_best=True)
            
            # Progress callback
            if progress_callback:
                progress_callback(epoch + 1, self.num_epochs, epoch_metrics)
            
            # Print progress
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.6f}, Train MAE: {train_metrics['mae']:.6f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.6f}, Val MAE: {val_metrics['val_mae']:.6f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}, Time: {epoch_metrics['epoch_time']:.2f}s")
            print("-" * 50)
        
        return self.history
    
    def _fit_preprocessor(self, train_pairs: List[Tuple[Path, Path]], preprocessor: SeismicPreprocessor):
        """Fit preprocessor on a sample of training data"""
        print("Fitting preprocessor...")
        
        # Load a sample of data for fitting
        seismic_samples = []
        velocity_samples = []
        
        for data_file, model_file in train_pairs[:5]:  # Use first 5 files
            try:
                seismic = np.load(data_file)
                velocity = np.load(model_file)
                
                # Take first 10 samples from each file
                seismic_samples.append(seismic[:10])
                velocity_samples.append(velocity[:10])
                
            except Exception as e:
                print(f"Error loading {data_file}: {e}")
                continue
        
        if seismic_samples and velocity_samples:
            seismic_array = np.concatenate(seismic_samples, axis=0)
            velocity_array = np.concatenate(velocity_samples, axis=0)
            preprocessor.fit(seismic_array, velocity_array)
            print("Preprocessor fitted successfully")
        else:
            print("Warning: Could not fit preprocessor")
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        for seismic_batch, velocity_batch in train_loader:
            seismic_batch = seismic_batch.to(self.device)
            velocity_batch = velocity_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predicted_velocity = self.model(seismic_batch)
            
            # Compute loss
            loss, loss_components = self.criterion(
                predicted_velocity, 
                velocity_batch, 
                self.model, 
                seismic_batch
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Compute MAE
            with torch.no_grad():
                mae = torch.mean(torch.abs(predicted_velocity - velocity_batch))
            
            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'mae': total_mae / num_batches
        }
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for seismic_batch, velocity_batch in val_loader:
                seismic_batch = seismic_batch.to(self.device)
                velocity_batch = velocity_batch.to(self.device)
                
                # Forward pass
                predicted_velocity = self.model(seismic_batch)
                
                # Compute loss
                loss, _ = self.criterion(
                    predicted_velocity, 
                    velocity_batch, 
                    self.model, 
                    seismic_batch
                )
                
                # Compute MAE
                mae = torch.mean(torch.abs(predicted_velocity - velocity_batch))
                
                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'val_mae': total_mae / num_batches
        }
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        # Save regular checkpoint
        checkpoint_path = f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Best model saved at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.history = checkpoint.get('history', [])
        
        print(f"Checkpoint loaded from {checkpoint_path}")
    
    def reset(self):
        """Reset trainer state"""
        self.history = []
        self.best_val_loss = float('inf')
        
        # Reset model parameters
        for layer in self.model.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        print("Trainer reset successfully")
