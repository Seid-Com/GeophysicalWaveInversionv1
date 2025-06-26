"""
Utility functions for model operations and training
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path

def count_parameters(model):
    """Count total and trainable parameters in model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }

def get_model_size_mb(model):
    """Estimate model size in megabytes"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

def create_training_summary(history):
    """Create training summary statistics"""
    if not history:
        return {}
    
    final_epoch = history[-1]
    best_val_loss = min(epoch['val_loss'] for epoch in history)
    best_epoch = next(i for i, epoch in enumerate(history) if epoch['val_loss'] == best_val_loss) + 1
    
    return {
        'total_epochs': len(history),
        'final_train_loss': final_epoch.get('train_loss', 0),
        'final_val_loss': final_epoch.get('val_loss', 0),
        'final_train_mae': final_epoch.get('train_mae', 0),
        'final_val_mae': final_epoch.get('val_mae', 0),
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'total_training_time': sum(epoch.get('epoch_time', 0) for epoch in history),
        'final_learning_rate': final_epoch.get('lr', 0)
    }

def plot_training_curves(history):
    """Create plotly figure of training curves"""
    if not history:
        return None
    
    epochs = [h['epoch'] for h in history]
    train_losses = [h.get('train_loss', 0) for h in history]
    val_losses = [h.get('val_loss', 0) for h in history]
    train_maes = [h.get('train_mae', 0) for h in history]
    val_maes = [h.get('val_mae', 0) for h in history]
    
    # Create subplots
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Loss', 'Mean Absolute Error'),
        vertical_spacing=0.08
    )
    
    # Loss curves
    fig.add_trace(
        go.Scatter(x=epochs, y=train_losses, name='Train Loss', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=val_losses, name='Validation Loss', line=dict(color='red')),
        row=1, col=1
    )
    
    # MAE curves
    fig.add_trace(
        go.Scatter(x=epochs, y=train_maes, name='Train MAE', line=dict(color='lightblue')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=val_maes, name='Validation MAE', line=dict(color='lightcoral')),
        row=2, col=1
    )
    
    fig.update_layout(
        title='Training Progress',
        height=600,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="MAE", row=2, col=1)
    
    return fig

def validate_model_config(config):
    """Validate model configuration parameters"""
    errors = []
    warnings = []
    
    # Check encoder/decoder channels
    try:
        encoder_ch = [int(x.strip()) for x in config.get('encoder_channels', '').split(',')]
        decoder_ch = [int(x.strip()) for x in config.get('decoder_channels', '').split(',')]
        
        if not encoder_ch or not decoder_ch:
            errors.append("Encoder and decoder channels cannot be empty")
        
        if encoder_ch[-1] != decoder_ch[0]:
            warnings.append("Last encoder channel should match first decoder channel")
            
    except ValueError:
        errors.append("Invalid channel configuration - use comma-separated integers")
    
    # Check batch size vs memory
    batch_size = config.get('batch_size', 8)
    if batch_size > 32:
        warnings.append("Large batch size may cause memory issues")
    
    # Check learning rate
    lr = config.get('learning_rate', 0.001)
    if lr > 0.01:
        warnings.append("High learning rate may cause unstable training")
    elif lr < 1e-6:
        warnings.append("Very low learning rate may slow training significantly")
    
    return errors, warnings

def estimate_training_time(config, dataset_size):
    """Rough estimate of training time"""
    batch_size = config.get('batch_size', 8)
    num_epochs = config.get('num_epochs', 100)
    
    # Rough estimates based on model complexity
    encoder_channels = config.get('encoder_channels', '64,128,256,512')
    complexity = len(encoder_channels.split(','))
    
    # Approximate time per sample (in seconds)
    time_per_sample = 0.01 * complexity  # Base time
    
    batches_per_epoch = max(1, dataset_size // batch_size)
    total_batches = batches_per_epoch * num_epochs
    
    estimated_seconds = total_batches * time_per_sample * batch_size
    
    # Convert to human readable
    if estimated_seconds < 60:
        return f"{estimated_seconds:.0f} seconds"
    elif estimated_seconds < 3600:
        return f"{estimated_seconds/60:.1f} minutes"
    else:
        return f"{estimated_seconds/3600:.1f} hours"

def save_model_config(config, filepath="model_config.json"):
    """Save model configuration to file"""
    import json
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)

def load_model_config(filepath="model_config.json"):
    """Load model configuration from file"""
    import json
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def get_device_info():
    """Get information about available compute devices"""
    info = {
        'cpu_available': True,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'recommended_device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    return info