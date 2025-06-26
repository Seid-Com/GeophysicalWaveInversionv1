import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PhysicsGuidedFWI(nn.Module):
    """
    Physics-Guided Full Waveform Inversion Network
    
    This network combines data-driven learning with physics-based constraints
    to predict subsurface velocity maps from seismic waveform data.
    """
    
    def __init__(
        self,
        input_channels=1,
        output_channels=1,
        encoder_channels=[64, 128, 256, 512],
        decoder_channels=[512, 256, 128, 64],
        physics_weight=0.1,
        wave_equation_weight=0.05,
        boundary_condition_weight=0.03,
        smoothness_weight=0.02
    ):
        super(PhysicsGuidedFWI, self).__init__()
        
        self.physics_weight = physics_weight
        self.wave_equation_weight = wave_equation_weight
        self.boundary_condition_weight = boundary_condition_weight
        self.smoothness_weight = smoothness_weight
        
        # Seismic data encoder (processes 4D input)
        self.seismic_encoder = self._build_seismic_encoder(input_channels, encoder_channels)
        
        # Spatial feature processor
        self.spatial_processor = self._build_spatial_processor(encoder_channels[-1])
        
        # Velocity map decoder
        self.velocity_decoder = self._build_velocity_decoder(encoder_channels[-1], decoder_channels, output_channels)
        
        # Physics-informed layers
        self.physics_constraint = PhysicsConstraintLayer()
        
    def _build_seismic_encoder(self, input_channels, encoder_channels):
        """Build encoder for seismic waveform data"""
        layers = []
        in_channels = input_channels
        
        for out_channels in encoder_channels:
            layers.extend([
                # 3D convolution for temporal-spatial processing
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2, stride=2)
            ])
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _build_spatial_processor(self, channels):
        """Build spatial feature processor"""
        return nn.Sequential(
            nn.AdaptiveAvgPool3d((1, None, None)),  # Pool temporal dimension
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def _build_velocity_decoder(self, input_channels, decoder_channels, output_channels):
        """Build decoder for velocity map prediction"""
        layers = []
        in_channels = input_channels
        
        for out_channels in decoder_channels:
            layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
            in_channels = out_channels
        
        # Final output layer
        layers.append(nn.Conv2d(in_channels, output_channels, kernel_size=1))
        
        return nn.Sequential(*layers)
    
    def forward(self, seismic_data):
        """
        Forward pass
        
        Args:
            seismic_data: Tensor of shape (batch, sources, time, receivers)
                         or (batch, channels, sources, time, receivers)
        
        Returns:
            velocity_map: Tensor of shape (batch, channels, height, width)
        """
        # Handle different input shapes
        if len(seismic_data.shape) == 4:
            # Add channel dimension
            seismic_data = seismic_data.unsqueeze(1)
        
        # Encode seismic features
        encoded_features = self.seismic_encoder(seismic_data)
        
        # Process spatial features
        # Remove temporal dimension and convert to 2D
        spatial_features = encoded_features.squeeze(2)  # Remove time dimension
        processed_features = self.spatial_processor(spatial_features)
        
        # Decode to velocity map
        velocity_map = self.velocity_decoder(processed_features)
        
        # Apply physics constraints
        velocity_map = self.physics_constraint(velocity_map)
        
        return velocity_map
    
    def compute_physics_loss(self, velocity_map, seismic_data=None):
        """
        Compute physics-based loss terms
        
        Args:
            velocity_map: Predicted velocity map
            seismic_data: Input seismic data (optional)
        
        Returns:
            physics_loss: Combined physics loss
        """
        device = velocity_map.device
        
        # Wave equation constraint
        wave_loss = self._wave_equation_loss(velocity_map)
        
        # Boundary condition constraint
        boundary_loss = self._boundary_condition_loss(velocity_map)
        
        # Smoothness constraint
        smoothness_loss = self._smoothness_loss(velocity_map)
        
        # Combine losses
        physics_loss = (
            self.wave_equation_weight * wave_loss +
            self.boundary_condition_weight * boundary_loss +
            self.smoothness_weight * smoothness_loss
        )
        
        return physics_loss
    
    def _wave_equation_loss(self, velocity_map):
        """Approximate wave equation constraint"""
        # Compute spatial derivatives
        dx = torch.gradient(velocity_map, dim=-1)[0]
        dy = torch.gradient(velocity_map, dim=-2)[0]
        
        # Second derivatives
        dxx = torch.gradient(dx, dim=-1)[0]
        dyy = torch.gradient(dy, dim=-2)[0]
        
        # Wave equation approximation: ∇²v should be smooth
        laplacian = dxx + dyy
        
        return torch.mean(laplacian ** 2)
    
    def _boundary_condition_loss(self, velocity_map):
        """Enforce smooth boundary conditions"""
        # Penalize large gradients at boundaries
        top_grad = torch.mean((velocity_map[:, :, 0, :] - velocity_map[:, :, 1, :]) ** 2)
        bottom_grad = torch.mean((velocity_map[:, :, -1, :] - velocity_map[:, :, -2, :]) ** 2)
        left_grad = torch.mean((velocity_map[:, :, :, 0] - velocity_map[:, :, :, 1]) ** 2)
        right_grad = torch.mean((velocity_map[:, :, :, -1] - velocity_map[:, :, :, -2]) ** 2)
        
        return top_grad + bottom_grad + left_grad + right_grad
    
    def _smoothness_loss(self, velocity_map):
        """Enforce spatial smoothness"""
        # Total variation loss
        dx = torch.abs(velocity_map[:, :, :, 1:] - velocity_map[:, :, :, :-1])
        dy = torch.abs(velocity_map[:, :, 1:, :] - velocity_map[:, :, :-1, :])
        
        return torch.mean(dx) + torch.mean(dy)


class PhysicsConstraintLayer(nn.Module):
    """Apply physics constraints to velocity predictions"""
    
    def __init__(self, min_velocity=1500.0, max_velocity=6000.0):
        super(PhysicsConstraintLayer, self).__init__()
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
    
    def forward(self, velocity_map):
        """
        Apply velocity constraints
        
        Args:
            velocity_map: Raw velocity predictions
        
        Returns:
            constrained_velocity: Physically constrained velocities
        """
        # Apply sigmoid activation and scale to velocity range
        normalized = torch.sigmoid(velocity_map)
        constrained_velocity = (
            normalized * (self.max_velocity - self.min_velocity) + self.min_velocity
        )
        
        return constrained_velocity


class FWILoss(nn.Module):
    """Combined loss function for Full Waveform Inversion"""
    
    def __init__(self, data_weight=1.0, physics_weight=0.1):
        super(FWILoss, self).__init__()
        self.data_weight = data_weight
        self.physics_weight = physics_weight
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
    
    def forward(self, predicted_velocity, target_velocity, model, seismic_data=None):
        """
        Compute combined loss
        
        Args:
            predicted_velocity: Model predictions
            target_velocity: Ground truth velocity maps
            model: Physics-guided model for computing physics loss
            seismic_data: Input seismic data
        
        Returns:
            total_loss: Combined data and physics loss
            loss_components: Dictionary of individual loss components
        """
        # Data-driven loss
        data_loss = self.mae_loss(predicted_velocity, target_velocity)
        
        # Physics-guided loss
        physics_loss = model.compute_physics_loss(predicted_velocity, seismic_data)
        
        # Combined loss
        total_loss = self.data_weight * data_loss + self.physics_weight * physics_loss
        
        loss_components = {
            'total_loss': total_loss.item(),
            'data_loss': data_loss.item(),
            'physics_loss': physics_loss.item()
        }
        
        return total_loss, loss_components
