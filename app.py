import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import os
import glob
from datetime import datetime
from models.physics_guided_network import PhysicsGuidedFWI
from utils.data_loader import SeismicDataLoader
from utils.preprocessor import SeismicPreprocessor
from utils.submission_formatter import SubmissionFormatter
from utils.kaggle_submission import KaggleSubmissionGenerator, generate_kaggle_submission_code
from utils.large_dataset_utils import LargeDatasetProcessor, optimize_submission_workflow
from utils.wave_physics import WavePhysicsCalculator, SeismicDataAnalyzer, generate_physics_tutorial_data
from utils.model_utils import (
    count_parameters, get_model_size_mb, create_training_summary, 
    plot_training_curves, validate_model_config, estimate_training_time, 
    get_device_info
)
from utils.database import FWIDatabase
from trainer import FWITrainer

# Page configuration
st.set_page_config(
    page_title="Geophysical Waveform Inversion",
    page_icon="🌍",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'trainer' not in st.session_state:
    st.session_state.trainer = None
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'database' not in st.session_state:
    try:
        st.session_state.database = FWIDatabase()
    except Exception as e:
        st.session_state.database = None
        st.session_state.db_error = str(e)

st.title("🌍 Yale/UNC-CH Geophysical Waveform Inversion")
st.markdown("### Physics-Guided Machine Learning for Subsurface Velocity Map Prediction")

# Add competition info banner
st.info("""
**Competition Overview**: Develop physics-guided ML models to solve full-waveform inversion problems. 
Predict subsurface velocity maps from seismic waveform data with $50,000 in prizes. 
Use the sample data provided to test the complete workflow.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["Data Overview", "Model Configuration", "Training", "Prediction", "Submission", "Experiments"]
)

# Database status indicator
if st.session_state.database:
    st.sidebar.success("Database Connected")
else:
    st.sidebar.error("Database Unavailable")
    if hasattr(st.session_state, 'db_error'):
        st.sidebar.caption(f"Error: {st.session_state.db_error}")

if page == "Data Overview":
    st.header("📊 Data Overview and Analysis")
    
    # Quick start section
    st.subheader("Quick Start")
    if st.button("Load Sample Data", type="primary"):
        st.session_state.sample_data_loaded = True
        st.success("Sample data directory set: sample_data/train_samples")
        st.rerun()
    
    # Data directory input
    data_dir = st.text_input(
        "Training Data Directory", 
        value="sample_data/train_samples",
        help="Path to the directory containing training data"
    )
    
    if st.button("Load Data Overview"):
        if os.path.exists(data_dir):
            try:
                data_loader = SeismicDataLoader(data_dir)
                st.session_state.data_loader = data_loader
                
                # Display data statistics
                stats = data_loader.get_data_statistics()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Files", stats['total_files'])
                with col2:
                    st.metric("Dataset Families", len(stats['families']))
                with col3:
                    st.metric("Estimated Samples", stats['estimated_samples'])
                
                # Dataset family breakdown
                st.subheader("Dataset Families")
                for family, info in stats['families'].items():
                    with st.expander(f"{family} Family"):
                        st.write(f"**Files:** {info['file_count']}")
                        st.write(f"**Pattern:** {info['pattern']}")
                        if info['sample_files']:
                            st.write("**Sample Files:**")
                            for file in info['sample_files'][:5]:
                                st.code(file)
                
                # Save data statistics to database
                if st.session_state.database:
                    try:
                        for family, info in stats['families'].items():
                            st.session_state.database.save_data_statistics(
                                dataset_name=data_dir,
                                family_name=family,
                                file_count=info['file_count'],
                                sample_count=info['file_count'] * 500,  # 500 samples per file
                                stats={'pattern': info['pattern'], 'sample_files': info['sample_files'][:3]}
                            )
                    except Exception as e:
                        st.warning(f"Could not save data statistics: {e}")
                
                # Load a sample for visualization
                st.subheader("Sample Data Visualization")
                sample_data = data_loader.load_sample_data()
                
                if sample_data:
                    seismic_data, velocity_map = sample_data
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Seismic Data Shape:**", seismic_data.shape)
                        # Visualize a single trace
                        trace_idx = st.slider("Source Index", 0, seismic_data.shape[1]-1, 0)
                        receiver_idx = st.slider("Receiver Index", 0, seismic_data.shape[3]-1, 0)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(seismic_data[0, trace_idx, :, receiver_idx])
                        ax.set_xlabel("Time Steps")
                        ax.set_ylabel("Amplitude")
                        ax.set_title(f"Seismic Trace (Source {trace_idx}, Receiver {receiver_idx})")
                        st.pyplot(fig)
                    
                    with col2:
                        st.write("**Velocity Map Shape:**", velocity_map.shape)
                        # Visualize velocity map
                        sample_idx = st.slider("Sample Index", 0, velocity_map.shape[0]-1, 0)
                        
                        fig = px.imshow(
                            velocity_map[sample_idx], 
                            color_continuous_scale='viridis',
                            title=f"Velocity Map (Sample {sample_idx})",
                            labels={'color': 'Velocity (m/s)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
        
        # Wave physics analysis section
        with st.expander("Wave Physics Analysis", expanded=False):
            try:
                st.subheader("Wave Traveltime Calculations")
                st.write("Based on OpenFWI tutorial concepts:")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Direct Wave:** t_direct = d/c")
                    offset = st.slider("Source-Receiver Offset (m)", 0, 700, 300)
                    velocity = st.slider("P-wave Velocity (m/s)", 1500, 6000, 3000)
                    
                    calculator = WavePhysicsCalculator()
                    direct_time = calculator.calculate_direct_wave_traveltime(offset, velocity)
                    st.metric("Direct Wave Traveltime", f"{direct_time:.3f} s")
                
                with col2:
                    st.write("**Reflection Wave:** t_refl = √(d² + 4h²)/c")
                    depth = st.slider("Reflector Depth (m)", 10, 700, 200)
                    
                    refl_time = calculator.calculate_reflection_traveltime(offset, depth, velocity)
                    st.metric("Reflection Traveltime", f"{refl_time:.3f} s")
                
                # Physics tutorial data generation
                if st.button("Generate Physics Tutorial Data"):
                    with st.spinner("Generating layered velocity model and synthetic seismograms..."):
                        velocity_tutorial, seismic_tutorial = generate_physics_tutorial_data()
                        
                        # Analyze the generated data
                        analyzer = SeismicDataAnalyzer()
                        seismic_stats = analyzer.analyze_seismic_data(seismic_tutorial)
                        velocity_stats = calculator.analyze_velocity_quality(velocity_tutorial)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Generated Velocity Model")
                            fig_vel = plt.figure(figsize=(8, 6))
                            plt.imshow(velocity_tutorial, aspect='auto', cmap='jet', 
                                     extent=[0, 700, 700, 0])
                            plt.xlabel('Horizontal Distance (m)')
                            plt.ylabel('Depth (m)')
                            plt.title('Layered Velocity Model')
                            plt.colorbar(label='Velocity (m/s)')
                            st.pyplot(fig_vel)
                            plt.close()
                            
                            st.write("**Quality Analysis:**")
                            st.write(f"• Velocity range: {velocity_stats['min_velocity']:.0f} - {velocity_stats['max_velocity']:.0f} m/s")
                            st.write(f"• Physical constraints: {'✓' if velocity_stats['within_physical_range'] else '✗'}")
                            st.write(f"• Velocity gradient: {'✓' if velocity_stats['reasonable_gradient'] else '✗'}")
                            st.write(f"• Smoothness score: {velocity_stats['smoothness_score']:.2f}")
                        
                        with col2:
                            st.subheader("Synthetic Seismic Data")
                            fig_seis = plt.figure(figsize=(8, 6))
                            plt.imshow(seismic_tutorial[0], aspect='auto', cmap='gray',
                                     extent=[0, 700, 1, 0], vmin=-0.5, vmax=0.5)
                            plt.xlabel('Receiver Position (m)')
                            plt.ylabel('Time (s)')
                            plt.title('Shot Gather (Source 1)')
                            plt.colorbar(label='Amplitude')
                            st.pyplot(fig_seis)
                            plt.close()
                            
                            st.write("**Data Analysis:**")
                            st.write(f"• Shape: {seismic_stats['shape']}")
                            st.write(f"• Sources: {seismic_stats['num_sources']}")
                            st.write(f"• Duration: {seismic_stats['time_duration']:.1f} s")
                            st.write(f"• Aperture: {seismic_stats['geophone_aperture']} m")
                            st.write(f"• Dominant freq: {seismic_stats['dominant_frequency']:.1f} Hz")
                    
            except Exception as e:
                st.error(f"Error in wave physics analysis: {str(e)}")
    else:
        st.error(f"Directory {data_dir} does not exist")

elif page == "Model Configuration":
    st.header("🧠 Model Configuration")
    
    # Quick configuration presets
    st.subheader("Configuration Presets")
    preset = st.selectbox(
        "Choose a preset configuration",
        ["Custom", "Fast Training", "High Accuracy", "Balanced"]
    )
    
    if preset == "Fast Training":
        st.info("Optimized for quick training with reduced model complexity")
        default_encoder = "32,64,128"
        default_decoder = "128,64,32"
        default_epochs = 20
        default_batch = 16
    elif preset == "High Accuracy":
        st.info("Optimized for best performance with larger model")
        default_encoder = "64,128,256,512"
        default_decoder = "512,256,128,64"
        default_epochs = 100
        default_batch = 4
    elif preset == "Balanced":
        st.info("Balanced configuration for good performance and training speed")
        default_encoder = "64,128,256"
        default_decoder = "256,128,64"
        default_epochs = 50
        default_batch = 8
    else:
        default_encoder = "64,128,256,512"
        default_decoder = "512,256,128,64"
        default_epochs = 100
        default_batch = 8
    
    # Model hyperparameters
    st.subheader("Network Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Input dimensions (will be auto-detected from data)
        st.write("**Input Configuration**")
        input_channels = st.number_input("Input Channels", value=1, min_value=1, max_value=10)
        
        st.write("**Encoder Configuration**")
        encoder_channels = st.text_input(
            "Encoder Channels", 
            value=default_encoder if preset != "Custom" else "64,128,256,512",
            help="Comma-separated list of channel sizes"
        )
        
    with col2:
        st.write("**Decoder Configuration**")
        decoder_channels = st.text_input(
            "Decoder Channels", 
            value=default_decoder if preset != "Custom" else "512,256,128,64",
            help="Comma-separated list of channel sizes"
        )
        
        output_channels = st.number_input("Output Channels", value=1, min_value=1, max_value=5)
    
    # Physics-guided parameters
    st.subheader("Physics-Guided Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        physics_weight = st.slider("Physics Loss Weight", 0.0, 1.0, 0.1, 0.01)
        wave_equation_weight = st.slider("Wave Equation Weight", 0.0, 1.0, 0.05, 0.01)
        
    with col2:
        boundary_condition_weight = st.slider("Boundary Condition Weight", 0.0, 1.0, 0.03, 0.01)
        smoothness_weight = st.slider("Smoothness Regularization", 0.0, 1.0, 0.02, 0.01)
    
    # Training parameters
    st.subheader("Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        learning_rate = st.number_input("Learning Rate", value=0.001, format="%.6f")
        batch_size = st.selectbox("Batch Size", [4, 8, 16, 32], index=[4, 8, 16, 32].index(default_batch) if preset != "Custom" else 1)
        
    with col2:
        num_epochs = st.number_input("Number of Epochs", value=default_epochs if preset != "Custom" else 100, min_value=1, max_value=1000)
        validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2, 0.05)
        
    # Additional model options
    st.subheader("Advanced Options")
    
    with st.expander("Device and Performance Settings"):
        device_option = st.selectbox("Device", ["auto", "cpu", "cuda"])
        use_mixed_precision = st.checkbox("Use Mixed Precision Training", value=False)
        gradient_clip_norm = st.number_input("Gradient Clipping Norm", value=1.0, min_value=0.1)
    
    # Model validation
    config = {
        'encoder_channels': encoder_channels,
        'decoder_channels': decoder_channels,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs
    }
    
    errors, warnings = validate_model_config(config)
    
    if errors:
        st.error("Configuration Errors:")
        for error in errors:
            st.error(f"• {error}")
    
    if warnings:
        st.warning("Configuration Warnings:")
        for warning in warnings:
            st.warning(f"• {warning}")
    
    # Device information
    device_info = get_device_info()
    st.info(f"**Compute Device**: {device_info['recommended_device'].upper()} {'(CUDA available)' if device_info['cuda_available'] else '(CPU only)'}")
    
    if st.button("Initialize Model"):
        try:
            # Parse channel configurations
            encoder_ch = [int(x.strip()) for x in encoder_channels.split(',')]
            decoder_ch = [int(x.strip()) for x in decoder_channels.split(',')]
            
            # Initialize model
            model = PhysicsGuidedFWI(
                input_channels=input_channels,
                output_channels=output_channels,
                encoder_channels=encoder_ch,
                decoder_channels=decoder_ch,
                physics_weight=physics_weight,
                wave_equation_weight=wave_equation_weight,
                boundary_condition_weight=boundary_condition_weight,
                smoothness_weight=smoothness_weight
            )
            
            # Initialize preprocessor
            preprocessor = SeismicPreprocessor()
            
            # Initialize trainer with device selection
            device = device_option if device_option != "auto" else device_info['recommended_device']
            trainer = FWITrainer(
                model=model,
                learning_rate=learning_rate,
                batch_size=batch_size,
                num_epochs=num_epochs,
                validation_split=validation_split,
                device=device
            )
            
            # Store in session state
            st.session_state.model = model
            st.session_state.preprocessor = preprocessor
            st.session_state.trainer = trainer
            
            # Create experiment in database
            if st.session_state.database:
                try:
                    experiment_config = {
                        'encoder_channels': encoder_channels,
                        'decoder_channels': decoder_channels,
                        'input_channels': input_channels,
                        'output_channels': output_channels,
                        'physics_weight': physics_weight,
                        'wave_equation_weight': wave_equation_weight,
                        'boundary_condition_weight': boundary_condition_weight,
                        'smoothness_weight': smoothness_weight,
                        'learning_rate': learning_rate,
                        'batch_size': batch_size,
                        'num_epochs': num_epochs,
                        'validation_split': validation_split,
                        'device': device,
                        'preset': preset
                    }
                    
                    experiment_name = f"FWI_{preset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    experiment_id = st.session_state.database.create_experiment(
                        name=experiment_name,
                        config=experiment_config,
                        description=f"Physics-guided FWI model with {preset} configuration"
                    )
                    st.session_state.current_experiment_id = experiment_id
                    st.info(f"Experiment created: {experiment_name} (ID: {experiment_id})")
                except Exception as e:
                    st.warning(f"Could not create experiment in database: {e}")
            
            st.success("✅ Model initialized successfully!")
            
            # Display detailed model summary
            param_info = count_parameters(model)
            model_size = get_model_size_mb(model)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Parameters", f"{param_info['total']:,}")
            with col2:
                st.metric("Trainable Parameters", f"{param_info['trainable']:,}")
            with col3:
                st.metric("Model Size", f"{model_size:.1f} MB")
            
            # Training time estimate
            estimated_time = estimate_training_time(config, 1000)  # Assume 1000 samples
            st.info(f"**Estimated Training Time**: {estimated_time} (approximate)")
            
        except Exception as e:
            st.error(f"Error initializing model: {str(e)}")

elif page == "Training":
    st.header("🎯 Model Training")
    
    if st.session_state.model is None:
        st.warning("⚠️ Please configure and initialize the model first.")
    elif st.session_state.data_loader is None:
        st.warning("⚠️ Please load data overview first.")
    else:
        # Training data selection
        st.subheader("Training Data Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_families = st.multiselect(
                "Select Dataset Families",
                ["Vel", "Fault", "Style"],
                default=["Vel"]
            )
            
            max_samples_per_family = st.number_input(
                "Max Samples per Family", 
                value=1000, 
                min_value=100, 
                max_value=10000
            )
        
        with col2:
            # Additional training options
            use_data_augmentation = st.checkbox("Use Data Augmentation", value=True)
            save_checkpoints = st.checkbox("Save Training Checkpoints", value=True)
        
        # Training controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Start Training", type="primary"):
                st.session_state.training_active = True
        
        with col2:
            if st.button("Stop Training"):
                st.session_state.training_active = False
        
        with col3:
            if st.button("Reset Training"):
                st.session_state.training_active = False
                if st.session_state.trainer:
                    st.session_state.trainer.reset()
        
        # Training progress and metrics
        if hasattr(st.session_state, 'training_active') and st.session_state.training_active:
            st.subheader("Training Progress")
            
            # Create placeholders for dynamic updates
            progress_bar = st.progress(0)
            metrics_placeholder = st.empty()
            loss_chart_placeholder = st.empty()
            
            try:
                # Update experiment status
                if st.session_state.database and hasattr(st.session_state, 'current_experiment_id'):
                    st.session_state.database.update_experiment_status(
                        st.session_state.current_experiment_id, 'training'
                    )
                
                # Start training with database logging
                def training_callback(epoch, total, metrics):
                    update_training_ui(epoch, total, metrics, progress_bar, metrics_placeholder, loss_chart_placeholder)
                    
                    # Log metrics to database
                    if st.session_state.database and hasattr(st.session_state, 'current_experiment_id'):
                        try:
                            st.session_state.database.log_training_metrics(
                                st.session_state.current_experiment_id, metrics
                            )
                        except Exception as e:
                            st.warning(f"Could not log metrics: {e}")
                
                history = st.session_state.trainer.train(
                    data_loader=st.session_state.data_loader,
                    selected_families=selected_families,
                    max_samples_per_family=max_samples_per_family,
                    preprocessor=st.session_state.preprocessor,
                    use_augmentation=use_data_augmentation,
                    progress_callback=training_callback
                )
                
                # Update experiment status to completed
                if st.session_state.database and hasattr(st.session_state, 'current_experiment_id'):
                    st.session_state.database.update_experiment_status(
                        st.session_state.current_experiment_id, 'completed'
                    )
                
                st.success("🎉 Training completed successfully!")
                
                # Display comprehensive training summary
                if history:
                    st.subheader("Training Summary")
                    summary = create_training_summary(history)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Final Train Loss", f"{summary['final_train_loss']:.6f}")
                    with col2:
                        st.metric("Final Val Loss", f"{summary['final_val_loss']:.6f}")
                    with col3:
                        st.metric("Best Val Loss", f"{summary['best_val_loss']:.6f}")
                    with col4:
                        st.metric("Best Epoch", summary['best_epoch'])
                    
                    # Additional metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Final Train MAE", f"{summary['final_train_mae']:.6f}")
                    with col2:
                        st.metric("Final Val MAE", f"{summary['final_val_mae']:.6f}")
                    with col3:
                        st.metric("Total Time", f"{summary['total_training_time']:.1f}s")
                    with col4:
                        st.metric("Final LR", f"{summary['final_learning_rate']:.6f}")
                    
                    # Training curves
                    st.subheader("Training Curves")
                    training_fig = plot_training_curves(history)
                    if training_fig:
                        st.plotly_chart(training_fig, use_container_width=True)
                    
                    # Store history for later analysis
                    st.session_state.training_history = history
                
            except Exception as e:
                # Update experiment status to failed
                if st.session_state.database and hasattr(st.session_state, 'current_experiment_id'):
                    st.session_state.database.update_experiment_status(
                        st.session_state.current_experiment_id, 'failed'
                    )
                st.error(f"Training error: {str(e)}")
            finally:
                st.session_state.training_active = False

elif page == "Prediction":
    st.header("🔮 Velocity Map Prediction")
    
    if st.session_state.model is None:
        st.warning("⚠️ Please configure and initialize the model first.")
    else:
        # Test data input
        st.subheader("Test Data Configuration")
        
        test_data_dir = st.text_input(
            "Test Data Directory", 
            value="sample_data/test",
            help="Path to the directory containing test .npy files"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Model checkpoint selection
            checkpoint_path = st.text_input(
                "Model Checkpoint Path (optional)",
                help="Path to saved model checkpoint"
            )
        
        with col2:
            # Prediction batch size
            pred_batch_size = st.selectbox("Prediction Batch Size", [1, 4, 8, 16], index=2)
        
        # Load model checkpoint if provided
        if checkpoint_path and st.button("Load Checkpoint"):
            try:
                if os.path.exists(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    st.session_state.model.load_state_dict(checkpoint['model_state_dict'])
                    st.success("✅ Model checkpoint loaded successfully!")
                else:
                    st.error("Checkpoint file not found")
            except Exception as e:
                st.error(f"Error loading checkpoint: {str(e)}")
        
        # Generate predictions
        if st.button("Generate Predictions", type="primary"):
            if os.path.exists(test_data_dir):
                try:
                    st.subheader("Generating Predictions...")
                    
                    # Get test files
                    test_files = glob.glob(os.path.join(test_data_dir, "*.npy"))
                    
                    if not test_files:
                        st.error("No .npy files found in test directory")
                        st.stop()
                    
                    st.info(f"Found {len(test_files)} test files")
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    predictions = {}
                    
                    # Set model to evaluation mode
                    st.session_state.model.eval()
                    
                    with torch.no_grad():
                        for i, test_file in enumerate(test_files):
                            status_text.text(f"Processing {os.path.basename(test_file)}")
                            
                            # Load test data
                            test_data = np.load(test_file)
                            
                            # Preprocess if preprocessor is available
                            if st.session_state.preprocessor:
                                test_data = st.session_state.preprocessor.preprocess_seismic_data(test_data)
                            
                            # Convert to tensor
                            if len(test_data.shape) == 3:
                                test_data = test_data[np.newaxis, :]  # Add batch dimension
                            
                            test_tensor = torch.FloatTensor(test_data)
                            
                            # Generate prediction
                            pred_tensor = st.session_state.model(test_tensor)
                            pred_numpy = pred_tensor.cpu().numpy()
                            
                            # Store prediction
                            file_id = os.path.splitext(os.path.basename(test_file))[0]
                            predictions[file_id] = pred_numpy
                            
                            # Update progress
                            progress_bar.progress((i + 1) / len(test_files))
                    
                    # Store predictions in session state
                    st.session_state.predictions = predictions
                    
                    st.success(f"✅ Generated predictions for {len(predictions)} files!")
                    
                    # Display sample prediction
                    if predictions:
                        st.subheader("Sample Prediction Visualization")
                        
                        sample_file = st.selectbox("Select file to visualize", list(predictions.keys()))
                        sample_pred = predictions[sample_file]
                        
                        # Visualize first sample in batch
                        fig = px.imshow(
                            sample_pred[0, 0] if len(sample_pred.shape) == 4 else sample_pred[0],
                            color_continuous_scale='viridis',
                            title=f"Predicted Velocity Map - {sample_file}",
                            labels={'color': 'Velocity (m/s)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error generating predictions: {str(e)}")
            else:
                st.error(f"Test directory {test_data_dir} does not exist")

elif page == "Submission":
    st.header("📄 Kaggle Competition Submission")
    
    # Competition instructions
    st.info("""
    **Kaggle Submission Instructions:**
    1. Generate predictions for test files
    2. Format predictions according to competition requirements (odd columns only)
    3. Download the CSV file
    4. Upload to: https://www.kaggle.com/competitions/waveform-inversion/submissions
    """)
    
    # Show test data information
    st.subheader("Test Data Information")
    test_dir = "sample_data/test"
    
    if os.path.exists(test_dir):
        kaggle_gen = KaggleSubmissionGenerator()
        test_file_ids = kaggle_gen.load_test_file_ids(test_dir)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Test Files Found", len(test_file_ids))
        with col2:
            st.metric("Expected Rows", len(test_file_ids) * 70)
        with col3:
            st.metric("Output Columns", 36)  # 35 odd columns + oid_ypos
        
        if test_file_ids:
            st.write("**Test File IDs:**")
            st.write(", ".join(test_file_ids[:10]) + ("..." if len(test_file_ids) > 10 else ""))
    
    if not hasattr(st.session_state, 'predictions') or not st.session_state.predictions:
        st.warning("⚠️ Please generate predictions first on the Prediction page.")
        
        # Show example submission code
        st.subheader("Example Submission Code")
        with st.expander("View example code for creating submissions"):
            st.code(generate_kaggle_submission_code(), language='python')
    else:
        st.subheader("Generate Kaggle Submission")
        
        # Submission configuration
        col1, col2 = st.columns(2)
        
        with col1:
            submission_filename = st.text_input(
                "Submission Filename",
                value="kaggle_submission.csv"
            )
        
        with col2:
            st.write("**Competition Format:**")
            st.write("- Height: 70 rows per file")
            st.write("- Width: 35 odd columns (x_1, x_3, ..., x_69)")
            st.write("- Format: oid_ypos + velocity values")
        
        # Generate Kaggle submission file
        if st.button("Generate Kaggle Submission", type="primary"):
            try:
                kaggle_gen = KaggleSubmissionGenerator()
                
                # Analyze dataset size and optimize workflow
                num_predictions = len(st.session_state.predictions)
                workflow_config = optimize_submission_workflow(num_predictions)
                
                st.info(f"Processing {num_predictions} predictions using {workflow_config['method']} method")
                
                # Create submission with optimized settings
                if workflow_config['streaming'] and num_predictions > 1000:
                    # Use streaming for very large datasets
                    processor = LargeDatasetProcessor()
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def progress_callback(current, total):
                        progress = current / total
                        progress_bar.progress(progress)
                        status_text.text(f"Processing batch {current}/{total}")
                    
                    result = processor.create_streaming_submission(
                        predictions=st.session_state.predictions,
                        output_path=submission_filename,
                        progress_callback=progress_callback
                    )
                    
                    st.success(f"Streaming submission completed: {result['file_size_mb']:.1f} MB")
                    submission_df = None  # Don't load large datasets into memory
                    
                else:
                    # Use standard method with chunking
                    submission_df = kaggle_gen.create_submission_from_predictions(
                        predictions=st.session_state.predictions,
                        output_path=submission_filename,
                        chunk_size=workflow_config['chunk_size']
                    )
                
                st.success(f"✅ Kaggle submission saved as {submission_filename}")
                
                # Display submission statistics
                if submission_df is not None:
                    stats = kaggle_gen.preview_submission(submission_df, num_rows=5)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Rows", stats['total_rows'])
                    with col2:
                        st.metric("Total Columns", stats['total_columns'])
                    with col3:
                        st.metric("Test Files", stats['unique_files'])
                    with col4:
                        st.metric("Value Range", f"{stats['value_range'][0]:.0f}-{stats['value_range'][1]:.0f}")
                    
                    # Display format verification
                    st.subheader("Competition Format Verification")
                    validation = kaggle_gen._validate_submission(submission_df)
                    
                    if validation['is_valid']:
                        st.success("✅ Submission format is valid for Kaggle upload")
                    else:
                        st.error("❌ Submission format issues:")
                        for error in validation['errors']:
                            st.error(f"• {error}")
                    
                    if validation['warnings']:
                        st.warning("⚠️ Warnings:")
                        for warning in validation['warnings']:
                            st.warning(f"• {warning}")
                    
                    # Preview sample rows
                    st.subheader("Submission Preview (First 10 Rows)")
                    st.dataframe(submission_df.head(10), use_container_width=True)
                
                else:
                    # Large dataset processed with streaming
                    file_size = os.path.getsize(submission_filename) / (1024 * 1024)
                    total_rows = num_predictions * 70
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Rows", f"{total_rows:,}")
                    with col2:
                        st.metric("Total Columns", 36)
                    with col3:
                        st.metric("Test Files", num_predictions)
                    with col4:
                        st.metric("File Size", f"{file_size:.1f} MB")
                    
                    st.success("✅ Large dataset processed successfully using streaming method")
                    st.info("Format validation performed during streaming process")
                
                # Competition submission instructions
                st.subheader("Upload to Kaggle")
                st.markdown("""
                **Next Steps:**
                1. Download the submission file below
                2. Go to: https://www.kaggle.com/competitions/waveform-inversion/submissions
                3. Click "Submit Predictions"
                4. Upload your CSV file
                5. Add a description of your model/approach
                6. Submit!
                """)
                
                # Download button
                with open(submission_filename, 'rb') as f:
                    st.download_button(
                        label="📥 Download for Kaggle Upload",
                        data=f.read(),
                        file_name=submission_filename,
                        mime='text/csv',
                        help="Download this file and upload it to the Kaggle competition"
                    )
                
                # Expected format details
                with st.expander("Competition Format Details"):
                    st.write("**Required Format:**")
                    st.write("- Column 1: oid_ypos (format: fileID_y_rowNumber)")
                    st.write("- Columns 2-36: x_1, x_3, x_5, ..., x_69 (odd columns only)")
                    st.write("- 70 rows per test file (y_0 to y_69)")
                    st.write("- Values: velocity predictions in m/s")
                    
                    st.write("**Example row:**")
                    st.code("000030dca2_y_0,3000.0,3100.0,2950.0,...")
                
            except Exception as e:
                st.error(f"Error generating Kaggle submission: {str(e)}")
                st.write("Full error details:")
                st.exception(e)
        
        # Alternative: Create sample submission
        st.subheader("Create Sample Submission")
        st.write("Generate a sample submission with default values for testing:")
        
        if st.button("Generate Sample Submission"):
            try:
                kaggle_gen = KaggleSubmissionGenerator()
                test_file_ids = kaggle_gen.load_test_file_ids(test_dir)
                
                # Show dataset size analysis
                processor = LargeDatasetProcessor()
                estimates = processor.estimate_memory_usage(len(test_file_ids))
                
                st.write("**Dataset Analysis:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Files", len(test_file_ids))
                with col2:
                    st.metric("Estimated Rows", f"{estimates['total_rows']:,}")
                with col3:
                    st.metric("Est. Size", f"{estimates['estimated_csv_size_mb']:.1f} MB")
                
                # Generate sample submission with progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Generating sample submission...")
                sample_df = kaggle_gen.create_sample_submission(
                    test_file_ids=test_file_ids,
                    output_path="sample_submission.csv",
                    chunk_size=estimates['recommended_chunk_size']
                )
                
                progress_bar.progress(1.0)
                status_text.text("Sample submission complete!")
                
                st.success("Sample submission created: sample_submission.csv")
                st.dataframe(sample_df.head(10))
                
                with open("sample_submission.csv", 'rb') as f:
                    st.download_button(
                        label="Download Sample Submission",
                        data=f.read(),
                        file_name="sample_submission.csv",
                        mime='text/csv'
                    )
            except Exception as e:
                st.error(f"Error creating sample submission: {str(e)}")

elif page == "Experiments":
    st.header("🧪 Experiment Management")
    
    if not st.session_state.database:
        st.error("Database connection required for experiment management")
        st.stop()
    
    # Experiment overview
    st.subheader("Recent Experiments")
    
    try:
        experiments_df = st.session_state.database.get_experiments(limit=20)
        
        if not experiments_df.empty:
            # Display experiments table
            st.dataframe(
                experiments_df[['id', 'name', 'status', 'batch_size', 'learning_rate', 'created_at']],
                use_container_width=True
            )
            
            # Experiment details
            st.subheader("Experiment Details")
            selected_exp_id = st.selectbox(
                "Select experiment to view details",
                experiments_df['id'].tolist(),
                format_func=lambda x: f"ID {x}: {experiments_df[experiments_df['id']==x]['name'].iloc[0]}"
            )
            
            if selected_exp_id:
                # Get experiment summary
                summary = st.session_state.database.get_experiment_summary(selected_exp_id)
                
                # Display experiment info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Status", summary['experiment'].get('status', 'unknown'))
                with col2:
                    st.metric("Total Epochs", summary['metrics'].get('total_epochs', 0))
                with col3:
                    best_loss = summary['metrics'].get('best_val_loss')
                    st.metric("Best Val Loss", f"{best_loss:.6f}" if best_loss else "N/A")
                
                # Training history chart
                if summary['metrics'].get('total_epochs', 0) > 0:
                    st.subheader("Training History")
                    history_df = st.session_state.database.get_training_history(selected_exp_id)
                    
                    if not history_df.empty:
                        # Create loss chart
                        import plotly.graph_objects as go
                        from plotly.subplots import make_subplots
                        
                        fig = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=('Training Loss', 'Mean Absolute Error'),
                            vertical_spacing=0.08
                        )
                        
                        fig.add_trace(
                            go.Scatter(x=history_df['epoch'], y=history_df['train_loss'], 
                                     name='Train Loss', line=dict(color='blue')),
                            row=1, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=history_df['epoch'], y=history_df['val_loss'], 
                                     name='Val Loss', line=dict(color='red')),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(x=history_df['epoch'], y=history_df['train_mae'], 
                                     name='Train MAE', line=dict(color='lightblue')),
                            row=2, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=history_df['epoch'], y=history_df['val_mae'], 
                                     name='Val MAE', line=dict(color='lightcoral')),
                            row=2, col=1
                        )
                        
                        fig.update_layout(height=600, showlegend=True)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show metrics table
                        st.subheader("Training Metrics")
                        st.dataframe(history_df, use_container_width=True)
                
                # Export experiment data
                if st.button("Export Experiment Data"):
                    export_data = st.session_state.database.export_experiment_data(selected_exp_id)
                    st.json(export_data)
        
        else:
            st.info("No experiments found. Create and train a model to see experiments here.")
        
        # Best experiments leaderboard
        st.subheader("Best Performing Experiments")
        best_experiments = st.session_state.database.get_best_experiments(limit=10)
        
        if not best_experiments.empty:
            st.dataframe(
                best_experiments[['id', 'name', 'best_val_loss', 'total_epochs', 'created_at']],
                use_container_width=True
            )
        
        # Data overview
        st.subheader("Dataset Overview")
        data_overview = st.session_state.database.get_data_overview()
        
        if not data_overview.empty:
            st.dataframe(data_overview, use_container_width=True)
        
        # Database management
        st.subheader("Database Management")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clean Old Failed Experiments"):
                deleted_count = st.session_state.database.cleanup_old_data(days=7)
                st.success(f"Deleted {deleted_count} old failed experiments")
        
        with col2:
            if st.button("View Database Schema"):
                st.code("""
                Tables:
                - experiments: Main experiment records
                - training_metrics: Per-epoch training metrics
                - model_checkpoints: Saved model information
                - predictions: Prediction results
                - data_statistics: Dataset statistics
                """)
    
    except Exception as e:
        st.error(f"Database error: {str(e)}")

# Helper function for training UI updates
def update_training_ui(epoch, total_epochs, metrics, progress_bar, metrics_placeholder, loss_chart_placeholder):
    """Update training UI components"""
    # Update progress bar
    progress = epoch / total_epochs
    progress_bar.progress(progress)
    
    # Update metrics display
    with metrics_placeholder.container():
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Epoch", f"{epoch}/{total_epochs}")
        with col2:
            st.metric("Train Loss", f"{metrics.get('train_loss', 0):.6f}")
        with col3:
            st.metric("Val Loss", f"{metrics.get('val_loss', 0):.6f}")
        with col4:
            st.metric("Val MAE", f"{metrics.get('val_mae', 0):.6f}")

# Footer
st.markdown("---")
st.markdown("### 🌍 Yale/UNC-CH Geophysical Waveform Inversion Competition")
st.markdown("Physics-guided machine learning for subsurface imaging")
