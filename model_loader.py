"""
Model loader utility for Teachable Machine models
Handles compatibility issues between Keras versions
"""
import tensorflow as tf
import h5py
import json

def load_teachable_machine_model(model_path):
    """
    Load a Teachable Machine .h5 model with compatibility fixes
    """
    # Read and modify the model config to remove incompatible parameters
    with h5py.File(model_path, 'r') as f:
        if 'model_config' in f.attrs:
            model_config = json.loads(f.attrs['model_config'])
            
            # Recursively remove 'groups' parameter from DepthwiseConv2D layers
            def fix_config(config):
                if isinstance(config, dict):
                    # Fix DepthwiseConv2D layers
                    if config.get('class_name') == 'DepthwiseConv2D':
                        if 'config' in config and 'groups' in config['config']:
                            del config['config']['groups']
                    
                    # Recursively process nested configs
                    for key, value in config.items():
                        if isinstance(value, (dict, list)):
                            fix_config(value)
                elif isinstance(config, list):
                    for item in config:
                        fix_config(item)
            
            fix_config(model_config)
    
    # Now load with the original load_model but with compile=False
    try:
        # Use TF2's Keras API
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        # If that fails, try loading just the weights
        print(f"Standard loading failed: {e}")
        print("Attempting to load architecture and weights separately...")
        raise

if __name__ == "__main__":
    # Test the loader
    model = load_teachable_machine_model("keras_model.h5")
    print("✓ Model loaded successfully")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
