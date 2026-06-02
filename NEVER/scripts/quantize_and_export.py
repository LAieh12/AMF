import torch
import numpy as np
import argparse
import struct

def quantize_ternary(tensor, threshold=0.1):
    """
    Quantizes a float tensor to {-1, 0, 1} based on a threshold.
    """
    quantized = torch.zeros_like(tensor, dtype=torch.int8)
    quantized[tensor > threshold] = 1
    quantized[tensor < -threshold] = -1
    return quantized

def export_weights_to_bin(model_path, output_bin_path, threshold=0.1):
    print(f"Loading weights from {model_path}...")
    
    try:
        state_dict = torch.load(model_path, map_location='cpu')
    except Exception as e:
        print(f"Could not load model: {e}")
        # Generate dummy weights for demonstration if file doesn't exist
        print("Generating dummy weights for demonstration purposes...")
        state_dict = {
            'layer1.weight': torch.randn(256, 2),
            'layer1.bias': torch.randn(256),
            'layer2.weight': torch.randn(4, 256),
            'layer2.bias': torch.randn(4)
        }
        
    print(f"Quantizing to Ternary {{-1, 0, 1}} with threshold: {threshold}")
    
    total_params = 0
    quantized_params = 0
    
    with open(output_bin_path, 'wb') as f:
        # Write a simple header: Magic number + number of layers
        f.write(struct.pack('<I', 0x4E455652)) # 'NEVR'
        f.write(struct.pack('<I', len(state_dict)))
        
        for name, tensor in state_dict.items():
            print(f"Processing layer: {name} | Shape: {tensor.shape}")
            
            # Write layer name length and name
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)
            
            # Write tensor shape
            f.write(struct.pack('<I', len(tensor.shape)))
            for dim in tensor.shape:
                f.write(struct.pack('<I', dim))
                
            # Flatten tensor for quantization
            flat_tensor = tensor.flatten()
            total_params += flat_tensor.numel()
            
            if 'weight' in name: # Only quantize weights, keep biases as float32
                q_tensor = quantize_ternary(flat_tensor, threshold)
                quantized_params += q_tensor.abs().sum().item()
                # Write as int8
                f.write(q_tensor.numpy().astype(np.int8).tobytes())
            else:
                # Write bias as float32
                f.write(flat_tensor.numpy().astype(np.float32).tobytes())
                
    sparsity = 100.0 * (1.0 - (quantized_params / total_params)) if total_params > 0 else 0
    print(f"\nExport complete: {output_bin_path}")
    print(f"Total Parameters: {total_params}")
    print(f"Sparsity achieved: {sparsity:.2f}% (zeros)")
    print("These weights are now ready to be mmap'd or loaded directly into VRAM by N.E.V.E.R.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ternary Quantizer & Exporter for N.E.V.E.R.")
    parser.add_argument("--model", default="tokyo_static_inr_weights.pth", help="Path to PyTorch model")
    parser.add_argument("--output", default="weights.neverbin", help="Output binary file")
    parser.add_argument("--threshold", type=float, default=0.1, help="Threshold for ternary quantization")
    
    args = parser.parse_args()
    
    print("N.E.V.E.R. Ternary Weight Exporter (Phase 6) - Boss 1")
    export_weights_to_bin(args.model, args.output, args.threshold)
