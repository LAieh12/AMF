import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import argparse

# "Surgical Overfitting" of the Static Branch
# We train a small MLP to memorize a single high-res image of Tokyo.
# This prevents consuming VRAM during inference, as it generates the StaticINRCache.

class StaticINR(nn.Module):
    def __init__(self, hidden_dim=256, layers=4):
        super().__init__()
        
        # Mapping (x, y) -> (R, G, B, Depth)
        net = [nn.Linear(2, hidden_dim), nn.ReLU()]
        for _ in range(layers - 2):
            net.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        net.append(nn.Linear(hidden_dim, 4))
        net.append(nn.Sigmoid()) # Output bounds [0, 1]
        
        self.net = nn.Sequential(*net)
        
    def forward(self, x):
        return self.net(x)

def get_image_coords_and_colors(image_path):
    # Load image and convert to RGB
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w, _ = img.shape
    
    # Create normalized coordinates [-1, 1]
    y_coords = torch.linspace(-1, 1, steps=h)
    x_coords = torch.linspace(-1, 1, steps=w)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    coords = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
    
    # Add dummy depth channel (0 for now)
    colors = torch.tensor(img, dtype=torch.float32).reshape(-1, 3) / 255.0
    depth = torch.zeros((colors.shape[0], 1), dtype=torch.float32)
    targets = torch.cat([colors, depth], dim=1)
    
    return coords, targets, (w, h)

def bake_static_inr(image_path, epochs=1000):
    print(f"Preparing to bake {image_path} into INR...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (Requirement: 6GB VRAM limit)")
    
    try:
        coords, targets, (w, h) = get_image_coords_and_colors(image_path)
    except Exception as e:
        print(e)
        return
        
    coords = coords.to(device)
    targets = targets.to(device)
    
    model = StaticINR().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    print(f"Starting surgical overfitting for {epochs} epochs...")
    batch_size = 4096 * 4 # Process in chunks to respect 6GB VRAM
    
    for epoch in range(epochs):
        # Random permutation for batching
        permutation = torch.randperm(coords.size()[0])
        
        epoch_loss = 0.0
        for i in range(0, coords.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_coords, batch_targets = coords[indices], targets[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_coords)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if epoch % 50 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {epoch_loss:.6f}")
            
    print("Training complete. Exporting weights for C++ StaticINRCache...")
    
    # Save the model state
    torch.save(model.state_dict(), "tokyo_static_inr_weights.pth")
    print("Saved to tokyo_static_inr_weights.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bake Static Background for N.E.V.E.R.")
    parser.add_argument("--image", default="tokyo_highres.jpg", help="High-res background image")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    
    args = parser.parse_args()
    print("N.E.V.E.R. Static Branch Overfitter (Surgical Overfitting) - Step 02")
    # bake_static_inr(args.image, args.epochs)
    print("Run with a valid input image. Example: python train_static_inr.py --image tokyo-3.jpg")
