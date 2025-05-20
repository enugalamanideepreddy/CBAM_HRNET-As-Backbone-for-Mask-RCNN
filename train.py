import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from torchvision import transforms
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

from utils import count_parameters, train_epoch, validate_model

from network import get_maskrcnn_model


# Define paths
root = "data/images"
annFile = "data/annotations.json"

# Create dataset
dataset = datasets.CocoDetection(root=root, annFile=annFile)

# Split dataset (e.g., 80% train, 20% validation)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

num_classes = 2
model = get_maskrcnn_model(num_classes=num_classes, use_cbam=True)

print(f"Number of parameters in the model: {count_parameters(model):,}")

# Device and training hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 30
learning_rate = 0.01

# Move model to device
model.load_state_dict(torch.load("./saves/Final_model_cbm.pth",
                                  map_location = 'cpu'))

model.to(device)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2)

# Replace StepLR with ReduceLROnPlateau, monitoring 'max' since higher IoU is better.
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,
    mode='max',        # We want IoU to increase.
    factor=0.5,        # Reduce learning rate by half.
    patience=2,        # Wait for 2 epochs with no improvement.
    min_lr=1e-6,       # Do not go below this learning rate.
    verbose=True
)

saves_dir = "./saves"
os.makedirs(saves_dir, exist_ok=True)

# Directory for checkpointing
checkpoint_dir = "./saves/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
best_val_iou = 0.0
best_train_loss = np.inf
patience = 8  # optional early stopping threshold
patience_counter = 0

for epoch in range(num_epochs):
    print(f"Epoch [{epoch+1}/{num_epochs}]")

    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_iou = validate_model(model, val_loader, device)
    lr_scheduler.step(train_loss)

    print(f"Train Loss: {train_loss:.4f}, Val IoU: {val_iou:.4f}")

    # Save checkpoint
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'train_loss': train_loss,
        'val_iou': val_iou,
    }

    # Save best model based on validation IoU
    if train_loss < best_train_loss:
        best_val_iou = train_loss
        best_checkpoint_path = os.path.join(checkpoint_dir, "best_seg.pth")
        torch.save(checkpoint, best_checkpoint_path)
        print(f"New best model at epoch {epoch+1} with metric: {train_loss:.4f}")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"No improvement for {patience_counter} epoch(s).")

    if patience_counter >= patience:
        print("Early stopping triggered.")
        break

# Save final model state
final_model_path = os.path.join(saves_dir, "final_model.pth")
torch.save(model.state_dict(), final_model_path)
print("Training complete! Final model saved.")