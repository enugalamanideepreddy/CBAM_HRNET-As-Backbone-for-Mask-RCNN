import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from PIL import Image
from torchvision import transforms, datasets
import numpy as np
import timm
import torch.nn as nn
import torch.optim as optim 
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import random
import time

from utils import count_parameters, train_epoch, validate_model
from network import get_maskrcnn_model

# Define paths
root = "data/train2017"
annFile = "data/annotations/instances_train2017.json"

# Load full COCO dataset
full_dataset = datasets.CocoDetection(root=root, annFile=annFile)

# Take a small random subset of the dataset for quick testing
subset_indices = random.sample(range(len(full_dataset)), k=500)  # Change k for larger subset
subset_dataset = Subset(full_dataset, subset_indices)

# Split subset into train and validation
train_size = int(0.8 * len(subset_dataset))
val_size = len(subset_dataset) - train_size
train_dataset, val_dataset = random_split(subset_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

# Initialize model
num_classes = 2
model = get_maskrcnn_model(num_classes=num_classes, use_cbam=True)
print(f"Number of parameters in the model: {count_parameters(model)}")

# Device and training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 30
learning_rate = 0.01

# Load model weights if available
model.to(device)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=True
)

saves_dir = "./saves"
os.makedirs(saves_dir, exist_ok=True)

checkpoint_dir = "./saves/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

best_train_loss = np.inf
patience = 8
patience_counter = 0

for epoch in range(num_epochs):
    start_time = time.time()
    print(f"Epoch [{epoch+1}/{num_epochs}]")

    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_iou = validate_model(model, val_loader, device)
    lr_scheduler.step(train_loss)

    elapsed_time = time.time() - start_time
    print(f"Epoch time: {elapsed_time:.2f} seconds")
    print(f"Train Loss: {train_loss:.4f}, Val IoU: {val_iou:.4f}")

    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'train_loss': train_loss,
        'val_iou': val_iou,
    }

    # Save best model
    if train_loss < best_train_loss:
        best_train_loss = train_loss
        best_checkpoint_path = os.path.join(checkpoint_dir, "best_seg.pth")
        torch.save(checkpoint, best_checkpoint_path)
        print(f"New best model at epoch {epoch+1} with train loss: {train_loss:.4f}")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"No improvement for {patience_counter} epoch(s).")

    if patience_counter >= patience:
        print("Early stopping triggered.")
        break

# Save final model
final_model_path = os.path.join(saves_dir, "final_model.pth")
torch.save(model.state_dict(), final_model_path)
print("Training complete! Final model saved.")
