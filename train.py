import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from torchvision import transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim 
from tqdm import tqdm
import torch.nn.functional as F
import time
import glob

from utils import count_parameters, train_epoch, validate_model, collate_fn
from network import get_maskrcnn_model

class ShapesDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = os.path.abspath(data_dir)  # Convert to absolute path
        self.transform = transform
        
        # Ensure all required directories exist
        img_dir = os.path.join(self.data_dir, "images")
        mask_dir = os.path.join(self.data_dir, "masks")
        anno_dir = os.path.join(self.data_dir, "annotations")
        
        # Verify directories exist
        for dir_path in [img_dir, mask_dir, anno_dir]:
            if not os.path.exists(dir_path):
                raise ValueError(f"Directory not found: {dir_path}")
        
        # Get all image paths and verify each has corresponding files
        self.image_paths = []
        self.mask_paths_dict = {}
        self.anno_paths = {}
        
        # Get all PNG files in the images directory
        potential_images = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        
        for img_path in potential_images:
            base_name = os.path.basename(img_path).replace(".png", "")
            
            # Check for mask files
            mask_pattern = os.path.join(mask_dir, f"{base_name}_mask_*.png")
            mask_files = sorted(glob.glob(mask_pattern))
            
            # Check for annotation file
            anno_path = os.path.join(anno_dir, f"{base_name}.txt")
            
            # Only add if all required files exist
            if mask_files and os.path.exists(anno_path):
                self.image_paths.append(img_path)
                self.mask_paths_dict[base_name] = mask_files
                self.anno_paths[base_name] = anno_path
            else:
                print(f"Warning: Skipping {base_name} - missing required files:")
                if not mask_files:
                    print(f"  No mask files found matching {mask_pattern}")
                if not os.path.exists(anno_path):
                    print(f"  No annotation file found at {anno_path}")
        
        if not self.image_paths:
            raise ValueError(f"No valid image sets found in {data_dir}")
        
        print(f"Loaded {len(self.image_paths)} valid image sets from {data_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get base name for finding corresponding files
        base_name = os.path.basename(image_path).replace(".png", "")
        
        # Load all mask files for this image
        masks = []
        for mask_path in self.mask_paths_dict[base_name]:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            masks.append(mask)
        
        # Stack masks into a single array
        if masks:
            masks = np.stack(masks, axis=0)
        else:
            masks = np.zeros((0, image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Load annotations (bounding boxes and classes)
        boxes = []
        labels = []
        
        with open(self.anno_paths[base_name], 'r') as f:
            for line in f:
                cls, x1, y1, x2, y2 = map(float, line.strip().split())
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls))
        
        # Convert to tensors
        image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks
        }
        
        if self.transform:
            image = self.transform(image)
        
        return image, target

def main():
    # Define transforms
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    # Create datasets with absolute paths and error handling
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        train_dir = os.path.join(base_dir, 'data', 'shapes', 'train')
        val_dir = os.path.join(base_dir, 'data', 'shapes', 'val')

        print("\nInitializing datasets:")
        print("Training data directory:", train_dir)
        print("Validation data directory:", val_dir)

        train_dataset = ShapesDataset(data_dir=train_dir, transform=transform)
        val_dataset = ShapesDataset(data_dir=val_dir, transform=transform)
        
        print("\nDataset initialization complete:")
        print("Training samples:", len(train_dataset))
        print("Validation samples:", len(val_dataset))

    except Exception as e:
        print("\nError initializing datasets:", str(e))
        raise

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, 
                            batch_size=16, 
                            shuffle=True, 
                            num_workers=4,
                            collate_fn=collate_fn)

    val_loader = DataLoader(val_dataset, 
                          batch_size=16, 
                          shuffle=False, 
                          num_workers=4,
                          collate_fn=collate_fn)

    # Initialize model with number of shape classes + background
    num_classes = 7  # Background + 6 shapes (square, circle, triangle, hexagon, pentagon, star)
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

    # Initialize scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"Epoch [{epoch+1}/{num_epochs}]")

        train_losses = train_epoch(model, train_loader, optimizer, device, scaler)
        val_metrics = validate_model(model, val_loader, device)
        
        # Get the total loss for scheduler
        train_loss = train_losses['total_loss']
        val_iou = val_metrics['mean_iou']
        
        lr_scheduler.step(train_loss)

        elapsed_time = time.time() - start_time
        print(f"Epoch time: {elapsed_time:.2f} seconds")
        
        # Print detailed metrics
        print("\nTraining Losses:")
        for k, v in train_losses.items():
            print(f"  {k}: {v:.4f}")
            
        print("\nValidation Metrics:")
        print(f"  Mean IoU: {val_metrics['mean_iou']:.4f}")
        print(f"  Mean Precision: {val_metrics['mean_precision']:.4f}")
        print(f"  Mean Recall: {val_metrics['mean_recall']:.4f}")
        print(f"  F1 Score: {val_metrics['f1_score']:.4f}")
        print(f"  Detection Rate: {val_metrics['detection_rate']:.4f}")
        print(f"  False Positive Rate: {val_metrics['false_positive_rate']:.4f}")
        
        # Per-class performance
        print("\nPer-class Performance:")
        for class_id in sorted(val_metrics['class_mean_ious'].keys()):
            print(f"  Class {class_id}:")
            print(f"    IoU: {val_metrics['class_mean_ious'][class_id]:.4f}")
            print(f"    Detection Rate: {val_metrics['class_detection_rates'][class_id]:.4f}")
            print(f"    Confidence: {val_metrics['class_mean_confidence'][class_id]:.4f}")

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'train_loss': train_loss,
            'val_iou': val_iou,
            'scaler_state_dict': scaler.state_dict() if scaler else None
        }

        # Save best model
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_checkpoint_path = os.path.join(checkpoint_dir, "best_seg.pth")
            torch.save(checkpoint, best_checkpoint_path)
            print(f"\nNew best model at epoch {epoch+1} with train loss: {train_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"\nNo improvement for {patience_counter} epoch(s).")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Save final model
    final_model_path = os.path.join(saves_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print("Training complete! Final model saved.")

if __name__ == '__main__':
    import platform
    if platform.system() == 'Windows':
        # Windows needs this to handle multiprocessing properly
        from torch.multiprocessing import freeze_support
        freeze_support()
    main()
