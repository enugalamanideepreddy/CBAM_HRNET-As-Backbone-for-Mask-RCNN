import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from torchvision import transforms
import numpy as np
import random
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
import numpy as np

import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.patches as patches
from prettytable import PrettyTable

import os
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

class MultiSubjectClassificationSegmentationDataset(Dataset):
    """
    A PyTorch dataset for multi-task learning (classification + segmentation) across multiple subjects.

    Each subject inside MainFolder is expected to have the following structure:

        SubjectFolder/
            train/
                good/         <- Good images (classification label 0)
                anomoly/      <- Anomalous images (classification label 1)
            test/
                good/
                anomoly/
            ground truth segmentation map for anomolies/  <- Ground truth masks for anomalous images

    For "good" images, the mask is generated as a blank mask.
    For "anomoly" images, the dataset looks for a mask with a matching filename in the segmentation folder.

    Args:
        root (str): Path to MainFolder containing subject folders (e.g., "path/to/MainFolder").
        mode (str): Either 'train' or 'test'. Specifies which split to use.
        img_transform (callable, optional): Transformation to apply to input images.
        mask_transform (callable, optional): Transformation to apply to segmentation masks.
    """
    def __init__(self, root, img_transform=None, mask_transform=None):
        self.root = root
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.samples = []  # List of tuples: (image_path, label, mask_path)
        self.targets = []

        # Iterate over each subject folder in MainFolder.
        for subject in os.listdir(root):
            subject_path = os.path.join(root, subject)
            if not os.path.isdir(subject_path):
                continue  # Skip if not a directory

            for mode in ['train','test']:
              mode_dir = os.path.join(subject_path, mode)
              if not os.path.isdir(mode_dir):
                  continue  # If the mode folder doesn't exist, skip

              # Subfolders for images.
              good_dir = os.path.join(mode_dir, "good")
              anomaly_dir = os.path.join(mode_dir, "defect")
              # The segmentation maps folder is common for both splits.
              mask_dir = os.path.join(subject_path, "ground_truth/defect")

              # Process "good" images (label 0; no mask available).
              if os.path.isdir(good_dir):
                  for fname in os.listdir(good_dir):
                      if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                          img_path = os.path.join(good_dir, fname)
                          # For good images, mask is not available.
                          self.samples.append((img_path, 0, None))
                          self.targets.append(0)

              if os.path.isdir(anomaly_dir):
                    for fname in os.listdir(anomaly_dir):
                        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(anomaly_dir, fname)
                            mask_path = None
                            if os.path.isdir(mask_dir):
                                # Modify filename to include _mask before the extension.
                                base, ext = os.path.splitext(fname)
                                mask_fname = base + "_mask" + ext
                                candidate = os.path.join(mask_dir, mask_fname)
                                # print(candidate)
                                if os.path.exists(candidate):
                                    mask_path = candidate
                            self.samples.append((img_path, 1, mask_path))
                            self.targets.append(1)

        self.targets = torch.Tensor(self.targets).long()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, mask_path = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.img_transform:
            image = self.img_transform(image)
        else:
            image = transforms.ToTensor()(image)

        # Load or generate mask
        if label == 1 and mask_path is not None and os.path.exists(mask_path):
            # Load mask for anomalous images
            mask = Image.open(mask_path).convert("L")  # grayscale
            # Define the threshold value
            threshold = 0
            mask_array = np.array(mask)
            binary_array = (mask_array > threshold).astype(np.uint8) * 255
            binary_mask = Image.fromarray(binary_array)

            if self.mask_transform:
                mask = self.mask_transform(binary_mask).squeeze(0)
            else:
                mask = transforms.ToTensor()(binary_mask).squeeze(0)  # shape: (H, W)

            # Optionally binarize the mask (assuming pixel values are normalized 0-1)
            mask = (mask > 0.5).long()

        else:
            # For good images or if mask is not available: create a blank (zero) mask.
            # Create a blank mask with the same height and width as the image.
            # Since image is tensor of shape (3, H, W), extract H, W.
            _, H, W = image.shape
            mask = torch.zeros((H, W), dtype=torch.long)

        return image, label, mask.type(torch.long)

# Function to display images and masks (for anomalies)
def show_images_and_masks(dataset, num_images=10):
    indices = random.sample(range(len(dataset)), num_images)
    images = []
    masks = []

    for i in indices:
        image, label, mask = dataset[i]
        images.append(image)
        masks.append(mask.unsqueeze(0))

    # Create grids for images and masks
    image_grid = make_grid(torch.stack(images), nrow=5)  # adjust nrow as needed
    mask_grid = make_grid(torch.stack(masks), nrow=5)

    # Plot the images and masks
    fig, axs = plt.subplots(1, 2, figsize=(40, 20))
    axs[0].imshow(image_grid.permute(1, 2, 0))  # Permute to (H,W,C) for matplotlib
    axs[0].set_title('Images')
    axs[0].axis('off')
    axs[1].imshow(mask_grid.permute(1, 2, 0),cmap = 'gray')
    axs[1].set_title('Masks')
    axs[1].axis('off')
    plt.show()

def plot_images_side_by_side(image1, image2):
    """Plots two images side by side using matplotlib.

    Args:
        image1: The first image (can be a PIL Image or a NumPy array).
        image2: The second image (can be a PIL Image or a NumPy array).
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Adjust figsize as needed

    axes[0].imshow(image1)
    axes[0].set_title("Image")
    axes[0].axis("off")  # Hide axis ticks and labels

    axes[1].imshow(image2, cmap = "gray")
    axes[1].set_title("Mask")
    axes[1].axis("off")

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()


class OnTheFlyAugmentationDataset(Dataset):
    """
    A dataset wrapper that performs on-the-fly augmentation for anomaly samples.

    For each sample from the base dataset (which returns (image, label, mask) as tensors),
    if the sample is an anomaly (label==1) and with a probability defined by
    augmentation_probability, the sample is augmented using a joint augmentation function.

    This avoids doubling RAM usage by not storing extra copies of augmented samples.
    """
    def __init__(self, base_dataset, album_transform):
        """
        Args:
            base_dataset (Dataset): The original dataset that returns (image, label, mask).
        """
        self.base_dataset = base_dataset
        self.album_transform = album_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, label, mask = self.base_dataset[index]

        # Only apply augmentation for anomaly samples (label == 1) with given probability.
        if label == 1:

          np_image = image.detach().cpu().numpy()
          image = np.transpose(np_image, (1, 2, 0))

          mask = mask.detach().cpu().numpy()

          # Apply augmentation
          augmented = self.album_transform(image=image, mask=mask)

        return augmented["image"], label, augmented["mask"]
    

class CustData(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Retrieve image, label, and segmentation mask from the dataset.
        img, label, mask = self.dataset[idx]
        mask = np.array(mask)

        # Get unique instance IDs, ignoring background (assumed to be 0)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]

        # If no objects are found, return None
        if len(obj_ids) == 0:
            return None

        boxes = []
        masks_list = []

        # Process each unique instance id
        for obj_id in obj_ids:
            # Create binary mask for current instance
            binary_mask = (mask == obj_id).astype(np.uint8)

            # Optional: Clean up the mask using morphological opening to remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

            # Use connected components with 4-connectivity to break disjoint regions
            num_labels, labels_im = cv2.connectedComponents(clean_mask, connectivity=4)

            # For each connected region, compute the bounding box
            for comp in range(1, num_labels):
                comp_mask = (labels_im == comp)
                if comp_mask.sum() == 0:
                    continue

                # Locate the pixel coordinates
                pos_y, pos_x = np.where(comp_mask)
                xmin = pos_x.min()
                xmax = pos_x.max()
                ymin = pos_y.min()
                ymax = pos_y.max()

                # Account for degenerate regions (e.g., a single pixel or a thin line)
                if xmax <= xmin:
                    xmin = max(xmin - 1, 0)
                    xmax = min(xmax + 1, mask.shape[1] - 1)
                if ymax <= ymin:
                    ymin = max(ymin - 1, 0)
                    ymax = min(ymax + 1, mask.shape[0] - 1)

                boxes.append([xmin, ymin, xmax, ymax])
                masks_list.append(comp_mask.astype(np.uint8))

        if len(masks_list) == 0:
          return None

        # Convert lists to Torch tensors
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.ones((len(boxes),), dtype=torch.int64)
        masks_tensor = torch.as_tensor(np.stack(masks_list, axis=0), dtype=torch.uint8)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "masks": masks_tensor
        }
        return img, target
    

def visualize_sample(dataset, index):
  """Visualizes the image, bounding box, and mask for a given index in the dataset.

  Args:
      dataset: The dataset containing images, bounding boxes, and masks.
      index: The index of the sample to visualize.
  """
  try:
    image, target = dataset[index]

    # Convert image to numpy array and permute dimensions if necessary
    image = image.permute(1, 2, 0).numpy()

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis("off")

    #Bounding boxes
    boxes = target["boxes"]
    for box in boxes:
        xmin, ymin, xmax, ymax = box.int().tolist()
        rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, edgecolor='red', linewidth=2)
        axes[1].add_patch(rect)
    axes[1].imshow(image)
    axes[1].set_title("Bounding Boxes")
    axes[1].axis("off")

    # Masks
    masks = target["masks"]
    axes[2].imshow(image)
    for mask in masks:
        axes[2].imshow(mask, alpha = 0.5, cmap='gray')  # Overlay masks with transparency
    axes[2].set_title("Masks")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

  except (IndexError, TypeError) as e:
    print(f"Error visualizing sample at index {index}: {e}")
    if index >= len(dataset):
        print(f"Index {index} out of range for dataset of size {len(dataset)}")
    elif dataset[index] is None:
        print(f"Sample at index {index} is None (likely no objects were found)")


def visualize_samples(dataset, num_images=5, indices = None):
    """Visualizes images, masks, and bounding boxes from a dataset."""
    if indices is None:
      indices = random.sample(range(len(dataset)), num_images)

    plt.figure(figsize=(20, 10))  # Adjust figure size for better visualization

    for i, index in enumerate(indices):
        try:
          img, target = dataset[index]
        except:
          print(index)
          continue

        masks = target["masks"]
        boxes = target["boxes"]

        plt.subplot(1, num_images, i + 1)  # Create subplots
        plt.imshow(img.permute(1, 2, 0))  # Display image
        plt.axis('off')  # Hide axis

        # Plot bounding boxes
        for box in boxes:
          xmin, ymin, xmax, ymax = box.numpy()
          plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                             linewidth=2, edgecolor='r', facecolor='none'))

        # Overlay masks
        for mask in masks:
          plt.imshow(mask.squeeze(0), alpha=0.5, cmap='jet') # Adjust alpha for visibility

        plt.title(f"Image {index}")
    plt.tight_layout() #adjust spacing
    plt.show()

def visualize_gt_and_predictions(test_dataset, model, plot_boxes=True, score_threshold=0.5, device='cuda', n_samples=5):
    """
    Visualizes ground truth vs. model predictions for randomly selected test images.

    For each selected sample:
      - The top row shows the image with ground truth masks (e.g., in green) and bounding boxes.
      - The bottom row shows the image overlaid with predicted masks (in red) and bounding boxes.

    Args:
        test_dataset: A dataset where each sample is (image, target). The target is expected
                      to be a dictionary with keys "masks" and "boxes".
        model: The trained segmentation/detection model which, given an image input, returns a list
               of dictionaries with keys like "masks", "boxes", and optionally "scores".
        plot_boxes (bool): Whether to overlay bounding boxes on both ground truth and predicted images.
        score_threshold (float): Threshold for filtering prediction boxes (if scores are provided).
        device (str): The device to run the model on (e.g., 'cuda' or 'cpu').
        n_samples (int): Number of random samples to visualize.
    """
    model.eval()
    indices = random.sample(range(len(test_dataset)), n_samples)

    # Create a figure with 2 rows and n_samples columns.
    fig, axs = plt.subplots(2, n_samples, figsize=(4 * n_samples, 10))

    for i, idx in enumerate(indices):
        # -----------------
        # Ground Truth (Top Row)
        # -----------------
        sample = test_dataset[idx]
        if isinstance(sample, tuple) and len(sample) >= 2:
            image, target = sample[0], sample[1]
        else:
            raise ValueError("Test sample should be a tuple (image, target).")

        # Convert image to numpy array (if necessary)
        if torch.is_tensor(image):
            image_np = image.detach().cpu().numpy()
            # If the shape is (C, H, W) with one or three channels, transpose it to (H, W, C)
            if image_np.ndim == 3 and image_np.shape[0] in [1, 3]:
                image_np = np.transpose(image_np, (1, 2, 0))
            # If image values are normalized ([0, 1]), scale them to 0-255
            if image_np.dtype == np.float32 and image_np.max() <= 1:
                image_np = (image_np * 255).astype(np.uint8)
        elif isinstance(image, np.ndarray):
            image_np = image.copy()
        else:
            raise ValueError("Image must be a torch tensor or a NumPy array.")

        ax_gt = axs[0, i]
        ax_gt.imshow(image_np)
        ax_gt.set_title("Ground Truth")
        ax_gt.axis("off")

        # Overlay Ground Truth Masks (if available)
        if "masks" in target:
            gt_masks = target["masks"]
            if torch.is_tensor(gt_masks):
                gt_masks = gt_masks.detach().cpu().numpy()
            # Ensure the masks array is 3D: (num_masks, H, W)
            if gt_masks.ndim == 2:
                gt_masks = np.expand_dims(gt_masks, axis=0)
            for mask in gt_masks:
                colored_mask = np.zeros((*mask.shape, 4), dtype=np.uint8)
                # Use green color for ground truth masks
                colored_mask[..., 1] = 255
                # Apply semi-transparency where the mask is on
                colored_mask[..., 3] = (mask > 0) * 127
                ax_gt.imshow(colored_mask)

        # Overlay Ground Truth Bounding Boxes (if available)
        if plot_boxes and "boxes" in target:
            boxes = target["boxes"]
            if torch.is_tensor(boxes):
                boxes = boxes.detach().cpu().numpy()
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                         linewidth=2, edgecolor='yellow', facecolor='none')
                ax_gt.add_patch(rect)

        # -----------------
        # Prediction (Bottom Row)
        # -----------------
        ax_pred = axs[1, i]
        ax_pred.imshow(image_np)
        ax_pred.set_title("Prediction")
        ax_pred.axis("off")

        # Prepare image for model inference
        if torch.is_tensor(image):
            image_input = image.clone()
        else:
            # If image was a numpy array, convert it to tensor and rearrange dimensions.
            image_input = torch.from_numpy(image_np).permute(2, 0, 1)
        if image_input.dim() == 3:
            image_input = image_input.unsqueeze(0)
        image_input = image_input.to(device)

        # Obtain model predictions.
        with torch.no_grad():
            prediction = model(image_input)[0]

        # Process Predicted Masks (if available)
        if "masks" in prediction:
            pred_masks = prediction["masks"]
            # Often predictions are in shape [N, 1, H, W]
            if pred_masks.dim() == 4:
                pred_masks = pred_masks[:, 0, :, :]
            pred_masks = pred_masks.detach().cpu().numpy()
            binary_masks = (pred_masks > 0.5).astype(np.uint8)
            for mask in binary_masks:
                colored_mask = np.zeros((*mask.shape, 4), dtype=np.uint8)
                # Use red color for predicted masks
                colored_mask[..., 0] = 255
                colored_mask[..., 3] = mask * 127  # Semi-transparent alpha channel
                ax_pred.imshow(colored_mask)

        # Process Predicted Bounding Boxes (if available)
        if plot_boxes and "boxes" in prediction:
            boxes = prediction["boxes"]
            if "scores" in prediction:
                scores = prediction["scores"].detach().cpu().numpy()
                # Filter out low-scoring predictions
                keep = scores >= score_threshold
                boxes = boxes[keep]
            boxes = boxes.detach().cpu().numpy()
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                         linewidth=2, edgecolor='yellow', facecolor='none')
                ax_pred.add_patch(rect)

    plt.tight_layout()
    plt.show()

def visualize_random_images(test_dataset, plot_boxes=True):
    """
    Visualizes 5 random images sampled from the test dataset.

    For each sample, the function displays:
      - The image (assumed to be in a torch tensor or numpy array format).
      - All segmentation masks overlaid on the image with a semi-transparent red color.
      - Bounding boxes (if available in the target and if plot_boxes=True).

    Args:
        test_dataset: The dataset object. Expected to return either (image, target)
                      or (image, label, mask).
        plot_boxes (bool): Whether to overlay bounding boxes (if available in target).
    """
    # Randomly sample 5 indices from the dataset (without replacement)
    indices = random.sample(range(len(test_dataset)), 5)

    # Create a 1x5 grid for visualization
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))

    for ax, idx in zip(axs, indices):
        sample = test_dataset[idx]

        # Unpack sample based on its structure:
        if isinstance(sample, tuple) and len(sample) == 2:
            # Expected format: (image, target)
            image, target = sample
        elif isinstance(sample, tuple) and len(sample) == 3:
            # Expected format: (image, label, mask)
            image, label, mask = sample
            # We'll create a minimal target dictionary with the mask.
            target = {"masks": mask}
        else:
            raise ValueError("Dataset sample must be a tuple of length 2 or 3.")

        # Convert image to a numpy array if it's a torch tensor.
        # Also, if the shape is (C, H, W), we convert it to (H, W, C).
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
            if image.ndim == 3:  # expect (C, H, W) or (H, W, C)
                if image.shape[0] == 1 or image.shape[0] == 3:
                    image = np.transpose(image, (1, 2, 0))
            # Assume image is normalized to [0, 1] if float—rescale to 0-255.
            if image.dtype == np.float32 and image.max() <= 1:
                image = (image * 255).astype(np.uint8)
        elif not isinstance(image, np.ndarray):
            raise ValueError("Image must be a torch tensor or a numpy array")

        # Display the image.
        ax.imshow(image)
        ax.axis("off")

        # Overlay masks (if available in target)
        if "masks" in target:
            masks = target["masks"]
            if torch.is_tensor(masks):
                masks = masks.detach().cpu().numpy()
            # The expected shape is either (H, W) or (num_masks, H, W)
            if masks.ndim == 2:
                masks = np.expand_dims(masks, axis=0)
            num_masks = masks.shape[0]
            for i in range(num_masks):
                mask = masks[i]
                # Make a colored overlay where the mask's nonzero area gets a red tint.
                # We create an RGBA overlay image.
                colored_mask = np.zeros((*mask.shape, 4), dtype=np.uint8)
                colored_mask[..., 0] = 255   # Red channel
                colored_mask[..., 3] = (mask > 0) * 127  # Alpha channel (semi-transparent)
                # Overlay the colored mask.
                ax.imshow(colored_mask)

        # Draw bounding boxes (if available and requested)
        if plot_boxes and "boxes" in target:
            boxes = target["boxes"]
            if torch.is_tensor(boxes):
                boxes = boxes.detach().cpu().numpy()
            # Expect boxes as an array or list of shape (num_boxes, 4)
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                # Create a rectangle patch. Adjust parameters (linewidth, color) as desired.
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                         linewidth=2, edgecolor='yellow', facecolor='none')
                ax.add_patch(rect)

    plt.tight_layout()
    plt.show()

def count_parameters(model):
  table = PrettyTable(["Modules", "Parameters"])
  total_params = 0
  for name, parameter in model.named_parameters():
    if not parameter.requires_grad: continue
    param = parameter.numel()
    table.add_row([name, param])
    total_params+=param
  #print(table)
  print(f"Total Trainable Params: {total_params}")
#   return total_params


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    for inst in tqdm(dataloader, desc='Train', leave=False):
        # Skip empty batches
        if not inst:
            continue

        # Filter out samples with no bounding boxes in the target
        inst = [(image, target) for image, target in inst if target['boxes'].shape[0] != 0]

        # Skip batch if all samples were filtered out
        if not inst:
            continue

        images = [image.to(device) for image, _ in inst]
        targets = [{k: v.to(device) for k, v in t.items()} for _, t in inst]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item() * len(images)

    return running_loss / len(dataloader.dataset)

def compute_iou(mask1, mask2):
    """
    Computes Intersection over Union between two binary masks.
    """
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

def validate_model(model, dataloader, device, pred_mask_threshold=0.5, iou_threshold=0.5):
    """
    Evaluate the model by computing the average IoU between predicted masks
    and ground truth masks.
    """
    model.eval()
    all_ious = []
    for inst in tqdm(dataloader, desc='Validation', leave=False):
        if not inst:
            continue

        images = [image.to(device) for image, _ in inst]
        targets = [{k: v.to(device) for k, v in t.items()} for _, t in inst]

        with torch.no_grad():
            outputs = model(images)  # list of dicts per image

        for output, target in zip(outputs, targets):
            # Ground-truth masks (expected shape: [num_objs, H, W])
            gt_masks = target['masks'].cpu().numpy()
            # Predicted masks (shape: [num_preds, 1, H, W]) as float probabilities
            pred_masks = output['masks'].detach().cpu().numpy()
            pred_masks = (pred_masks > pred_mask_threshold).astype(np.uint8)
            if pred_masks.ndim == 4 and pred_masks.shape[1] == 1:
                pred_masks = np.squeeze(pred_masks, axis=1)

            for gt_mask in gt_masks:
                best_iou = 0
                for pred_mask in pred_masks:
                    iou = compute_iou(gt_mask, pred_mask)
                    if iou > best_iou:
                        best_iou = iou
                if best_iou >= iou_threshold:
                    all_ious.append(best_iou)

    average_iou = np.mean(all_ious) if all_ious else 0.0
    return average_iou