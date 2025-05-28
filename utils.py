import torch
import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm


def count_parameters(model):
  table = PrettyTable(["Modules", "Parameters"])
  total_params = 0
  for name, parameter in model.named_parameters():
    if not parameter.requires_grad: continue
    param = parameter.numel()
    table.add_row([name, param])
    total_params+=param
#   print(table)
  return total_params


def train_epoch(model, dataloader, optimizer, device, scaler=None):
    """
    Optimized training epoch with gradient scaling and detailed loss tracking.
    
    Args:
        model: The model to train
        dataloader: DataLoader containing the training data
        optimizer: The optimizer
        device: Device to train on
        scaler: Optional gradient scaler for mixed precision training
    """
    model.train()
    running_losses = {
        'total_loss': 0.0,
        'loss_classifier': 0.0,
        'loss_box_reg': 0.0,
        'loss_mask': 0.0,
        'loss_objectness': 0.0,
        'loss_rpn_box_reg': 0.0
    }
    num_samples = 0
    
    progress_bar = tqdm(dataloader, desc='Training', leave=False)
    
    for inst in progress_bar:
        try:
            images, targets = inst
            # Convert list of images to a tensor list on device
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            
            # Clear gradients
            optimizer.zero_grad(set_to_none=True)
            
            if scaler is not None:
                # Mixed precision training
                with torch.cuda.amp.autocast():
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular training
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                optimizer.step()
            
            # Update running losses
            batch_size = len(images)
            num_samples += batch_size
            running_losses['total_loss'] += losses.item() * batch_size
            
            # Track individual losses
            for k, v in loss_dict.items():
                if k in running_losses:
                    running_losses[k] += v.item() * batch_size
            
            # Update progress bar with current losses
            progress_bar.set_postfix({
                'total_loss': f"{running_losses['total_loss']/num_samples:.4f}",
                'mask_loss': f"{running_losses['loss_mask']/num_samples:.4f}"
            })
            
            # Clear GPU cache periodically
            if torch.cuda.is_available() and progress_bar.n % 10 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error in batch: {str(e)}")
            continue
    
    if num_samples == 0:
        return running_losses
    
    # Calculate average losses
    avg_losses = {k: v / num_samples for k, v in running_losses.items()}
    
    return avg_losses

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
    Optimized model validation with improved metric tracking and efficient batch processing.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader containing the validation data
        device: Device to validate on
        pred_mask_threshold: Threshold for mask predictions
        iou_threshold: IoU threshold for considering a prediction correct
    
    Returns:
        dict: Dictionary containing validation metrics
    """
    model.eval()
    metrics = {
        'all_ious': [],
        'class_ious': {},
        'class_counts': {},
        'precision': [],
        'recall': [],
        'total_instances': 0,
        'detected_instances': 0,
        'false_positives': 0,
        'class_confidence': {}
    }
    
    progress_bar = tqdm(dataloader, desc='Validation', leave=False)
    
    try:
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            for batch_idx, inst in enumerate(progress_bar):
                if not inst:
                    continue

                # Efficient batch processing
                images = [image.to(device, non_blocking=True) for image, _ in inst]
                targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for _, t in inst]

                # Get model predictions
                outputs = model(images)

                # Process each image in the batch
                for output, target in zip(outputs, targets):
                    # Move data to CPU for numpy operations
                    gt_masks = target['masks'].cpu().numpy()
                    gt_labels = target['labels'].cpu().numpy()
                    metrics['total_instances'] += len(gt_masks)
                    
                    # Update class counts
                    for label in gt_labels:
                        label = int(label)
                        metrics['class_counts'][label] = metrics['class_counts'].get(label, 0) + 1

                    # Get predicted masks, scores, and labels
                    pred_masks = output['masks'].cpu().numpy()
                    pred_scores = output['scores'].cpu().numpy()
                    pred_labels = output['labels'].cpu().numpy()

                    # Apply threshold to predicted masks
                    pred_masks = (pred_masks > pred_mask_threshold).astype(np.uint8)
                    if pred_masks.ndim == 4 and pred_masks.shape[1] == 1:
                        pred_masks = np.squeeze(pred_masks, axis=1)

                    # Match predictions to ground truth using Hungarian algorithm
                    num_gt = len(gt_masks)
                    num_pred = len(pred_masks)
                    
                    if num_pred > 0 and num_gt > 0:
                        iou_matrix = np.zeros((num_gt, num_pred))
                        for i, gt_mask in enumerate(gt_masks):
                            for j, pred_mask in enumerate(pred_masks):
                                if gt_labels[i] == pred_labels[j]:
                                    iou_matrix[i, j] = compute_iou(gt_mask, pred_mask)

                        # Use Hungarian algorithm for optimal matching
                        from scipy.optimize import linear_sum_assignment
                        gt_indices, pred_indices = linear_sum_assignment(-iou_matrix)
                        
                        # Process matched pairs
                        for gt_idx, pred_idx in zip(gt_indices, pred_indices):
                            iou = iou_matrix[gt_idx, pred_idx]
                            if iou >= iou_threshold:
                                metrics['all_ious'].append(iou)
                                metrics['detected_instances'] += 1
                                
                                # Update class-specific metrics
                                label = int(pred_labels[pred_idx])
                                if label not in metrics['class_ious']:
                                    metrics['class_ious'][label] = []
                                    metrics['class_confidence'][label] = []
                                metrics['class_ious'][label].append(iou)
                                metrics['class_confidence'][label].append(pred_scores[pred_idx])
                            
                        # Calculate precision and recall for this image
                        true_positives = sum(iou_matrix[gt_idx, pred_idx] >= iou_threshold 
                                           for gt_idx, pred_idx in zip(gt_indices, pred_indices))
                        metrics['false_positives'] += num_pred - true_positives
                        
                        precision = true_positives / num_pred if num_pred > 0 else 0
                        recall = true_positives / num_gt if num_gt > 0 else 0
                        
                        metrics['precision'].append(precision)
                        metrics['recall'].append(recall)

                # Update progress bar
                mean_iou = np.mean(metrics['all_ious']) if metrics['all_ious'] else 0
                progress_bar.set_postfix({
                    'mIoU': f"{mean_iou:.4f}",
                    'Detected': metrics['detected_instances']
                })

    except Exception as e:
        print(f"Error during validation: {str(e)}")
        return None

    # Calculate final metrics
    results = {
        'mean_iou': np.mean(metrics['all_ious']) if metrics['all_ious'] else 0.0,
        'class_mean_ious': {k: np.mean(v) for k, v in metrics['class_ious'].items()},
        'class_mean_confidence': {k: np.mean(v) for k, v in metrics['class_confidence'].items()},
        'mean_precision': np.mean(metrics['precision']) if metrics['precision'] else 0.0,
        'mean_recall': np.mean(metrics['recall']) if metrics['recall'] else 0.0,
        'detection_rate': metrics['detected_instances'] / metrics['total_instances'] if metrics['total_instances'] > 0 else 0.0,
        'false_positive_rate': metrics['false_positives'] / metrics['total_instances'] if metrics['total_instances'] > 0 else 0.0,
        'class_detection_rates': {
            k: len(metrics['class_ious'].get(k, [])) / count 
            for k, count in metrics['class_counts'].items()
        }
    }
    
    # Add F1 score
    if results['mean_precision'] + results['mean_recall'] > 0:
        results['f1_score'] = 2 * (results['mean_precision'] * results['mean_recall']) / (results['mean_precision'] + results['mean_recall'])
    else:
        results['f1_score'] = 0.0

    return results

def collate_fn(batch):
    """
    Custom collate function for Mask R-CNN data.
    This replaces the lambda x: tuple(zip(*x)) with a proper named function
    that can be pickled for multiprocessing.
    
    Args:
        batch: A list of tuples (image, target)
    Returns:
        tuple: (list_of_images, list_of_targets)
    """
    return tuple(zip(*batch))