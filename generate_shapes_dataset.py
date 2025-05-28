import os
import sys
import random
import math
import numpy as np
import cv2
from tqdm import tqdm

class ShapesDataset:
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes placed randomly on a blank surface.
    """
    def __init__(self):
        self.class_names = ['BG', 'square', 'circle', 'triangle', 'hexagon', 'pentagon', 'star']

    def draw_shape(self, image, shape, dims, color):
        """Draws a shape from the given specs."""
        # Get the center x, y and the size s
        x, y, s = dims
        if shape == 'square':
            cv2.rectangle(image, (x-s, y-s), (x+s, y+s), color, -1)
            bbox = [x-s, y-s, x+s, y+s]
        elif shape == "circle":
            cv2.circle(image, (x, y), s, color, -1)
            bbox = [x-s, y-s, x+s, y+s]
        elif shape == "triangle":
            points = np.array([[(x, y-s),
                              (x-s/math.sin(math.radians(60)), y+s),
                              (x+s/math.sin(math.radians(60)), y+s),
                            ]], dtype=np.int32)
            cv2.fillPoly(image, points, color)
            bbox = [
                int(x-s/math.sin(math.radians(60))),  # x1
                y-s,                                   # y1
                int(x+s/math.sin(math.radians(60))),  # x2
                y+s                                    # y2
            ]
        elif shape == "hexagon":
            # Create a regular hexagon
            angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 points
            points = np.array([[int(x + s*np.cos(a)), int(y + s*np.sin(a))] for a in angles])
            cv2.fillPoly(image, [points], color)
            bbox = [x-s, y-s, x+s, y+s]
        elif shape == "pentagon":
            # Create a regular pentagon
            angles = np.linspace(0, 2*np.pi, 6)[:-1]  # 5 points
            points = np.array([[int(x + s*np.cos(a)), int(y + s*np.sin(a))] for a in angles])
            cv2.fillPoly(image, [points], color)
            bbox = [x-s, y-s, x+s, y+s]
        elif shape == "star":
            # Create a 5-pointed star
            outer_points = []
            inner_points = []
            for i in range(5):
                # Outer points of the star
                angle = (2 * np.pi * i) / 5 - np.pi / 2
                outer_points.append([
                    int(x + s * np.cos(angle)),
                    int(y + s * np.sin(angle))
                ])
                # Inner points of the star (36-degree offset, shorter radius)
                angle += np.pi / 5
                inner_points.append([
                    int(x + 0.4*s * np.cos(angle)),
                    int(y + 0.4*s * np.sin(angle))
                ])
            
            # Interleave outer and inner points
            points = []
            for i in range(5):
                points.append(outer_points[i])
                points.append(inner_points[i])
            
            cv2.fillPoly(image, [np.array(points)], color)
            bbox = [x-s, y-s, x+s, y+s]
        
        return image, bbox

    def random_shape(self, height, width, shape=None):
        """Generates specifications of a random or specified shape."""
        # Shape
        if shape is None:
            shape = random.choice(self.class_names[1:])  # Exclude BG
        # Color
        color = tuple([random.randint(0, 255) for _ in range(3)])
        # Center x, y
        buffer = 20
        y = random.randint(buffer, height - buffer - 1)
        x = random.randint(buffer, width - buffer - 1)
        # Size
        s = random.randint(buffer, height//4)
        return shape, color, (x, y, s)

    def random_image(self, height, width, num_shapes=None):
        """Creates random specifications of an image with multiple shapes."""
        # Pick random background color
        bg_color = np.array([random.randint(0, 255) for _ in range(3)])
        
        # Generate shapes ensuring equal distribution
        shapes = []
        if num_shapes is None:
            num_shapes = random.randint(1, 4)
        
        # Get all shape types (excluding BG) and shuffle them
        available_shapes = self.class_names[1:]
        random.shuffle(available_shapes)
        
        # Select shapes to use
        for i in range(num_shapes):
            shape = available_shapes[i % len(available_shapes)]  # Cycle through shapes
            shape_spec = self.random_shape(height, width, shape)
            shapes.append(shape_spec)
            
        return bg_color, shapes    
    
    def create_image_and_mask(self, height, width, num_shapes=None, forced_shape=None):
        """Generate image, mask, and bounding boxes."""
        bg_color, shapes = self.random_image(height, width, num_shapes)
        
        if forced_shape is not None:
            # Override the shape type while keeping position and color
            shapes = [(forced_shape, color, dims) for _, color, dims in shapes]
        
        # Create image
        image = np.ones([height, width, 3], dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)
        
        # Create mask
        mask = np.zeros([height, width, len(shapes)], dtype=np.uint8)
        
        # Store bounding boxes and shape classes
        bboxes = []
        shape_classes = []
        
        # Draw shapes
        for i, (shape, color, dims) in enumerate(shapes):
            image, bbox = self.draw_shape(image, shape, dims, color)
            mask_slice = np.zeros([height, width, 1], dtype=np.uint8)
            mask[:, :, i:i+1], _ = self.draw_shape(mask_slice, shape, dims, 1)
            
            # Store bbox and class
            bboxes.append(bbox)
            shape_classes.append(self.class_names.index(shape))
        
        return image, mask, bboxes, shape_classes

def generate_dataset(output_dir, num_train=5000, num_val=2000, image_size=128):
    """Generate and save the shapes dataset."""
    dataset = ShapesDataset()
    
    # Calculate images per shape type
    num_shape_types = len(dataset.class_names) - 1  # Exclude BG
    train_imgs_per_shape = num_train // num_shape_types
    val_imgs_per_shape = num_val // num_shape_types
    
    # Create directories
    train_img_dir = os.path.join(output_dir, 'train', 'images')
    train_mask_dir = os.path.join(output_dir, 'train', 'masks')
    train_anno_dir = os.path.join(output_dir, 'train', 'annotations')
    val_img_dir = os.path.join(output_dir, 'val', 'images')
    val_mask_dir = os.path.join(output_dir, 'val', 'masks')
    val_anno_dir = os.path.join(output_dir, 'val', 'annotations')
    
    # Create all directories
    for d in [train_img_dir, train_mask_dir, train_anno_dir,
              val_img_dir, val_mask_dir, val_anno_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Generate training data
    print("Generating training data...")
    for shape_type in dataset.class_names[1:]:  # Exclude BG
        print(f"Generating {train_imgs_per_shape} training images for {shape_type}")
        for i in tqdm(range(train_imgs_per_shape)):
            # Force specific shape type for equal distribution
            image, mask, bboxes, shape_classes = dataset.create_image_and_mask(
                image_size, image_size, num_shapes=1, forced_shape=shape_type)
              # Calculate unique index for each shape type
            shape_idx = dataset.class_names.index(shape_type) - 1
            idx = shape_idx * train_imgs_per_shape + i
            
            # Save image
            image_path = os.path.join(train_img_dir, f'train_shape_{idx:04d}.png')
            cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # Save individual masks
            for j in range(mask.shape[-1]):
                mask_path = os.path.join(train_mask_dir, f'train_shape_{idx:04d}_mask_{j}.png')
                cv2.imwrite(mask_path, mask[:, :, j] * 255)
            
            # Save annotations
            anno_path = os.path.join(train_anno_dir, f'train_shape_{idx:04d}.txt')
            with open(anno_path, 'w') as f:
                for bbox, cls in zip(bboxes, shape_classes):
                    f.write(f"{cls} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
    
    # Generate validation data
    print("\nGenerating validation data...")
    for shape_type in dataset.class_names[1:]:  # Exclude BG
        print(f"Generating {val_imgs_per_shape} validation images for {shape_type}")
        for i in tqdm(range(val_imgs_per_shape)):
            # Force specific shape type for equal distribution
            image, mask, bboxes, shape_classes = dataset.create_image_and_mask(
                image_size, image_size, num_shapes=1, forced_shape=shape_type)
              # Calculate unique index for each shape type
            shape_idx = dataset.class_names.index(shape_type) - 1
            idx = shape_idx * val_imgs_per_shape + i
            
            # Save image
            image_path = os.path.join(val_img_dir, f'val_shape_{idx:04d}.png')
            cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # Save individual masks
            for j in range(mask.shape[-1]):
                mask_path = os.path.join(val_mask_dir, f'val_shape_{idx:04d}_mask_{j}.png')
                cv2.imwrite(mask_path, mask[:, :, j] * 255)
            
            # Save annotations
            anno_path = os.path.join(val_anno_dir, f'val_shape_{idx:04d}.txt')
            with open(anno_path, 'w') as f:
                for bbox, cls in zip(bboxes, shape_classes):
                    f.write(f"{cls} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create dataset directory
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'shapes')
    
    # Generate dataset
    print(f"Generating shapes dataset in {data_dir}")
    generate_dataset(
        output_dir=data_dir,
        num_train=5000,  # 5000 training images
        num_val=2000,    # 2000 validation images
        image_size=128   # Same as CONFIG.IMAGE_SHAPE
    )
    
    print("✅ Dataset generation complete!")
    print(f"📁 Dataset saved in: {os.path.abspath(data_dir)}")

if __name__ == "__main__":
    main()
