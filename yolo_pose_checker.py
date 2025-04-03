import os
import glob
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class YoloPoseChecker:
    def __init__(self, images_dir, labels_dir, num_keypoints=17, visualization_dir=None, samples_to_visualize=5):
        """
        Initialize the YOLO pose dataset checker.
        
        Args:
            images_dir: Directory containing image files
            labels_dir: Directory containing label files
            num_keypoints: Expected number of keypoints (default: 17 for COCO format)
            visualization_dir: Directory to save visualization images (if None, won't visualize)
            samples_to_visualize: Number of random samples to visualize
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.num_keypoints = num_keypoints
        self.visualization_dir = Path(visualization_dir) if visualization_dir else None
        self.samples_to_visualize = samples_to_visualize
        
        if self.visualization_dir and not self.visualization_dir.exists():
            self.visualization_dir.mkdir(parents=True)
        
        # File extensions to check
        self.img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'total_labels': 0,
            'missing_labels': 0,
            'missing_images': 0,
            'empty_labels': 0,
            'format_issues': 0,
            'bbox_issues': 0,
            'keypoint_issues': 0,
            'files_with_issues': [],
            'class_distribution': {}
        }

    def get_image_files(self):
        """Get all image files from the images directory."""
        image_files = []
        for ext in self.img_extensions:
            image_files.extend(list(self.images_dir.glob(f'*{ext}')))
        return image_files

    def get_label_files(self):
        """Get all label files from the labels directory."""
        return list(self.labels_dir.glob('*.txt'))

    def check_missing_files(self, image_files, label_files):
        """Check for missing label or image files."""
        image_basenames = {file.stem for file in image_files}
        label_basenames = {file.stem for file in label_files}
        
        missing_labels = image_basenames - label_basenames
        missing_images = label_basenames - image_basenames
        
        self.stats['missing_labels'] = len(missing_labels)
        self.stats['missing_images'] = len(missing_images)
        
        print(f"Found {len(missing_labels)} images without label files")
        if missing_labels:
            print(f"Sample missing labels: {list(missing_labels)[:5]}")
            self.stats['files_with_issues'].extend([f"{name} (missing label)" for name in list(missing_labels)[:5]])
            
        print(f"Found {len(missing_images)} labels without image files")
        if missing_images:
            print(f"Sample missing images: {list(missing_images)[:5]}")
            self.stats['files_with_issues'].extend([f"{name} (missing image)" for name in list(missing_images)[:5]])
            
        return missing_labels, missing_images

    def check_annotation_format(self, label_files):
        """Check if annotation files match expected YOLO pose format."""
        format_issues = []
        empty_files = []
        
        for label_file in tqdm(label_files, desc="Checking annotation format"):
            try:
                content = label_file.read_text().strip()
                
                # Check if file is empty
                if not content:
                    empty_files.append(label_file)
                    continue
                
                # Check each annotation line
                lines = content.split('\n')
                for line_idx, line in enumerate(lines):
                    parts = line.strip().split()
                    
                    # Calculate expected parts: class_id + bbox(4) + keypoints(num_keypoints*3)
                    expected_parts = 1 + 4 + (self.num_keypoints * 3)
                    
                    if len(parts) != expected_parts:
                        format_issues.append((label_file, f"Line {line_idx+1}: Expected {expected_parts} values, got {len(parts)}"))
                    else:
                        # Update class distribution
                        class_id = parts[0]
                        if class_id in self.stats['class_distribution']:
                            self.stats['class_distribution'][class_id] += 1
                        else:
                            self.stats['class_distribution'][class_id] = 1
                
            except Exception as e:
                format_issues.append((label_file, str(e)))
                
        self.stats['empty_labels'] = len(empty_files)
        self.stats['format_issues'] = len(format_issues)
        
        print(f"Found {len(empty_files)} empty label files")
        if empty_files:
            print(f"Sample empty files: {[f.name for f in empty_files[:5]]}")
            self.stats['files_with_issues'].extend([f"{f.stem} (empty label)" for f in empty_files[:5]])
            
        print(f"Found {len(format_issues)} files with format issues")
        if format_issues:
            for file, issue in format_issues[:5]:
                print(f"  {file.name}: {issue}")
                self.stats['files_with_issues'].append(f"{file.stem} ({issue})")
                
        return format_issues, empty_files

    def check_bbox_and_keypoints(self, label_files, image_files):
        """Check for issues with bounding boxes and keypoints."""
        bbox_issues = []
        keypoint_issues = []
        
        # Create a dictionary for quick image lookup
        image_dict = {img.stem: img for img in image_files}
        
        for label_file in tqdm(label_files, desc="Checking bounding boxes and keypoints"):
            try:
                content = label_file.read_text().strip()
                if not content:
                    continue
                
                # Get corresponding image file
                img_file = image_dict.get(label_file.stem)
                if not img_file:
                    continue
                    
                # Load image to get dimensions
                img = cv2.imread(str(img_file))
                if img is None:
                    print(f"Warning: Could not load image {img_file}")
                    continue
                    
                img_height, img_width = img.shape[:2]
                
                lines = content.split('\n')
                for line_idx, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) < 5:  # Need at least class + bbox
                        continue
                    
                    try:
                        # Check bounding box values
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        
                        # Check if bbox coordinates are normalized (0-1)
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                            bbox_issues.append((label_file, f"Line {line_idx+1}: Bounding box values not normalized: {x_center}, {y_center}, {width}, {height}"))
                            continue
                            
                        # Check for extremely small bounding boxes
                        if width < 0.01 or height < 0.01:
                            bbox_issues.append((label_file, f"Line {line_idx+1}: Extremely small bounding box: {width}, {height}"))
                            
                        # Check keypoints
                        if len(parts) >= 5 + (self.num_keypoints * 3):
                            keypoints = parts[5:]
                            
                            # Check groups of 3 values (x, y, visibility)
                            for i in range(0, len(keypoints), 3):
                                if i+2 >= len(keypoints):
                                    break
                                    
                                kp_x, kp_y, kp_v = map(float, keypoints[i:i+3])
                                
                                # Check if keypoint coordinates are normalized (0-1)
                                if not (0 <= kp_x <= 1 and 0 <= kp_y <= 1):
                                    keypoint_issues.append((label_file, f"Line {line_idx+1}, keypoint {i//3}: Coordinates not normalized: {kp_x}, {kp_y}"))
                                
                                # Check if visibility flag is valid (0, 1, or 2)
                                if kp_v not in [0, 1, 2]:
                                    keypoint_issues.append((label_file, f"Line {line_idx+1}, keypoint {i//3}: Invalid visibility flag: {kp_v}"))
                                    
                                # Check if all keypoints are marked as invisible (0)
                                if all(float(keypoints[i+2]) == 0 for i in range(0, len(keypoints), 3)):
                                    keypoint_issues.append((label_file, f"Line {line_idx+1}: All keypoints marked as invisible"))
                                    
                    except ValueError as e:
                        keypoint_issues.append((label_file, f"Line {line_idx+1}: Could not parse values - {str(e)}"))
                
            except Exception as e:
                keypoint_issues.append((label_file, str(e)))
                
        self.stats['bbox_issues'] = len(bbox_issues)
        self.stats['keypoint_issues'] = len(keypoint_issues)
        
        print(f"Found {len(bbox_issues)} files with bounding box issues")
        if bbox_issues:
            for file, issue in bbox_issues[:5]:
                print(f"  {file.name}: {issue}")
                self.stats['files_with_issues'].append(f"{file.stem} ({issue})")
                
        print(f"Found {len(keypoint_issues)} files with keypoint issues")
        if keypoint_issues:
            for file, issue in keypoint_issues[:5]:
                print(f"  {file.name}: {issue}")
                self.stats['files_with_issues'].append(f"{file.stem} ({issue})")
                
        return bbox_issues, keypoint_issues

    def visualize_samples(self, label_files, image_files):
        """Visualize random samples to check annotations visually."""
        if not self.visualization_dir:
            return
            
        # Create a dictionary for quick image lookup
        image_dict = {img.stem: img for img in image_files}
        
        # Filter to only labels that have corresponding images
        valid_labels = [lbl for lbl in label_files if lbl.stem in image_dict]
        
        if not valid_labels:
            print("No valid samples to visualize")
            return
            
        # Select random samples
        samples = random.sample(valid_labels, min(self.samples_to_visualize, len(valid_labels)))
        
        # COCO keypoint connections for visualization (if using COCO format)
        # This is for the standard 17-keypoint COCO format - modify as needed
        if self.num_keypoints == 17:
            connections = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Face
                (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
                (5, 6), (5, 11), (6, 12),  # Body
                (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
            ]
        else:
            connections = []  # No connections if not using COCO format
        
        colors = plt.cm.rainbow(np.linspace(0, 1, 10))  # Colors for different classes
        
        # Create visualizations
        for label_file in tqdm(samples, desc="Creating visualizations"):
            try:
                img_file = image_dict[label_file.stem]
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_height, img_width = img.shape[:2]
                
                # Create figure and axis
                fig, ax = plt.subplots(1, figsize=(12, 8))
                ax.imshow(img)
                
                # Parse label file
                content = label_file.read_text().strip()
                if not content:
                    ax.set_title(f"{label_file.stem} - Empty label file")
                    plt.savefig(self.visualization_dir / f"{label_file.stem}_viz.png")
                    plt.close(fig)
                    continue
                
                ax.set_title(f"{label_file.stem}")
                
                lines = content.split('\n')
                for line_idx, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    try:
                        # Get bounding box
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        
                        # Convert normalized coordinates to pixel coordinates
                        x1 = int((x_center - width/2) * img_width)
                        y1 = int((y_center - height/2) * img_height)
                        box_width = int(width * img_width)
                        box_height = int(height * img_height)
                        
                        # Draw bounding box
                        color = colors[class_id % len(colors)]
                        rect = patches.Rectangle((x1, y1), box_width, box_height, 
                                                linewidth=2, edgecolor=color, facecolor='none')
                        ax.add_patch(rect)
                        ax.text(x1, y1-5, f"Class {class_id}", color=color, fontsize=10, 
                                bbox=dict(facecolor='white', alpha=0.7))
                        
                        # Draw keypoints if available
                        if len(parts) >= 5 + (self.num_keypoints * 3):
                            keypoints = []
                            
                            for i in range(0, self.num_keypoints * 3, 3):
                                if 5 + i + 2 >= len(parts):
                                    break
                                
                                kp_x = float(parts[5 + i]) * img_width
                                kp_y = float(parts[5 + i + 1]) * img_height
                                kp_v = int(float(parts[5 + i + 2]))
                                
                                keypoints.append((kp_x, kp_y, kp_v))
                                
                                # Draw keypoint based on visibility
                                if kp_v > 0:  # Visible or occluded
                                    marker_style = 'o' if kp_v == 2 else 'x'  # Circle for visible, x for occluded
                                    ax.plot(kp_x, kp_y, marker_style, markersize=8, 
                                          color=color, alpha=0.7 if kp_v == 2 else 0.4)
                            
                            # Draw connections between keypoints
                            for conn in connections:
                                p1, p2 = conn
                                if p1 < len(keypoints) and p2 < len(keypoints):
                                    if keypoints[p1][2] > 0 and keypoints[p2][2] > 0:  # Both keypoints are labeled
                                        ax.plot([keypoints[p1][0], keypoints[p2][0]], 
                                              [keypoints[p1][1], keypoints[p2][1]], 
                                              '-', linewidth=2, color=color, alpha=0.5)
                    
                    except Exception as e:
                        print(f"Error visualizing {label_file.name}, line {line_idx+1}: {str(e)}")
                
                plt.savefig(self.visualization_dir / f"{label_file.stem}_viz.png")
                plt.close(fig)
                
            except Exception as e:
                print(f"Error visualizing {label_file.name}: {str(e)}")

    def print_summary(self):
        """Print a summary of all findings."""
        print("\n" + "="*50)
        print("YOLO POSE DATASET ANALYSIS SUMMARY")
        print("="*50)
        print(f"Total images: {self.stats['total_images']}")
        print(f"Total label files: {self.stats['total_labels']}")
        print(f"Missing label files: {self.stats['missing_labels']}")
        print(f"Missing image files: {self.stats['missing_images']}")
        print(f"Empty label files: {self.stats['empty_labels']}")
        print(f"Files with format issues: {self.stats['format_issues']}")
        print(f"Files with bounding box issues: {self.stats['bbox_issues']}")
        print(f"Files with keypoint issues: {self.stats['keypoint_issues']}")
        
        print("\nClass distribution:")
        for class_id, count in self.stats['class_distribution'].items():
            print(f"  Class {class_id}: {count} instances")
            
        print("\nPossible reasons for 'background data' detection:")
        total_issues = (self.stats['missing_labels'] + self.stats['empty_labels'] + 
                       self.stats['format_issues'] + self.stats['bbox_issues'] + 
                       self.stats['keypoint_issues'])
                       
        if total_issues > 0:
            print("✓ Dataset contains issues that could cause the model to detect background data")
            
            if self.stats['missing_labels'] > 0:
                print("  - Some images don't have corresponding label files")
                
            if self.stats['empty_labels'] > 0:
                print("  - Some label files are empty")
                
            if self.stats['format_issues'] > 0:
                print("  - Some label files have incorrect format")
                
            if self.stats['bbox_issues'] > 0:
                print("  - Some bounding boxes have issues (not normalized, too small)")
                
            if self.stats['keypoint_issues'] > 0:
                print("  - Some keypoints have issues (not normalized, invalid visibility)")
                
            print("\nRecommendations:")
            print("1. Fix the identified issues in your dataset")
            print("2. Check your dataset.yaml file for correct paths and class mappings")
            print("3. Verify that your model configuration matches your keypoint format")
            
            if self.visualization_dir:
                print(f"4. Review the visualizations in {self.visualization_dir} for a better understanding")
        else:
            print("✗ No obvious issues found in the dataset structure")
            print("\nOther possible causes:")
            print("1. Model configuration doesn't match dataset format")
            print("2. Incorrect class mapping in dataset.yaml")
            print("3. Training parameters may need adjustment")
            print("4. Dataset augmentation issues")
            
        print("="*50)

    def run_checks(self):
        """Run all checks on the dataset."""
        print(f"Analyzing YOLO pose dataset...")
        print(f"Images directory: {self.images_dir}")
        print(f"Labels directory: {self.labels_dir}")
        print(f"Expected keypoints per instance: {self.num_keypoints}")
        
        # Get all files
        image_files = self.get_image_files()
        label_files = self.get_label_files()
        
        self.stats['total_images'] = len(image_files)
        self.stats['total_labels'] = len(label_files)
        
        print(f"Found {len(image_files)} image files and {len(label_files)} label files")
        
        # Run checks
        self.check_missing_files(image_files, label_files)
        self.check_annotation_format(label_files)
        self.check_bbox_and_keypoints(label_files, image_files)
        
        # Visualize samples if requested
        if self.visualization_dir:
            self.visualize_samples(label_files, image_files)
            
        # Print summary
        self.print_summary()
        
        return self.stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Pose Dataset Checker")
    parser.add_argument("--images", required=True, help="Directory containing image files")
    parser.add_argument("--labels", required=True, help="Directory containing label files")
    parser.add_argument("--keypoints", type=int, default=17, help="Number of keypoints expected (default: 17 for COCO)")
    parser.add_argument("--visualize", help="Directory to save visualization images (optional)")
    parser.add_argument("--samples", type=int, default=5, help="Number of random samples to visualize (default: 5)")
    
    args = parser.parse_args()
    
    checker = YoloPoseChecker(
        images_dir=args.images,
        labels_dir=args.labels,
        num_keypoints=args.keypoints,
        visualization_dir=args.visualize,
        samples_to_visualize=args.samples
    )
    
    checker.run_checks()
    # Example usage in terminal:
    # python yolo_pose_checker.py --images /path/to/images --labels /path/to/labels --keypoints 17 --visualize /path/to/output/visualizations
