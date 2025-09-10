import os
import shutil
import random
import glob

# from PIL import Image

def rename_with_prefix(folder_path: str, prefix: str) -> None:
    """
    Renames all files in `folder_path` by adding `prefix` in front of the file name.
    """

    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif",".txt")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(valid_extensions):
            old_filepath = os.path.join(folder_path, filename)
            new_filename = prefix + filename
            new_filepath = os.path.join(folder_path, new_filename)
            os.rename(old_filepath, new_filepath)


def list_images_in_txt(
    folder_path: str,
    output_txt_path: str,
    base_path_for_list: str = ""
) -> None:
    """
    Creates/overwrites `output_txt_path` with a list of all image filenames in `folder_path`.
    Each line will be prefixed by `base_path_for_list` if provided.
    Arguments:
    - folder_path: Directory containing the images.
    - output_txt_path: Path to the output text file where image names will be listed.
    - base_path_for_list: Optional base path to prepend to each image filename in the list
    (e.g., "./images/train"). If empty, only filenames will be listed.
    """

    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif")

    with open(output_txt_path, "w") as txt_file:
        for filename in sorted(os.listdir(folder_path)):
            if filename.lower().endswith(valid_extensions):
                if base_path_for_list:
                    # If you want lines like: ./images/train/Kaist_BG_KH_bot_IR_v2/<filename>
                    line = os.path.join(base_path_for_list, filename)
                else:
                    line = filename
                txt_file.write(line + "\n")

def train_val_split(root_dir: str, val_ratio: float = 0.2, seed: int = 42) -> None:
    """
    Splits data in the 'images' and 'labels' subdirectories of `root_dir` into 
    train/val subfolders based on val_ratio. 
    Both images and labels share the same base filenames (extensions differ).
    
    Example structure before:
    root_dir/
      ├─ images/
      │    ├─ image1.jpg
      │    ├─ image2.jpg
      │    └─ ...
      └─ labels/
           ├─ image1.txt
           ├─ image2.txt
           └─ ...
    
    After running this, you'll get:
    root_dir/
      ├─ images/
      │    ├─ train/
      │    │    ├─ ...
      │    └─ val/
      │         ├─ ...
      └─ labels/
           ├─ train/
           │    ├─ ...
           └─ val/
                ├─ ...
    """
    random.seed(seed)

    # Directories
    images_dir = os.path.join(root_dir, "images")
    labels_dir = os.path.join(root_dir, "labels")

    # Create train/val subdirs if they don't exist
    train_images_dir = os.path.join(images_dir, "train")
    val_images_dir = os.path.join(images_dir, "val")
    train_labels_dir = os.path.join(labels_dir, "train")
    val_labels_dir = os.path.join(labels_dir, "val")

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # Gather all possible base filenames in images directory (ignoring the subfolders)
    all_image_files = [
        f for f in os.listdir(images_dir)
        if os.path.isfile(os.path.join(images_dir, f))
    ]
    
    # Remove subfolder items from the list, if any subfolders already existed
    # Typically, we only want the files that are directly inside images_dir
    # (not images_dir/train or images_dir/val).
    # (We assume the user is starting with all data in the top-level 'images' folder.)
    
    # Filter out anything that doesn't look like an image if desired, but
    # for demonstration, we simply use all files we found.

    # Shuffle file list so we can randomly split
    random.shuffle(all_image_files)

    # Compute how many items go to val
    val_count = int(len(all_image_files) * val_ratio)
    val_set = set(all_image_files[:val_count])
    train_set = set(all_image_files[val_count:])

    # Function to move one image file and its label
    def move_file_pair(image_file_name, destination="train"):
        # Move the image
        src_image_path = os.path.join(images_dir, image_file_name)
        dst_image_path = os.path.join(
            train_images_dir if destination == "train" else val_images_dir,
            image_file_name
        )
        shutil.move(src_image_path, dst_image_path)

        # Construct the corresponding label file name, by replacing extension
        # We assume the label has the same base name but .txt extension.
        # If your label extension differs, adjust here.
        base_name, _ = os.path.splitext(image_file_name)
        label_file_name = base_name + ".txt"
        src_label_path = os.path.join(labels_dir, label_file_name)
        if os.path.exists(src_label_path):
            dst_label_path = os.path.join(
                train_labels_dir if destination == "train" else val_labels_dir,
                label_file_name
            )
            shutil.move(src_label_path, dst_label_path)
        else:
            # If no label exists for a particular image, you can decide whether
            # to throw an error or just pass. For now, we'll just warn.
            print(f"Warning: No label file found for {image_file_name}")

    # Move the files
    for image_file in all_image_files:
        if image_file in val_set:
            move_file_pair(image_file, destination="val")
        else:
            move_file_pair(image_file, destination="train")

# def convert_jpg_to_png(jpg_path: str, png_path: str) -> None:
#     """
#     Converts a JPEG image to PNG format.
    
#     Parameters:
#     - jpg_path: The file path of the input .jpg image.
#     - png_path: The file path for the output .png image.
#     """
#     with Image.open(jpg_path) as img:
#         # Ensure image is converted to RGB if necessary
#         img = img.convert("RGB")
#         img.save(png_path, "PNG")

def fix_yolo_annotations(input_dir, output_dir=None):
    """
    Fix YOLO annotation files where center coordinates were incorrectly calculated
    by subtracting half width/height from the center coordinates.
    
    Args:
        input_dir: Directory containing the erroneous YOLO annotation files
        output_dir: Directory to save corrected files (if None, will overwrite original files)
    """
    if output_dir is None:
        output_dir = input_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Find all .txt annotation files
    annotation_files = glob.glob(os.path.join(input_dir, "*.txt"))
    print(f"Found {len(annotation_files)} annotation files to fix")
    
    for file_path in annotation_files:
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, filename)
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        fixed_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:  # Ensure we have at least class_id + bbox
                class_id = parts[0]
                
                # Parse bbox values
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Fix center coordinates by subtracting half width/height
                fixed_x_center = x_center - width/2
                fixed_y_center = y_center - height/2
                
                # Ensure coordinates stay within 0-1 range
                fixed_x_center = max(0, min(1, fixed_x_center))
                fixed_y_center = max(0, min(1, fixed_y_center))
                
                # Rebuild the line with fixed coordinates
                fixed_parts = [class_id, 
                              f"{fixed_x_center:.6f}", 
                              f"{fixed_y_center:.6f}", 
                              f"{width:.6f}", 
                              f"{height:.6f}"]
                
                # Add keypoints if they exist (parts 5 and beyond)
                if len(parts) > 5:
                    fixed_parts.extend(parts[5:])
                
                fixed_lines.append(" ".join(fixed_parts) + "\n")
        
        # Write the fixed annotations
        with open(output_path, 'w') as f:
            f.writelines(fixed_lines)
        
        print(f"Fixed annotations in {filename}")
    
    print(f"Successfully fixed {len(annotation_files)} annotation files")

def DG_BOX_separate_files(directory):
    """
    Separates .txt files and image files in the given directory.
    Creates 'labels' and 'images' subdirectories and moves files accordingly.
    
    Parameters:
        directory (str): The path to the directory containing the mixed files.
    """
    # Define subdirectories
    labels_dir = os.path.join(directory, 'labels')
    images_dir = os.path.join(directory, 'images')
    
    # Create subdirectories if they don't exist
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    # Define image file extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
    
    # Get all files in the directory
    files = glob.glob(os.path.join(directory, '*'))
    
    for file in files:
        if os.path.isfile(file):  # Ensure it's a file
            ext = os.path.splitext(file)[1].lower()  # Get file extension
            
            if ext == '.txt':
                shutil.move(file, os.path.join(labels_dir, os.path.basename(file)))
            elif ext in image_extensions:
                shutil.move(file, os.path.join(images_dir, os.path.basename(file)))
    
    print(f"Separation completed. Check '{labels_dir}' and '{images_dir}'.")

def box_clearobj(folder_path: str, class_threshold: int = 6) -> None:
    """
    Scans all .txt files in `folder_path` (YOLO label files),
    and removes any annotation lines where the class number > `class_threshold`.
    
    Example:
        If class_threshold = 6, any annotation line with class 7, 8, 9... is removed.
        
    Each label line is assumed to have the format:
        class x_center y_center width height
    """
    # Loop over each file in the folder
    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue
        
        file_path = os.path.join(folder_path, filename)
        
        # Read all lines from the label file
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        # Filter out lines where the class number is > class_threshold
        filtered_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            
            # The first element should be the class index
            class_idx_str = parts[0]
            try:
                class_idx = int(class_idx_str)
            except ValueError:
                # If it's not an integer, skip or handle differently if you like
                continue
            
            if class_idx <= class_threshold:
                filtered_lines.append(line)
        
        # Overwrite the file with the filtered lines
        with open(file_path, "w") as f:
            for line in filtered_lines:
                f.write(line)

def insert_dummy_keypoint(folder_path: str, keypoint_position: int) -> None:
    """
    Inserts a dummy keypoint (0, 0, 0) at the specified position in all YOLO annotation files in the folder.
    
    This function handles multiple instances of annotations per file (multiple objects per image).
    
    Parameters:
        folder_path (str): Directory containing YOLO annotation files (.txt)
        keypoint_position (int): Position to insert the dummy keypoint (0-indexed within keypoints)
                                For example, 0 would insert before the first keypoint,
                                3 would insert before the 4th keypoint, etc.
    """
    # Loop over each file in the folder
    modified_count = 0
    instances_modified = 0
    
    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue
        
        file_path = os.path.join(folder_path, filename)
        file_modified = False
        
        # Read all lines from the annotation file
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        modified_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts or len(parts) < 5:  # Skip invalid lines
                modified_lines.append(line)
                continue
            
            # First 5 values are: class_id, x, y, w, h
            bbox_parts = parts[:5]
            
            # All remaining values are keypoints in (x, y, visibility) triplets
            keypoint_parts = parts[5:]
            
            # Calculate the index to insert the dummy keypoint (0, 0, 0)
            # Each keypoint takes 3 values, so multiply position by 3
            insertion_index = keypoint_position * 3
            
            # Don't insert if it would be beyond the available keypoints
            if insertion_index <= len(keypoint_parts):
                # Insert dummy keypoint (0, 0, 0)
                keypoint_parts = (keypoint_parts[:insertion_index] + 
                                 ["0", "0", "0"] + 
                                 keypoint_parts[insertion_index:])
                
                # Combine everything back into a line
                modified_line = " ".join(bbox_parts + keypoint_parts) + "\n"
                modified_lines.append(modified_line)
                file_modified = True
                instances_modified += 1
            else:
                # If insertion point is beyond existing keypoints, leave as is
                modified_lines.append(line)
        
        # Write the modified content back to the file
        if file_modified:
            with open(file_path, "w") as f:
                f.writelines(modified_lines)
            modified_count += 1
    
    print(f"Successfully inserted dummy keypoint at position {keypoint_position} in {modified_count} files, modifying {instances_modified} annotation instances")


def reorder_keypoints(folder_path: str, new_order: list[int]) -> None:
    """
    Reorders YOLO keypoints in all annotation files within `folder_path` according to `new_order`.
    
    This function handles multiple instances of annotations per file (multiple objects per image).
    
    Each entry in `new_order` indicates where the original keypoint index should be placed.
    For example, if new_order = [0,1,3,2,4], 
    it means the 0th keypoint remains 0, the 1st remains 1, 2 goes to position 3, 3 goes to position 2, etc.
    
    Parameters:
        folder_path (str): Directory containing YOLO annotation files (.txt)
        new_order (list[int]): Desired ordering of the keypoints (0-based indices)
    """
    modified_count = 0
    instances_modified = 0
    
    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue
        
        file_path = os.path.join(folder_path, filename)
        file_modified = False
        
        # Read lines
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        modified_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:  # Skip invalid lines
                modified_lines.append(line)
                continue
            
            # First 5: class_id, x, y, w, h
            bbox_parts = parts[:5]
            keypoint_parts = parts[5:]
            
            # Group keypoint_parts into triplets
            keypoints = [keypoint_parts[i:i+3] for i in range(0, len(keypoint_parts), 3)]
            
            # Check if we can apply the reordering
            if len(keypoints) == len(new_order):
                # Rearrange keypoints according to new_order
                reordered = []
                for idx in new_order:
                    if 0 <= idx < len(keypoints):
                        reordered.append(keypoints[idx])
                    else:
                        # If index is out of range, add a placeholder
                        reordered.append(["0", "0", "0"])
                
                # Flatten it back to a list of strings
                keypoint_parts = [val for triplet in reordered for val in triplet]
                file_modified = True
                instances_modified += 1
            else:
                # If the number of keypoints doesn't match new_order length,
                # leave this instance as is
                pass
            
            # Combine everything back into a line
            modified_line = " ".join(bbox_parts + keypoint_parts) + "\n"
            modified_lines.append(modified_line)
        
        # Write the reordered annotations back
        if file_modified:
            with open(file_path, "w") as f:
                f.writelines(modified_lines)
            modified_count += 1
    
    print(f"Successfully reordered keypoints in {modified_count} files, modifying {instances_modified} annotation instances using new_order={new_order}")
    
def detect_num_instances(folder_path: str, target_instance: int) -> None:
    """
    Detects the number of annotation instances in all YOLO annotation files in the folder.
    
    Parameters:
        folder_path (str): Directory containing YOLO annotation files (.txt)
    """
    total_instance_errors = 0
    error_file_list = []
    
    # Loop over each file in the folder
    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue
        
        file_path = os.path.join(folder_path, filename)
        
        # Read all lines from the label file
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        # Count instances in this file
        instance_count = len(lines)
        if instance_count != target_instance:
            print(f"Warning: {filename} has {instance_count} instances, expected {target_instance}")
            error_file_list.append(filename)
            
        total_instance_errors += 1
    
    print(f"Total number of annotation files with erroneous instances: {total_instance_errors}")
    print(f"List of files with instance errors: {error_file_list}")

def set_keypoint_visibility(folder_path: str, keypoint_indices: list[int], new_visibility: int, output_dir: str | None = None, only_if_current_visibility_is: int | None = None) -> None:
    """
    Sets the visibility flag for specific keypoints in all YOLO annotation files in a folder.

    This function handles multiple annotation instances per file.

    Parameters:
        folder_path (str): Directory containing YOLO annotation files (.txt).
        keypoint_indices (list[int]): A list of 0-based indices of the keypoints to modify.
                                      For example, [0, 2] would target the 1st and 3rd keypoints.
        new_visibility (int): The new visibility value to set (e.g., 0, 1, or 2).
        output_dir (str | None, optional): Directory to save modified files. 
                                           If None, overwrites original files. Defaults to None.
        only_if_current_visibility_is (int | None, optional): Only change the visibility if the current
                                                              visibility flag matches this value. 
                                                              If None, the change is unconditional. Defaults to None.
    """
    if not isinstance(keypoint_indices, list):
        print("Error: keypoint_indices must be a list of integers.")
        return
    
    # Determine output directory and create if necessary
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output will be saved to: {output_dir}")
    else:
        # If no output_dir is provided, the output path will be the same as the input path.
        output_dir = folder_path
        print("Output will overwrite original files.")

    modified_files_count = 0
    modified_instances_count = 0

    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(folder_path, filename)
        output_path = os.path.join(output_dir, filename)
        file_was_modified = False
        
        with open(file_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            
            # A valid line must have at least a class, a box, and one keypoint triplet
            if len(parts) < 8:
                new_lines.append(line)
                continue

            keypoint_parts = parts[5:]
            num_keypoints = len(keypoint_parts) // 3
            line_was_modified = False

            for kp_index in keypoint_indices:
                # Check if the target keypoint is within the range of available keypoints for this instance
                if 0 <= kp_index < num_keypoints:
                    # The visibility flag is the 3rd value in each triplet (x, y, v)
                    # Its position in the flat list is (kp_index * 3) + 2
                    visibility_part_index = kp_index * 3 + 2
                    
                    # Check current visibility if a filter is provided
                    perform_update = False
                    if only_if_current_visibility_is is None:
                        perform_update = True
                    else:
                        try:
                            current_visibility = int(float(keypoint_parts[visibility_part_index]))

                            if current_visibility == only_if_current_visibility_is:
                                perform_update = True
                        except (ValueError, IndexError):
                            # If current visibility is not a valid int or index is bad, skip
                            print(f"Warning: Could not parse current visibility for keypoint {kp_index} in {filename}. Skipping.")
                            continue

                    if perform_update:
                        # Update the visibility flag
                        keypoint_parts[visibility_part_index] = str(new_visibility)
                        line_was_modified = True
                else:
                    print(f"Warning: In '{filename}', keypoint index {kp_index} is out of range for an instance with {num_keypoints} keypoints. Skipping.")

            if line_was_modified:
                # Reconstruct the line and add it to our list of new lines
                new_line = " ".join(parts[:5] + keypoint_parts) + "\n"
                new_lines.append(new_line)
                modified_instances_count += 1
                file_was_modified = True
            else:
                # If this line wasn't modified, add it back as is
                new_lines.append(line)

        # If any line in the file was changed, write to the determined output path
        if file_was_modified:
            with open(output_path, "w") as f:
                f.writelines(new_lines)
            modified_files_count += 1
            
    print(f"Operation complete. Modified {modified_instances_count} annotation instances across {modified_files_count} files.")

if __name__ == "__main__":
    # Example usage:
    folder_path = "/home/tarislada/YOLOprojects/YOLO_custom/Dataset/Real_3D_AVATAR_KH_r2/images/val"

    # 1) Rename images by adding a prefix
    # prefix = "tremorSUT_YW_v3_"
    # rename_with_prefix(folder_path, prefix)

    # 2) List all images in a txt file
    output_txt = "/home/tarislada/YOLOprojects/YOLO_custom/Dataset/Real_3D_AVATAR_KH_r2/images/val.txt"
    base_path_for_list = "./images/val"
    list_images_in_txt(folder_path, output_txt, base_path_for_list)
    
    # 3) Train/val split on manual_adjustment.py
    # root_dir = "/home/tarislada/YOLOprojects/YOLO_custom/Dataset/Nat/TST_250820"
    # train_val_split(root_dir, val_ratio=0.2, seed=42)

    # 4) Fixing YOLO box annotation from manual_adjustment.py
    # input_dir = "/home/tarislada/YOLOprojects/YOLO_custom/Dataset/KH/Cricket_v3/labels/tmp_train/Corner_frame_020141.txt"
    # output_dir = "/home/tarislada/YOLOprojects/YOLO_custom/Dataset/KH/Cricket_v3/labels/fixed_train/Corner_frame_020141.txt"  # Optional, remove to overwrite originals
    
    #TODO: Fixed annots have less # of floating points than original annots. need a mode that would turn keypoint annots into box only.
    # fix_yolo_annotations(input_dir, output_dir)

    # DG_BOX_separate_files('/home/tarislada/YOLOprojects/YOLO_custom/Dataset/AVATAR_box_img/img/Generic')
    # box_clearobj('/home/tarislada/YOLOprojects/YOLO_custom/Dataset/AVATAR_img/labels/train', class_threshold=6)
    # box_clearobj('/home/tarislada/YOLOprojects/YOLO_custom/Dataset/AVATAR_img/labels/val', class_threshold=6)

    # 5) Insert dummy keypoint at position 3 (which will be the 4th keypoint) # TODO: check system for 12 keypoints already existing case
    # labels_dir = "/home/tarislada/YOLOprojects/YOLO_custom/Dataset/Nat/KH_Top_aggregated0624/labels/val/Kaist_BG_KH"
    # insert_dummy_keypoint(labels_dir, keypoint_position=3)
    
    # # 6) Reorder keypoints
    # labels_dir = "/home/tarislada/YOLOprojects/YOLO_custom/Dataset/Nat/KH_Top_aggregated0624/labels/val/Kaist_BG_KH"
    # reorder_keypoints(labels_dir, new_order=[0,1,2,3,4,5,8,6,7,9,10,11])
    
    #7) Detect number of annotation instances
    # labels_dir = "/home/tarislada/YOLOprojects/YOLO_custom/Dataset/Real_3D_AVATAR_KH/labels/avatar_labels/val"
    # target_instance = 5
    # detect_num_instances(labels_dir, target_instance)
    
    # #8) Set keypoint visibility
    # labels_dir = "/home/tarislada/YOLOprojects/YOLO_custom/Dataset/Real_3D_AVATAR_KH_r2/labels/train_tobefixed"
    # output_dir = "/home/tarislada/YOLOprojects/YOLO_custom/Dataset/Real_3D_AVATAR_KH_r2/labels/train_fixed"
    # set_keypoint_visibility(labels_dir, keypoint_indices=[1, 2], new_visibility=0, only_if_current_visibility_is=1, output_dir=output_dir)
