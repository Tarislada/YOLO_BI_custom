import os
import shutil
import random
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



if __name__ == "__main__":
    # Example usage:
    folder_path = "/home/tarislada/YOLOprojects/YOLO_custom/Dataset/KH/YOLO_format/Bot_IR_Hunting/KH_bot_IR_v4/images/val"

    # 1) Rename images by adding a prefix
    prefix = "KH_set5_"
    # rename_with_prefix(folder_path, prefix)

    # 2) List all images in a txt file
    output_txt = "/home/tarislada/YOLOprojects/YOLO_custom/Dataset/KH/YOLO_format/Bot_IR_Hunting/KH_bot_IR_v4/val.txt"
    base_path_for_list = "./images/train"
    list_images_in_txt(folder_path, output_txt, base_path_for_list)
    
    # root_dir = "/path/to/dataset"
    # train_val_split(root_dir, val_ratio=0.2, seed=42)
