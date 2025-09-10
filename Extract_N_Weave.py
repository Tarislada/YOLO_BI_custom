import os
import cv2
import collections

def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    stride: int = 1,
    prefix: str = "frame"
):
    """
    Extracts frames from a given video and saves them as individual image files.
    
    :param video_path: Path to the input video file.
    :param output_dir: Directory where extracted frames will be saved.
    :param stride: Save every 'stride'-th frame. Defaults to 1 (save all frames).
    :param prefix: Prefix for the output image filenames.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a video capture object
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # No more frames or error
        
        # Save every 'stride'-th frame
        if frame_count % stride == 0:
            # Construct output filename
            output_filename = f"{prefix}_{frame_count:06d}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            # Write the frame as an image
            cv2.imwrite(output_path, frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Saved {saved_count} frames from '{video_path}' to '{output_dir}'.")


def extract_frames_from_directory(
    input_videos_dir: str,
    output_dir: str,
    stride: int = 1,
    prefix: str = "frame"
):
    """
    Extracts frames from all videos in the specified directory.
    
    :param input_videos_dir: Directory containing the input video files.
    :param output_dir: Directory where extracted frames will be saved (one subfolder per video).
    :param stride: Save every 'stride'-th frame. Defaults to 1 (save all).
    :param prefix: Prefix for the output image filenames.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Loop through each file in the input directory
    for file_name in os.listdir(input_videos_dir):
        if file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(input_videos_dir, file_name)
            
            # Create a subfolder named after the video (without extension)
            base_name = os.path.splitext(file_name)[0]
            video_output_dir = os.path.join(output_dir, base_name)
            
            extract_frames_from_video(
                video_path=video_path,
                output_dir=video_output_dir,
                stride=stride,
                prefix=prefix
            )


def images_to_video(input_dir: str,
                    output_video_path: str,
                    fps: int = 30,
                    resize_mode: str = "resize"):  # or "pad"
    """
    Creates a video from a sequence of images in a directory.
    Ensures consistent frame size; auto-resizes/pads if needed.
    """
    # Collect + natural sort so frame_2 comes before frame_10? actually after; natsort fixes numeric order.
    images = [f for f in os.listdir(input_dir) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    # images = images.sort()

    print(f"Found {len(images)} image files in {input_dir}")
    if not images:
        print("No images. Aborting.")
        return

    # Read first valid image
    first_frame = None
    for name in images:
        path = os.path.join(input_dir, name)
        im = cv2.imread(path, cv2.IMREAD_COLOR)  # force 3-channel
        if im is not None:
            first_frame = im
            break
    if first_frame is None:
        print("Could not read any image.")
        return

    h, w = first_frame.shape[:2]
    print(f"Base frame size: {w}x{h}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'mp4v' if you want mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    if not out.isOpened():
        print("Error: VideoWriter failed to initialize.")
        return

    written = 0
    mismatched = 0

    for name in images:
        path = os.path.join(input_dir, name)
        frame = cv2.imread(path, cv2.IMREAD_COLOR)  # ensures 3 channels
        if frame is None:
            print(f"Skipped unreadable image: {name}")
            continue

        fh, fw = frame.shape[:2]
        if (fh, fw) != (h, w):
            mismatched += 1
            if resize_mode == "resize":
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            elif resize_mode == "pad":
                canvas = cv2.copyMakeBorder(
                    frame, 0, max(0, h - fh), 0, max(0, w - fw),
                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )
                frame = canvas[:h, :w]
            else:
                print(f"Frame {name} size mismatch ({fw}x{fh}); skipping.")
                continue

        out.write(frame)
        written += 1

    out.release()
    print(f"Video saved to: {output_video_path}")
    print(f"Frames written: {written}  (mismatched resized: {mismatched})")

    
def inspect_image_shapes(input_dir):
    exts = ('.jpg', '.jpeg', '.png')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(exts)]
    files.sort()
    shapes = collections.Counter()
    bad = []
    for f in files:
        img = cv2.imread(os.path.join(input_dir, f))
        if img is None:
            bad.append(f)
            continue
        shapes[img.shape] += 1
    print("Total image files:", len(files))
    print("Unreadable:", len(bad))
    for shp, cnt in shapes.items():
        print(f"{cnt:4d}  -> shape={shp}")
    return shapes, bad


if __name__ == "__main__":
    # Example usage:
    
    # 1) Extract frames from multiple videos in 'videos_input' directory
    #    into 'extracted_frames' directory, using a stride of 5
    input_videos_dir = '/media/tarislada/SAMSUNG/Doric임시/855_0305'
    # input_videos_dir = '/home/tarislada/YOLOprojects/YOLO_custom/Hannah_sample'
    frames_output_dir = '/home/tarislada/YOLOprojects/YOLO_custom/KH/Doric_temp'
    # frames_output_dir = '/home/tarislada/YOLOprojects/YOLO_custom/Dataset/Walltask'
    stride_value = 15
    
    # extract_frames_from_directory(
    #     input_videos_dir, 
    #     frames_output_dir, 
    #     stride=stride_value, 
    #     prefix="frame"
    # )
    
    # 2) Combine all images in 'some_image_dir' into a video
    # images_dir = '/home/tarislada/YOLOprojects/YOLO_custom/Dataset/Walltask/represetative'
    # images_dir = '/home/tarislada/YOLOprojects/YOLO_custom/KH/KH_binocular_set6/extracted_frames/m33_t2'
    # output_video = "/home/tarislada/YOLOprojects/YOLO_custom/KH/KH_binocular_set6/representative.mp4"
    images_dir = '/home/tarislada/YOLOprojects/YOLO_custom/Dataset/KH/YOLO_format/Bot_IR_Hunting/KH_bot_IR_v4/images/val'
    output_video = "/home/tarislada/YOLOprojects/YOLO_custom/KH/KH_bot_IR_v4_val.mp4"

    # inspect_image_shapes('/home/tarislada/YOLOprojects/YOLO_custom/Dataset/KH/YOLO_format/Bot_IR_Hunting/KH_bot_IR_v4/images/val')
    images_to_video(images_dir, output_video, fps=30, resize_mode='resize')