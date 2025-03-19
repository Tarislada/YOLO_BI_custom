import os
import cv2

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


def images_to_video(
    input_dir: str,
    output_video_path: str,
    fps: int = 30
):
    """
    Creates a video from a sequence of images in a directory.
    
    :param input_dir: Directory containing image frames.
    :param output_video_path: Path to the output video file.
    :param fps: Frames per second for the output video.
    """
    # Get list of image files
    images = [f for f in os.listdir(input_dir) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Sort images by name so they go in the correct sequence
    images.sort()
    
    if not images:
        print(f"No images found in directory: {input_dir}")
        return
    
    # Read the first image to determine frame size
    first_image_path = os.path.join(input_dir, images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Error: Could not read image {first_image_path}")
        return
    
    height, width, channels = frame.shape
    
    # Define the codec and create the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change codec as needed
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Write each image to the video
    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        img_frame = cv2.imread(img_path)
        if img_frame is not None:
            out.write(img_frame)
    
    out.release()
    print(f"Video saved to: {output_video_path}")


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
    images_dir = '/home/tarislada/YOLOprojects/YOLO_custom/KH/KH_binocular_set6/extracted_frames/m33_t2'
    output_video = "/home/tarislada/YOLOprojects/YOLO_custom/KH/KH_binocular_set6/representative.mp4"
    images_to_video(images_dir, output_video, fps=30)