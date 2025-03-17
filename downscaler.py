import ffmpeg
import os
import glob

def get_target_bitrate(file_path, target_size_MB=700, duration_margin=0.98):
    """Estimate target bitrate to achieve the desired file size."""
    # Get video duration in seconds
    probe = ffmpeg.probe(file_path)
    duration = float(probe['streams'][0]['duration'])
    # Target size in bits, accounting for a margin to ensure we don't exceed the target size
    target_size_bits = target_size_MB * 8 * 1024 * 1024 * duration_margin
    # Calculate target bitrate
    target_bitrate = target_size_bits / duration
    return target_bitrate

def downscale_video(file_path, output_dir, target_size_MB=700):
    """Downscale video resolution and reduce file size."""
    base_name = os.path.basename(file_path)
    output_path = os.path.join(output_dir, base_name)
    target_bitrate = get_target_bitrate(file_path, target_size_MB)

    # Ensure bufsize does not exceed a reasonable value, e.g., double the capped bitrate
    bufsize = min(int(target_bitrate * 2), 10000)  # Cap bufsize at 10000 kbps

    # Command to downscale video and adjust bitrate
    ffmpeg.input(file_path).output(
        output_path, 
        vf='scale=-2:720',  # Downscale to 720p max, keeping aspect ratio
        b=str(int(target_bitrate))+'k',  # Set target bitrate
        maxrate=str(int(target_bitrate * 1.1))+'k',  # Max bitrate to allow some fluctuation
        bufsize=str(bufsize)+'k'  # Buffer size
    ).run()

# Specify your directories here
input_dir = 'YOLO_custom/Video_Monkey'
output_dir = 'YOLO_custom/Result_vid/Monkey/Test'

# Find all video files in the input directory and its subdirectories
video_files = glob.glob(input_dir + '/**/*.mp4', recursive=True)  # Adjust the extension if necessary

# Process each video file
for file_path in video_files:
    print(f"Processing {file_path}...")
    downscale_video(file_path, output_dir)
    print(f"Finished processing {file_path}. Output saved to {output_dir}")

print("All videos processed.")