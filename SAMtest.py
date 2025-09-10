import cv2
import numpy as np
import ultralytics
from PIL import Image
import os
import matplotlib.pyplot as plt

def save_reversed_segmentation(result, output_path):
    """
    Save segmented image with reversed mask (background visible, object blacked out)
    
    Args:
        result: YOLO segmentation result
        output_path: Path to save the processed image
    """
    # Get the original image
    original_img = result.orig_img.copy()
    
    # Create a mask for all detected objects
    if result.masks is not None:
        # Get image dimensions
        h, w = original_img.shape[:2]
        
        # Initialize combined mask
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Combine all segmentation masks
        for mask in result.masks.data:
            # Resize mask to original image size
            mask_resized = cv2.resize(mask.cpu().numpy(), (w, h))
            # Convert to binary mask
            binary_mask = (mask_resized > 0.5).astype(np.uint8)
            # Add to combined mask
            combined_mask = np.logical_or(combined_mask, binary_mask).astype(np.uint8)
        
        # Create the final image
        result_img = original_img.copy()
        
        # Black out the segmented objects (where mask is 1)
        result_img[combined_mask == 1] = [0, 0, 0]  # Set to black
        
        # Save the result
        cv2.imwrite(output_path, result_img)
        print(f"Reversed segmentation saved to: {output_path}")
    else:
        print("No segmentation masks found in the result")


def calculate_circularity(segment):
    area = cv2.contourArea(segment)
    perimeter = cv2.arcLength(segment, True)
    if perimeter == 0:
        return 0
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    return circularity, area

# Function to check if a segment is moving
def is_moving(segment, previous_segments):
    if previous_segments is None:
        return True
    segment_center = np.mean(segment, axis=0)
    for prev_segment in previous_segments:
        prev_segment_center = np.mean(prev_segment, axis=0)
        distance = np.linalg.norm(segment_center - prev_segment_center)
        if distance < motion_threshold:
            return True
    return False

# Function to calculate color histogram
def calculate_color_histogram(segment, frame):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [segment], 1)
    
    hist = cv2.calcHist([frame], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist

# Function to compare histograms
def compare_histograms(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


##### Initialization #####
# Load Yolo model
# model = ultralytics.YOLO('yolov9c-seg')
# model = ultralytics.FastSAM('FastSAM-s.pt')
model = ultralytics.FastSAM('/home/tarislada/YOLOprojects/YOLO_custom/Models/Nat_segment.pt')

# Video path
# video_path = 'YOLO_custom/preTST_CS1_M2.mp4'
video_path = '/home/tarislada/YOLOprojects/YOLO_custom/preTST_CS1_M3.mp4'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    print("Error: Video not found")

# Initialize previous segments for motion tracking
previous_segments = None
motion_threshold = 50  # Define an appropriate threshold for motion

# FASTSAM results
results = model.track(source=video_path, imgsz=640, stream=True, device="cuda:0", persist=True,conf=0.5,max_det=3)
# results = model.track(source=video_path, imgsz=640, stream=True, device="cuda:0", persist=True,conf=0.5,max_det=3,verbose=False)
enclosed_pixels = []
image_folder = 'YOLO_custom/FASTSAMtest'
csv_save_path = 'YOLO_custom/enclosed_pixels.npy'
os.makedirs(image_folder, exist_ok=True)
mouse_color_histogram = None
selected_segment_index = 0
mouse_selected = False


for index, result in enumerate(results):
    # Save the frame
    segments = result.masks.xy
    im = Image.fromarray(result.plot())
    image_path = os.path.join(image_folder, f"{index}.jpg")
    im.save(image_path)
    
    frame = np.array(im)
    valid_mouse_segment_found = False
    
    if index == 0:
        output_filename = f"reversed_segmentation.jpg"
        save_reversed_segmentation(result, output_filename)

    if index == 0 or not mouse_selected:
        while not mouse_selected:
            # Display the frame with segments using matplotlib
            plt.imshow(result.orig_img)
            print("Segments detected:")
            for i, segment in enumerate(segments):
                plt.plot(np.array(segment)[:, 0], np.array(segment)[:, 1], label=f'Segment {i}')
            plt.legend()
            plt.show()

            # Take user input for selecting the segment
            selected_segment_index = input("Enter the index of the mouse segment (or 'skip' to skip this frame): ")
            if selected_segment_index.lower() == 'skip':
                break
            selected_segment_index = int(selected_segment_index)
            selected_segment = segments[selected_segment_index]

            # Calculate color histogram
            segments_array = np.array(selected_segment, np.int32).reshape((-1, 1, 2))
            mouse_color_histogram = calculate_color_histogram(segments_array, frame)
            mouse_selected = True

        if not mouse_selected:
            print(f"Skipping frame {index} as no valid mouse segment was found.")
            continue

    for segment in segments:
        # Convert segments to a suitable format for cv2.fillPoly
        segments_array = np.array(segment, np.int32)
        segments_array = segments_array.reshape((-1, 1, 2))

        # Create an empty mask of the same dimensions as your frame
        height, width = result.orig_shape
        mask = np.zeros((height, width), dtype=np.uint8)

        # Draw the polygon defined by 'segments' on the mask
        cv2.fillPoly(mask, [segments_array], 1)

        # Calculate circularity
        circularity, area = calculate_circularity(segments_array)

        # Calculate color histogram
        frame = np.array(im)
        color_histogram = calculate_color_histogram(segments_array, frame)
        color_similarity = compare_histograms(mouse_color_histogram, color_histogram)

        # Check if the segment is likely a mouse based on circularity, motion, and color
        # if 0.2 <= circularity <= 1.0 and color_similarity > 0.7 and is_moving(segments_array, previous_segments):
        if 0.2 <= circularity <= 1.0 and color_similarity > 0.7 and 50000 <= area <= 100000:
            # Count the pixels
            enclosed_pixels.append(np.sum(mask))

    # Update previous segments for motion tracking
    # previous_segments = segments

np.save(csv_save_path, np.array(enclosed_pixels))
plt.figure()
plt.plot(enclosed_pixels, label='Enclosed Pixels')
plt.xlabel('Frame Index')
plt.ylabel('Pixel Area')
plt.title('Enclosed Pixel Area Over Time')
plt.legend()
plt.show()

cv2.destroyAllWindows()
