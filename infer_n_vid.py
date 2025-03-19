import os
import shutil
import tempfile
import cv2
from ultralytics import YOLO
from PIL import Image
import torch
import numpy as np
from pathlib import Path
import glob
import tempfile
import sys

class VideoProcessor:
    def __init__(self, model_path, video_path, output_tensor=None, output_video_name=None, image_folder=None, csv_file_path=None, fps=30.0):
        """
        Initialize VideoProcessor with model, paths, and fps.
        
        :param model_path: Path to the YOLO model.
        :param video_path: Path to the input video file.
        :param output_video_name: Name of the output video file.
        :param image_folder: Path to the folder where frame images will be stored.
        :param csv_file_path: Path to save the CSV file.
        :param fps: Frames per second for the output video.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.model = YOLO(model_path)  # Load the YOLO model
        self.video_path = video_path  # Set the input video path
        self.output_video_name = output_video_name  # Set the output video name
        self.generate_video = bool(output_video_name)  # Set self.generate_video based on whether output_video_name is provided or not
        self.image_folder = image_folder if image_folder else tempfile.mkdtemp()  # Set the image folder
        self.csv_file_path = csv_file_path  # Set the path to save the CSV file
        self.fps = fps  # Set the frames per second
        self.results = None  # Initialize results to None
        self.keep_frames = bool(image_folder)  # Set self.keep_frames based on whether image_folder is provided or not
        self.output_tensor = output_tensor
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # os.environ["QT_QPA_PLATFORM"] = "offscreen"

        # Future implementation - Determine the image folder path
        # if image_folder:
        #     self.image_folder = Path(image_folder)  # Convert to Path object for more robust path handling
        #     self.keep_frames = True
        # else:
        #     self.image_folder = Path(tempfile.mkdtemp())  # Create a temporary directory and convert to Path object
        #     self.keep_frames = False
        
        # Notify the user of the choices made by the program
        print(f"Image frames will be {'saved' if image_folder else 'discarded after use'}.")
        print(f"Video will be {'generated' if output_video_name else 'not generated'}.")
        print(f"Results will be {'saved to CSV' if csv_file_path else 'not saved to CSV'}.")


    def _process_result(self, result, index):
        """
            This method processes a single result: saves the frame and appends tensors to tensors_list.
            It handles the reshaping and concatenation of tensors to form the desired output structure.
            The reshaping is dynamic, adjusting to the number of keypoints detected.
        """
        # Save frame if required
        if self.generate_video or self.keep_frames:
            try:
                im = Image.fromarray(result.plot(kpt_radius=3)[...,::-1])
                image_path = os.path.join(self.image_folder, f"{index}.jpg")
                im.save(image_path)
            except IOError as e:  # Catching a more specific error related to I/O operations
                print(f"IOError occurred while saving frame {index}: {e}")
            except Exception as e:  # Catching any other unexpected errors that might occur
                print(f"An unexpected error occurred while saving frame {index}: {e}")

        if self.csv_file_path:
            # tensors_to_concat = torch.full((1,1+result.boxes.shape[-1]+result.keypoints.shape[-1]),-1) # initialize a tensor with -1 to fill in case of an error
            # size = index + boxes (box class-> box id) + keypoints
            # tensors_to_concat[0,0] = index
            # Another direction this code could go is initializing the tensor before try-except, then re-initializing via boxes_xywh.shape[0] to account for situations where there are more than one entry in a frame
            try:
                # List to hold tensors to concatenate
                # if result.boxes.xywh.nelement() != 0: # Check if there are no boxes detected
                # if result.boxes.id is not None: # Check if there are no boxes detected
                tensors_to_concat = []
                    
                if result.boxes is not None and result.boxes.xywh.nelement() != 0: # Check if there are no boxes detected    
                    # Boxes tensors
                    boxes_xywh = result.boxes.xywhn.to(self.device)
                    boxes_conf = result.boxes.conf.unsqueeze(-1).to(self.device)  # Adding an extra dimension to align with other tensors
                    # boxes_conf = boxes_conf.reshape(boxes_xywh.shape[0],-1)
                    # boxes_cls = result.boxes.cls.unsqueeze(-1).to(self.device)  # Adding an extra dimension to align with other tensors
                    if result.boxes.id is not None:
                        trackID = result.boxes.id.unsqueeze(-1).to(self.device)
                    else:
                        trackID = torch.full((boxes_xywh.shape[0], 1), -1, dtype=boxes_xywh.dtype, device=boxes_xywh.device)
                    
                    # Create an index tensor
                    index_tensor = torch.full((boxes_xywh.shape[0], 1), index, dtype=boxes_xywh.dtype, device=boxes_xywh.device) # this doubles as number of entity detected in the frame
                                
                    # Append index and box tensors to tensors_to_concat #TODO: Check if this is the correct way to concatenate tensors
                    tensors_to_concat.extend([index_tensor, trackID, boxes_xywh, boxes_conf])
                else:
                    # Create an index tensor
                    index_tensor = torch.full((1, 1), index, dtype=torch.float32, device=self.device) # this doubles as number of entity detected in the frame
                    # If no boxes are detected, append a tensor of -1 to maintain the structure
                    tensors_to_concat.extend([index_tensor, torch.full((1, 1), -1, dtype=torch.float32, device=self.device), torch.full((1, 4), -1, dtype=torch.float32, device=self.device), torch.full((1, 1), -1, dtype=torch.float32, device=self.device)])
                    
                # Check if keypoints exist, and if so, append them to tensors_to_concat 
                # if result.keypoints is not None:
                # if result.keypoints.conf is not None: 
                if result.keypoints is not None and result.keypoints.xyn.nelement() != 0: # Check if there are no keypoints detected TODO: Check if this is the correct way to check for keypoints
                    # Key points tensors
                    keypoints_xy = result.keypoints.xyn.to(self.device)
                    keypoints_conf = result.keypoints.conf.unsqueeze(-1)  # Add a new axis to make it [1, 11, 1]
                    
                    # Assuming each keypoint has two coordinates (x, y)
                    num_keypoints = keypoints_xy.shape[-2]  # Assuming keypoints_xy has a shape [..., num_keypoints, 2]
                    tensors_to_concat.append(keypoints_xy.reshape(-1, num_keypoints * 2))
                    tensors_to_concat.append(keypoints_conf.reshape(-1, num_keypoints))
                elif result.boxes.xywh.nelement() != 0: #TODO: this line has been going in and out. check for working and not-working cases. working case: custom mouse keypoints detection-YN
                    # If no keypoints are detected, append a tensor of -1 to maintain the structure
                    tensors_to_concat.append(torch.full((boxes_xywh.shape[0], 33), -1, dtype=torch.float32, device=self.device))
                else:    
                    tensors_to_concat.append(torch.full((1, 33), -1, dtype=torch.float32, device=self.device))
                
                # Concatenate all available tensors along the last axis
                concatenated = torch.cat(tensors_to_concat, dim=-1)
                
                # Append the concatenated tensor to the list
                self.tensors_list.append(concatenated)
            except AttributeError as e:
                print(f"AttributeError at result {index}: {e}")
        
    def process_video(self):
        """
        Process the input video, save frames, create output video if required, save results to CSV, and clean up.
        """
        if self.keep_frames:
            os.makedirs(self.image_folder, exist_ok=True)

        if self.csv_file_path:
            os.makedirs(os.path.dirname(self.csv_file_path), exist_ok=True)
            
        if self.output_tensor:
            os.makedirs(os.path.dirname(self.output_tensor), exist_ok=True)
        
        try:
            if self.keep_frames and os.path.exists(self.image_folder):
                print(f"Folder {self.image_folder} already exists. Proceeding with overwrite.")
                
            # Perform prediction only once and store the results in an instance variable
            # Redirect standard output
            # sys.stdout = open(os.devnull, 'w')
            # self.results = self.model.track(self.video_path, stream=True, device="cuda:0", persist=True, max_det=1, imgsz=1920, retina_masks=True) # for Custom videos
            self.results = self.model.track(self.video_path, stream=True, device="cuda:0", persist=True, max_det=1, imgsz=1920, augment=True, retina_masks=True) # for cricket detection
            # self.results = self.model.track(self.video_path,  stream=True, device="cuda:0", persist=True, max_det=5,conf=0.2,retina_masks=True,imgsz=1024) # For AVATAR videos
            # self.results = self.model.track(self.video_path, stream=True, device="cuda:0", persist=True, iou=0.6, conf=0.2,retina_masks=True,imgsz=2048) # For AVATAR social
            # self.results = self.model.track(self.video_path, stream=True, device="cuda:0", persist=True, iou=0.6, conf=0.2,retina_masks=True,imgsz=1280) # For MONKEY videos
            # sys.stdout = sys.__stdout__
            
            self.tensors_list = []
            for i, result in enumerate(self.results):
                self._process_result(result, i)
                
            if self.tensors_list:
                self._save_csv()
            
            if self.output_tensor:
                self._save_tensor()
            
            if self.generate_video:
                self._create_video_from_images()
                
            self._clean_up()
            
        except Exception as e:
            print(f"An error occurred while processing the video: {e}") 
    
    def process_image(self):
        """
        Process images in the directory specified by self.video_path (self.input_path),
        running inference on each and saving the results.
        """
        try:
            # Patterns for common image formats
            image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.tiff']
            
            # List all image files in the specified directory for each pattern
            image_paths = []
            for pattern in image_patterns:
                image_paths.extend(glob.glob(os.path.join(self.video_path, pattern)))  # Use self.input_path if renamed

            # Ensure the output directory exists
            os.makedirs(self.image_folder, exist_ok=True)

            # # Run batch inference
            # results = self.model.predict(image_paths,stream=True,device="cuda:0")
            # for i, result in enumerate(self.results):
            #     self._process_result(result, i)
            
            # for i,result in enumerate(results):
            for i in range(len(image_paths)):
                result = self.model.predict(image_paths[i],device="cuda:0")

            # Save the inferred image
                base_name = os.path.basename(image_paths[i])
                output_path = os.path.join(self.image_folder, f"inferred_{base_name}")

            # Extract inference result image (modify based on how your YOLO model returns results)
                im = Image.fromarray(result[0].plot(kpt_radius=3)[...,::-1])
                im.save(output_path)

                # result[0].save(filename=output_path)  # Adjust this line according to how your model's results are structured

            print(f"Processed and saved: {output_path}")

        except Exception as e:  
            print(f"An error occurred while processing image: {e}")

    def _save_csv(self):
        """
        Save the processed tensors to a CSV file.
        """
        try:
            if self.tensors_list:
                final_tensor = torch.cat(self.tensors_list, dim=0)
                np.savetxt(self.csv_file_path, final_tensor.cpu().numpy().reshape(-1, final_tensor.shape[-1]), fmt='%.4f', delimiter=',')
                print(f"Successfully saved results to {self.csv_file_path}")
            else:
                print("No tensors to save to CSV.")
        except Exception as e:
            print(f"An error occurred while saving results to CSV: {e}")
    
    def _save_tensor(self):
        """
        Save the processed tensors as it is
        """
        try:
            if self.tensors_list:
                final_tensor = torch.cat(self.tensors_list, dim=0)
                torch.save(final_tensor, self.output_tensor)
                print(f"Successfully saved results to {self.output_tensor}")
            else:
                print("No tensors to save to tensor.")
        except Exception as e:
            print(f"An error occurred while saving results to tensor: {e}")
                    
    def _create_video_from_images(self):
        """
        Create a video from the saved frame images.
        """
        # Get the list of image files from the image folder
        images = [img for img in os.listdir(self.image_folder) if img.endswith(".png") or img.endswith(".jpg")]
        images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        if not images:
            print("No images found in the specified directory!")
            return

        first_image_path = os.path.join(self.image_folder, images[0])
        first_image = cv2.imread(first_image_path)
        if first_image is None:
            print(f"Error reading image {first_image_path}")
            return

        height, width, layers = first_image.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video_name, fourcc, self.fps, (width, height))

        print(f"Creating video {self.output_video_name}, with {len(images)} frames.")
        for i, image_name in enumerate(images):
            image_path = os.path.join(self.image_folder, image_name)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Error reading image {image_path}. Skipping this frame.")
                continue

            out.write(image)
            if i % 100 == 0:
                print(f"Processed {i}/{len(images)} frames.")

        out.release()
        print("Video creation is complete.")

    def _clean_up(self):
        """
        Clean up the temporary image folder if keep_frames is False.
        """
        if not self.keep_frames:
            if os.path.exists(self.image_folder):
                try:
                    shutil.rmtree(self.image_folder)
                    print(f"Successfully deleted the temporary image folder: {self.image_folder}")
                except Exception as e:
                    print(f"Error occurred while trying to delete the temporary image folder: {e}")
            else:
                print(f"No folder found at {self.image_folder}. Nothing to delete.")
             
# Usage 
if __name__ == "__main__":
    # model_path = '/home/tarislada/YOLOprojects/YOLO_custom/Models/KH_binocular/KH_NoseNtail_sv1_2b.pt'
    # model_path = '/home/tarislada/YOLOprojects/YOLO_custom/Models/KH_bot_ir/KH_bot_sv3_8/weights/best.pt' # testrun on sv3 based additional trained model.
    # model_path = 'YOLO_custom/Models/YW/YW_v01.pt'
    # model_path = 'YOLO_custom/Models/KH_binocular/KH_noseperfect_s_v1.pt'
    # model_path = 'YOLO_custom/Models/Real_3D_AVATAR/Med_v11__hhres_bot_addv02_01.pt'
    # model_path = 'YOLO_custom/Models/Real_3D_AVATAR/Med_v11__hhres_bot_addv02_all1.pt'
    # model_path = 'YOLO_custom/Models/KH_bot_ir/KH_bot_sv3_7.pt'
    # model_path = 'runs/detect/train16/weights/best.pt'
    model_path = 'Models/Cricket_detection/Cricket_v2s.pt'
    # model_path = 'YOLO_custom/Models/Monkey_data/train23/weights/best.pt'
    # fps = 60.0
    fps = 30.0
    # Single file level usage
    
    # # video_path = '/home/tarislada/YOLOprojects/KH_frameextract2.mp4'#
    # video_path = 'YOLO_custom/Video_Monkey/2019-01-09_15-04-09 -Max & Louie.mp4'
    # video_path = 'YOLO_custom/AVATAR_NEW_VID/20240818_#202.mp4'
    # video_path = 'YOLO_custom/YN/MVI_2859.MP4'
    # video_path = 'YOLO_custom/Video_YW/OFT-Dark/a53t_ASO91/#14_MVI_1051.MP4'
    # # video_path = 'YOLO_custom/Video_KH/m2_40Hz.mp4'
    # # video_path = 'YOLO_custom/Video_KH/Photometory_CB/batch2/MVI_1674.mp4'
    # video_path = '/home/tarislada/Documents/Extra_python_projects/SKH FP/video_file/m17_t1.mp4'
    # video_path = '/mnt/disk3/Cricket_hunt/raw/m20_t2.MP4'
    video_path = '/home/tarislada/YOLOprojects/YOLO_custom/KH/KH_binocular_set6/representative.mp4'
    
    # image_folder = 'YOLO_custom/KH/KH_binocular_set5/images/m24_2_t7_KH_NoseNtail_sv1_2b' # Provide a path if you want to keep the frame images
    
    # # output_video_name = '/home/tarislada/YOLOprojects/YOLO_custom/Result_vid/KH_bot_test/MVI_1674_cricket_train30.mp4' # Provide a name if you want to generate a video file
    # output_video_name = '/home/tarislada/YOLOprojects/YOLO_custom/Result_vid/Monkey_2019-01-09_15-04-09.mp4'
    # output_video_name = 'YOLO_custom/Result_vid/YW/OFT-Dark_a53t_ASO91_#14_MVI_1051_test01.mp4'
    # output_video_name = '/home/tarislada/Documents/Extra_python_projects/SKH FP/m20_t2_cricket_v11s_06.mp4'
    # # output_video_name = 'YOLO_custom/Hannah_sample/Results/FM3_7d_session1_output.mp4'
    # # output_video_name = '/home/tarislada/YOLOprojects/YOLO_custom/Result_vid/m2_40hz_KH_NoseNtail_0625_test71b.mp4'
    # output_video_name = 'YOLO_custom/Result_vid/Allones_20240818_#202.mp4'
    # output_video_name = 'YOLO_custom/Result_vid/YN_MVI_2859.mp4'
    # output_video_name = '/home/tarislada/Documents/Extra_python_projects/SKH FP/m17_t1_cricket_v11s_04.mp4'
    # output_video_name = '/home/tarislada/YOLOprojects/YOLO_custom/Result_vid/m20_t2_natrez_v11s04_nat_aug.mp4'
    # output_video_name = 'YOLO_custom/Result_vid/m24_2_t7_KH_NoseNtail_sv1_2b.mp4'
    # output_video_name = 'YOLO_custom/Result_vid/m17_t1_KH_NoseNtail_sv1_2b.mp4'
    output_video_name = '/home/tarislada/YOLOprojects/YOLO_custom/Result_vid/KH_set6_m33_t2_crickettest_v11s04.mp4'
    
    # csv_file_path = '/home/tarislada/YOLOprojects/YOLO_custom/csv/MVI_0624_KH_bot_sv3_1b_original.csv'  # Provide a path if you want to save results to CSV
    # csv_file_path = '/home/tarislada/Documents/Extra_python_projects/SKH FP/m18_t7_v3_raw.csv'
    # csv_file_path = 'YOLO_custom/Hannah_sample/Results/FM3_7d_session1_output.csv''
    # csv_file_path = 'YOLO_custom/KH/KH_binocular_set5/m24_2_t7_KH_NoseNtail_sv1_2b.csv'
       
    # processor = VideoProcessor(model_path=model_path, video_path=video_path, output_video_name=output_video_name, fps=fps, csv_file_path=csv_file_path)
    processor = VideoProcessor(model_path=model_path, video_path=video_path, output_video_name=output_video_name, fps=fps)
    processor.process_video()
    # processor = VideoProcessor(model_path=model_path, video_path=video_path, image_folder=image_folder, fps=fps, csv_file_path=csv_file_path)
    # processor.process_video()
    # processor.process_image()
    
    # Multiple file level usage
    # video_directory = '/home/tarislada/Documents/Extra_python_projects/SKH FP/video_file'
    # video_directory = '/mnt/disk3/Video_KH/250108_video_set5_m24_2_m31'
    # # video_directory = 'YOLO_custom/YN/v2'
    
    # # # List all .mp4 files in the specified directory
    # video_extensions = ['*.mp4', '*.MP4', '*.avi']
    # video_paths = []
    # for extension in video_extensions:
    #     video_paths.extend(glob.glob(os.path.join(video_directory, extension)))

    # # # video_paths = glob.glob(os.path.join(video_directory, '*.mp4'))

    # # Define the model path and other parameters

    # # Iterate over all video files and process them
    # for video_path in video_paths:
    #     if not os.path.exists(video_path):
    #         print(f"Video file not found: {video_path}")
    #         continue
        
    #     # Extract the video file name (without extension) to use it as a base name for output files
    #     base_name = os.path.basename(video_path)
    #     name_without_extension = os.path.splitext(base_name)[0]
        
    #     # Define unique output paths for each video
    #     # output_video_name = f"/home/tarislada/YOLOprojects/YOLO_custom/Result_vid/{name_without_extension}_output.mp4"
    #     output_video_name = f"YOLO_custom/KH/KH_binocular_set5/{name_without_extension}_output.mp4"
    #     image_folder = f"YOLO_custom/KH/KH_binocular_set5/{name_without_extension}"
    #     csv_file_path = f"YOLO_custom/KH/KH_binocular_set5/{name_without_extension}.csv"
    #     # csv_file_path = f'YOLO_custom/YN/csv/{name_without_extension}.csv'
        
    #     # Create the processor object and start processing the video
    #     # processor = VideoProcessor(model_path=model_path, video_path=video_path, csv_file_path=csv_file_path, output_video_name = output_video_name, fps=fps)
    #     processor = VideoProcessor(model_path=model_path, video_path=video_path, csv_file_path=csv_file_path, fps=fps, output_video_name=output_video_name)
    #     processor.process_video()