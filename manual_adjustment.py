import numpy as np
import sys
import cv2
import csv
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QCursor, QKeySequence
from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtWidgets import QMainWindow, QLabel, QLineEdit, QPushButton, QSlider, QFileDialog, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt, QPointF, pyqtSignal 
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsRectItem
from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import QColor, QPen, QBrush
from PyQt5.QtWidgets import QFileDialog

# It assumes that the track_id is always 1. If this isn't the case in your data, you'll need to modify the code to handle different track_ids.
# It sets the confidence score for the bounding box and all keypoints to 1.0. If you need to preserve the original confidence scores or calculate new ones, you'll need to modify this part.
# It preserves the order of frames by sorting the annotations dictionary keys.

# 3) box수정
# 1) delete (개별, all)
# 2) control+z
# 4) point 이름 float
# 5) 확대 축소
import os
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins/"
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:" + os.environ.get("LD_LIBRARY_PATH", "")


class AnnotationGraphicsView(QGraphicsView):
    keypoint_selected = pyqtSignal(int)
    keypoint_moved = pyqtSignal(int, float, float)
    keypoint_released = pyqtSignal(int, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setMouseTracking(True)
        self.keypoint_items = []
        self.final_box = None
        self.bbox_item = None
        self.selected_keypoint = None
        self.add_annotation_mode = False
        self.start_pos = None
        self.temp_box = None
        self.drawing_box = False
        self.placing_keypoints = False
        self.keypoint_count = 0
        self.new_annotation = None
        self.original_keypoint_positions = {}
        self.setScene(self.scene)
        self.keypoint_items = []
        self.drag_start_pos = None

    def set_add_annotation_mode(self, mode):
        self.add_annotation_mode = mode
        if not mode:
            self.remove_temp_box()

    def remove_temp_box(self):
        if hasattr(self, 'temp_box') and self.temp_box is not None:
            self.scene.removeItem(self.temp_box)
            self.temp_box = None
    
    def get_main_window(self):
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, AnnotationTool):
                return parent
            parent = parent.parent()
        return None

    def mousePressEvent(self, event):
        pos = self.mapToScene(event.pos())
        if event.button() == Qt.LeftButton:
            if self.add_annotation_mode:
                if not self.drawing_box and not self.placing_keypoints:
                    self.start_pos = pos
                    self.drawing_box = True
                    print("Started drawing box")
                elif self.placing_keypoints:
                    self.add_keypoint(pos)
            else:
                self.selected_keypoint, self.selected_annotation_index = self.find_nearest_keypoint(pos)
                if self.selected_keypoint is not None:
                    print(f"Selected keypoint: {self.selected_keypoint}")
                    self.setCursor(Qt.ClosedHandCursor)
                    # self.drag_start_pos = self.mapToScene(event.pos())
                    self.keypoint_selected.emit(self.selected_keypoint)
        elif event.button() == Qt.RightButton:
            keypoint, annotation_index = self.find_nearest_keypoint(pos)
            if keypoint is not None:
                self.toggle_keypoint_visibility(keypoint, annotation_index)
        super().mousePressEvent(event)
        
    def mouseMoveEvent(self, event):
        try:
            pos = self.mapToScene(event.pos())
            if self.add_annotation_mode and self.drawing_box:
                self.draw_temporary_box(self.start_pos, pos)
                print(f"Drawing temporary box: {self.start_pos.x():.2f}, {self.start_pos.y():.2f} to {pos.x():.2f}, {pos.y():.2f}")
            elif self.selected_keypoint is not None:
                self.update_keypoint_position(self.selected_keypoint, pos.x(), pos.y())
                # Don't emit keypoint_moved signal here for dragging 
                self.keypoint_moved.emit(self.selected_keypoint, pos.x(), pos.y())
            super().mouseMoveEvent(event)
        except Exception as e:
            print(f"Error in mouseMoveEvent: {e}")
            import traceback
            traceback.print_exc()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.pos())
            if self.add_annotation_mode and self.drawing_box:
                try:
                    self.finish_drawing_box(pos)
                    print("Finished drawing box")
                except Exception as e:
                    print(f"Error in finish_drawing_box: {e}")
                    import traceback
                    traceback.print_exc()
            elif self.selected_keypoint is not None:
            # elif self.selected_keypoint is not None and self.drag_start_pos is not None: # dragging functionality
            # Emit keypoint_moved signal only at the end of drag
            # self.keypoint_moved.emit(self.selected_keypoint, pos.x(), pos.y())

                self.update_keypoint_position(self.selected_keypoint, pos.x(), pos.y())
                print(f"Released keypoint {self.selected_keypoint} at ({pos.x():.2f}, {pos.y():.2f})")
                self.keypoint_released.emit(self.selected_keypoint, pos.x(), pos.y())
                self.selected_keypoint = None
                # self.drag_start_pos = None #dragging functionality

                self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    def find_nearest_keypoint(self, pos):
        min_distance = float('inf')
        nearest_keypoint = None
        nearest_annotation_index = None
        scene_rect = self.sceneRect()
        main_window = self.get_main_window()

        if main_window and main_window.current_frame in main_window.annotations:
            for i, annotation in enumerate(main_window.annotations[main_window.current_frame]):
                keypoints = annotation['keypoints']
                for j in range(0, len(keypoints), 2):
                    kp_x, kp_y = keypoints[j] * scene_rect.width(), keypoints[j+1] * scene_rect.height()
                    distance = ((pos.x() - kp_x) ** 2 + (pos.y() - kp_y) ** 2) ** 0.5
                    if distance < min_distance and distance < 10:  # 10 pixel threshold
                        min_distance = distance
                        nearest_keypoint = j // 2
                        nearest_annotation_index = i
                        
        print(f"Debug: Nearest keypoint = {nearest_keypoint}, Annotation index = {nearest_annotation_index}")
        return nearest_keypoint, nearest_annotation_index

    def update_keypoint_position(self, index, x, y):
        main_window = self.get_main_window()
        if main_window and main_window.current_frame in main_window.annotations:
            annotation = main_window.annotations[main_window.current_frame][self.selected_annotation_index]
            keypoints = annotation['keypoints']
            
            print(f"Debug: Updating keypoint {index}")
            # print(f"Debug: Number of keypoints: {len(keypoints) // 2}")
            # print(f"Debug: Number of confidences: {len(annotation.get('confidences', []))}")
    
            scene_rect = self.sceneRect()
            normalized_x = x / scene_rect.width()
            normalized_y = y / scene_rect.height()
            if index * 2 + 1 < len(keypoints):
                keypoints[index*2] = normalized_x
                keypoints[index*2 + 1] = normalized_y
            else:
                print(f"Warning: Keypoint index {index} is out of range for keypoints list")
                return
        
            if 'confidences' not in annotation:
                annotation['confidences'] = [0.0] * (len(keypoints) // 2)
            if index < len(annotation['confidences']):
                annotation['confidences'][index] = 1.0
            else:
                print(f"Warning: Confidence index {index} is out of range")

            print(f"Updated keypoint {index} to ({normalized_x:.4f}, {normalized_y:.4f})")
            # print(f"Debug: Number of keypoints after update: {len(keypoints) // 2}")

            # Update the visual representation of the keypoint
            if index < len(self.keypoint_items):
                item = self.keypoint_items[index]
                item.setPos(x - 3, y - 3)  # Always use scene coordinates for visual representation
            else:
                print(f"Warning: Visual item for keypoint {index} not found")
            
            # Mark the frame as modified
            if 'modified' not in annotation:
                annotation['modified'] = set()
            annotation['modified'].add(index)
            
            self.scene.update()  # Use self.scene instead of self.scene()

    def toggle_keypoint_visibility(self, keypoint_index, annotation_index):
        main_window = self.get_main_window()
        if main_window and main_window.current_frame in main_window.annotations:
            annotation = main_window.annotations[main_window.current_frame][annotation_index]
            
            print(f"Debug: Keypoint index = {keypoint_index}")
            print(f"Debug: Number of keypoints = {len(annotation['keypoints']) // 2}")
            print(f"Debug: Number of visibilities = {len(annotation.get('visibilities', []))}")
    
            if 'visibilities' not in annotation:
                annotation['visibilities'] = [2] * (len(annotation['keypoints']) // 2)  # Initialize all as visible
            
            if keypoint_index >= len(annotation['visibilities']):
                print(f"Warning: Keypoint index {keypoint_index} is out of range. Max index is {len(annotation['visibilities']) - 1}")
                return
            
            old_visibility = annotation['visibilities'][keypoint_index]
            new_visibility = (old_visibility + 1) % 3  # Cycle through 0, 1, 2
            
            action = (main_window.current_frame, annotation_index, keypoint_index, 
                  old_visibility, new_visibility, 'visibility')
            main_window.add_to_undo_stack(action)

            annotation['visibilities'][keypoint_index] = new_visibility
            
            self.update_keypoint_appearance(keypoint_index, new_visibility)
            
            print(f"Toggled keypoint {keypoint_index} visibility to {new_visibility}")
            
            # Mark the frame as modified
            if 'modified' not in annotation:
                annotation['modified'] = set()
            if isinstance(annotation['modified'], bool):
                annotation['modified'] = set()
            annotation['modified'].add(keypoint_index)
    
    def update_keypoint_appearance(self, keypoint_index, visibility):
        if keypoint_index < len(self.keypoint_items):
            item = self.keypoint_items[keypoint_index]
            if visibility == 2:  # Visible
                item.setBrush(QBrush(QColor(0, 255, 0)))
                item.setPen(QPen(QColor(0, 255, 0)))
            elif visibility == 1:  # Invisible but labeled
                item.setBrush(QBrush(QColor(255, 255, 255)))
                item.setPen(QPen(QColor(255, 255, 255)))
            else:  # Invisible
                item.setBrush(QBrush(QColor(0, 0, 0)))
                pen = QPen(QColor(255, 255, 255))
                pen.setStyle(Qt.DotLine)
                item.setPen(pen)
            self.scene.update() 

    def update_frame(self, pixmap):
        if self.video is not None:
            self.current_frame = int(self.frame_slider.value())
            self.video.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.video.read()
            if ret:
                self.display_frame(frame, print_info=True)
                self.frame_label.setText(f'Frame: {self.current_frame}')  # Update frame label
            else:
                print(f"Failed to read frame {self.current_frame}")
    
    def update_annotations(self, annotations, frame_width, frame_height):
        for item in self.keypoint_items:
            self.scene.removeItem(item)
        self.keypoint_items = []

        if self.bbox_item:
            self.scene.removeItem(self.bbox_item)

        for annotation in annotations:
            # Draw bounding box
            x, y, w, h = annotation['bbox']
            
            # Convert normalized coordinates to pixel coordinates
            w = frame_width * w
            h = frame_height * h
            x = frame_width * x - (w/2)
            y = frame_height * y - (h/2)
            
            # Actual drawing
            self.bbox_item = QGraphicsRectItem(QRectF(x, y, w, h))
            self.bbox_item.setPen(QPen(QColor(255, 0, 0), 2))
            print(f"Drawing bbox_update: {x:.2f}, {y:.2f}, {w:.2f}, {h:.2f}")
            self.scene.addItem(self.bbox_item)

            # Draw keypoints
            keypoints = annotation['keypoints']
            for i in range(0, len(keypoints), 2):
                # Convert normalized coordinates to pixel coordinates
                kp_x, kp_y = keypoints[i] * frame_width, keypoints[i+1] * frame_height
            
                # Actual drawing
                keypoint_item = QGraphicsEllipseItem(0, 0, 10, 10)
                keypoint_item.setPos(kp_x - 5, kp_y - 5)
                keypoint_item.setBrush(QBrush(QColor(0, 255, 0, 128)))
                keypoint_item.setPen(QPen(QColor(0, 255, 0), 2))
                self.scene.addItem(keypoint_item)
                self.keypoint_items.append(keypoint_item)

        print(f"Drew {len(self.keypoint_items)} keypoints")
    
    def draw_temporary_box(self, start, end):
        try:
            if hasattr(self, 'temp_box') and self.temp_box is not None:
                self.scene.removeItem(self.temp_box)
            self.temp_box = self.scene.addRect(QRectF(start, end), QPen(QColor(255, 0, 0)))
            print(f"Drawing temporary box: {start.x():.2f}, {start.y():.2f} to {end.x():.2f}, {end.y():.2f}")
        except Exception as e:
            print(f"Error in draw_temporary_box: {e}")
            import traceback
            traceback.print_exc()

    def add_new_annotation(self, start, end):
        x = (start.x() + end.x()) / (2 * self.scene.width())
        y = (start.y() + end.y()) / (2 * self.scene.height())
        w = abs(end.x() - start.x()) / self.scene.width()
        h = abs(end.y() - start.y()) / self.scene.height()

        num_keypoints = 11

        new_annotation = {
            'bbox': [x, y, w, h],
            'keypoints': [0.0] * (num_keypoints * 2),  # Initialize with zeros
            'confidences': [1.0] * num_keypoints,
            'visibilities': [2] * num_keypoints,  # Initialize all as visible
            'modified': set(range(num_keypoints))  # Set containing indices of all keypoints
        }

        if self.parent().current_frame not in self.parent().annotations:
            self.parent().annotations[self.parent().current_frame] = []
        self.parent().annotations[self.parent().current_frame].append(new_annotation)
        self.parent().display_frame(self.parent().get_current_frame(), print_info=True)
        
        print(f"Debug: Added new annotation with {num_keypoints} keypoints")

    def finish_drawing_box(self, end_pos):
        self.drawing_box = False
        self.placing_keypoints = True
        self.keypoint_count = 0
        scene_rect = self.sceneRect()
        
        # Calculate box dimensions
        left = min(self.start_pos.x(), end_pos.x())
        top = min(self.start_pos.y(), end_pos.y())
        width = abs(end_pos.x() - self.start_pos.x())
        height = abs(end_pos.y() - self.start_pos.y())
        
        # Calculate center coordinates
        center_x = left + width / 2
        center_y = top + height / 2
        
        # Normalize coordinates
        norm_center_x = center_x / scene_rect.width()
        norm_center_y = center_y / scene_rect.height()
        norm_width = width / scene_rect.width()
        norm_height = height / scene_rect.height()
        
        num_keypoints = 11
        
        self.new_annotation = {
            'bbox': [norm_center_x, norm_center_y, norm_width, norm_height],
            'keypoints': [],
            'confidences': [1.0] * num_keypoints,  # Initialize confidences as 0.0
            'modified': set(range(num_keypoints))  # Mark as modified for YOLO export  
        }
        
        # Remove the temporary box
        if self.temp_box:
            # self.remove_temp_box()
            self.scene.removeItem(self.temp_box)
            self.temp_box = None
        
        # Draw the final box
        self.final_box = self.scene.addRect(QRectF(self.start_pos, end_pos), QPen(QColor(255, 0, 0), 2))
        
        print(f"Finished drawing box: x={norm_center_x:.4f}, y={norm_center_y:.4f}, w={norm_width:.4f}, h={norm_height:.4f}, num_keypoints={num_keypoints}")    

        
    def add_keypoint(self, pos):
        main_window = self.get_main_window()
        if main_window is None:
            print("Error: Unable to find main window")
            return

        scene_rect = self.sceneRect()
        normalized_x = pos.x() / scene_rect.width()
        normalized_y = pos.y() / scene_rect.height()
        self.new_annotation['keypoints'].extend([normalized_x, normalized_y])
        
        if 'confidences' not in self.new_annotation:
            self.new_annotation['confidences'] = [0.0] * 11
        self.new_annotation['confidences'][self.keypoint_count] = 1.0

        if 'visibilities' not in self.new_annotation:
            self.new_annotation['visibilities'] = [2] * 11  # 2 represents visible
        self.new_annotation['visibilities'][self.keypoint_count] = 2
        
        # Draw the keypoint
        keypoint_item = self.scene.addEllipse(pos.x() - 3, pos.y() - 3, 6, 6, QPen(QColor(0, 255, 0)), QBrush(QColor(0, 255, 0)))
        self.keypoint_items.append(keypoint_item)
        
        self.keypoint_count += 1
        print(f"Added keypoint {self.keypoint_count}: x={normalized_x:.4f}, y={normalized_y:.4f}")
        
        if self.keypoint_count == 11:
            self.finish_new_annotation()

    def finish_new_annotation(self):
        main_window = self.get_main_window()
        if main_window is None:
            print("Error: Unable to find main window")
            return

        if main_window.current_frame not in main_window.annotations:
            main_window.annotations[main_window.current_frame] = []
        main_window.annotations[main_window.current_frame].append(self.new_annotation)
        main_window.display_frame(main_window.get_current_frame(), print_info=True)
        self.placing_keypoints = False
        self.new_annotation = None
        main_window.toggle_add_annotation_mode()
        print("Finished adding new annotation")

        # Clear temporary drawing items
        for item in self.keypoint_items:
            self.scene.removeItem(item)
        self.keypoint_items.clear()

        if self.final_box is not None:
            self.remove_temp_box()
            self.final_box = None

        main_window.display_frame(main_window.get_current_frame(), print_info=True)

class AnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.video = None
        self.current_frame = 0
        self.annotations = {}
        self.undo_stack = []
        self.redo_stack = []
        self.zoom_factor = 1.0
        self.csv_file = None
        self.selected_keypoint = None
        self.selected_annotation_index = 0  # Assuming only one annotation per frame
        self.graphics_view.keypoint_selected.connect(self.on_keypoint_selected)
        self.graphics_view.keypoint_moved.connect(self.on_keypoint_moved)
        self.graphics_view.keypoint_released.connect(self.on_keypoint_released)
        self.add_annotation_mode = False
        self.use_normalized_coordinates = True  # Flag to determine coordinate system
        self.frame_slider.valueChanged.connect(self.update_frame_label)

    def initUI(self):
        self.setWindowTitle('Annotation Adjustment Tool')
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Video display
        self.graphics_view = AnnotationGraphicsView(self)
        layout.addWidget(self.graphics_view)

        # Controls
        controls_layout = QHBoxLayout()
        self.load_button = QPushButton('Load Video')
        self.load_button.clicked.connect(self.load_video)
        controls_layout.addWidget(self.load_button)
        
        self.load_csv_button = QPushButton('Load CSV')
        self.load_csv_button.clicked.connect(self.load_csv)
        controls_layout.addWidget(self.load_csv_button)

        self.prev_button = QPushButton('Previous Frame')
        self.prev_button.clicked.connect(self.prev_frame)
        controls_layout.addWidget(self.prev_button)

        self.next_button = QPushButton('Next Frame')
        self.next_button.clicked.connect(self.next_frame)
        controls_layout.addWidget(self.next_button)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.update_frame)
        controls_layout.addWidget(self.frame_slider)

        self.add_annotation_button = QPushButton('Add Annotation')
        self.add_annotation_button.clicked.connect(self.toggle_add_annotation_mode)
        controls_layout.addWidget(self.add_annotation_button)

        self.save_button = QPushButton('Save Annotations')
        self.save_button.clicked.connect(self.save_annotations)
        controls_layout.addWidget(self.save_button)
        
        self.save_yolo_button = QPushButton('Save YOLO Annotations')
        self.save_yolo_button.clicked.connect(self.save_yolo_annotations)
        controls_layout.addWidget(self.save_yolo_button)
        
        # self.toggle_coords_button = QPushButton('Toggle Coordinate System')
        # self.toggle_coords_button.clicked.connect(self.toggle_coordinate_system)
        # controls_layout.addWidget(self.toggle_coords_button)

        frame_nav_layout = QHBoxLayout()
    
        self.frame_label = QLabel('Frame: 0')
        frame_nav_layout.addWidget(self.frame_label)
        
        self.frame_input = QLineEdit()
        self.frame_input.setFixedWidth(100)
        frame_nav_layout.addWidget(self.frame_input)
        
        self.go_button = QPushButton('Go')
        self.go_button.clicked.connect(self.go_to_frame)
        frame_nav_layout.addWidget(self.go_button)
        
        controls_layout.addLayout(frame_nav_layout)        
        
        layout.addLayout(controls_layout)
        
    # def create_file_dialog(self, file_type="video"):
    #     dialog = QFileDialog(self)
    #     dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        
    #     if file_type == "video":
    #         dialog.setWindowTitle("Open Video File")
    #         dialog.setNameFilter("Video Files (*.mp4 *.avi)")
    #     else:  # CSV
    #         dialog.setWindowTitle("Open CSV File")
    #         dialog.setNameFilter("CSV Files (*.csv)")
        
    #     dialog.setFileMode(QFileDialog.ExistingFile)
    #     return dialog

    def load_video(self):
        # added for testing
        # dialog = self.create_file_dialog("video")
        # if dialog.exec_() == QFileDialog.Accepted:
        #     file_name = dialog.selectedFiles()[0]
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi)")
                                                #    ,options=QFileDialog.DontUseNativeDialog) # added for testing
        if file_name:
            self.video = cv2.VideoCapture(file_name)
            self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.original_video_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.original_video_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_slider.setRange(0, self.total_frames - 1)
            self.annotations = {}  # Initialize annotations dictionary
            self.current_frame = 0  # Reset current frame to 0
            self.update_frame()
            print(f"Loaded video with {self.total_frames} frames")  # Print total frames

    def toggle_coordinate_system(self):
        self.use_normalized_coordinates = not self.use_normalized_coordinates
        print(f"Using {'normalized' if self.use_normalized_coordinates else 'non-normalized'} coordinates")
        self.update_frame()  # Redraw the current frame with the new coordinate system

    def update_frame(self):
        if self.video is not None:
            self.current_frame = self.frame_slider.value()
            self.video.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.video.read()
            if ret:
                self.display_frame(frame,print_info=True)
            else:
                print(f"Failed to read frame {self.current_frame}")
                
    def go_to_frame(self):
        try:
            frame_number = int(self.frame_input.text())
            if 0 <= frame_number < self.total_frames:
                self.frame_slider.setValue(frame_number)
            else:
                print(f"Frame number out of range. Please enter a number between 0 and {self.total_frames - 1}")
        except ValueError:
            print("Please enter a valid frame number")
    
    def update_frame_label(self):
        self.frame_label.setText(f'Frame: {self.frame_slider.value()}')

    def create_custom_cursor(self):
        cursor_pixmap = QPixmap(20, 20)
        cursor_pixmap.fill(Qt.transparent)
        painter = QPainter(cursor_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(0, 255, 0, 128), 2))  # Semi-transparent green
        painter.setBrush(QColor(0, 255, 0, 64))  # Very transparent green fill
        painter.drawEllipse(0, 0, 19, 19)
        painter.end()
        return QCursor(cursor_pixmap)

    def load_csv(self):
        
        # added for testing
        # dialog = self.create_file_dialog("csv")
        # if dialog.exec_() == QFileDialog.Accepted:
        #     file_name = dialog.selectedFiles()[0]

        file_name, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
                                                #    ,options=QFileDialog.DontUseNativeDialog) # added for testing
        if file_name:
            self.csv_file = file_name
            self.load_annotations()

    def load_annotations(self):
        if self.csv_file:
            self.annotations = {}
            try:
                data = np.loadtxt(self.csv_file, delimiter=',')
                self.use_normalized_coordinates = np.all((0 <= data[0, 2:6]) & (data[0, 2:6] <= 1))

                print(f"Detected coordinate system: {'Normalized' if self.use_normalized_coordinates else 'Unnormalized'}")
                # print(f"Bounding box format: {'Center-based' if self.center_based_bbox else 'Corner-based'}")

                if not self.use_normalized_coordinates:
                    frame = self.get_current_frame()
                    if frame is not None:
                        self.original_height, self.original_width = frame.shape[:2]
                    else:
                        print("Warning: Unable to get frame dimensions. Using default values.")
                        self.original_width, self.original_height = 1920, 1080

                    # print("Before normalization:")
                    # print(f"bbox: {data[0, 2:6]}")
                    # print(f"keypoints: {data[0, 7:29]}")
                    
                    data[:, 2:6] = self.unnormalized_to_normalized(data[:, 2:6])
                    data[:, 7:29] = self.unnormalized_to_normalized(data[:, 7:29])
                    
                    # print("After normalization:")
                    # print(f"bbox: {data[0, 2:6]}")
                    # print(f"keypoints: {data[0, 7:29]}")

                for row in data:
                    frame = int(round(row[0]))
                    if frame not in self.annotations:
                        self.annotations[frame] = []
                    
                    self.annotations[frame].append({
                        'bbox': row[2:6].tolist(),
                        'keypoints': row[7:29].tolist(),
                        'confidences': row[29:40].tolist(),
                        'modified': set()
                    })

                print(f"Loaded annotations for {len(self.annotations)} frames")

                self.update_frame()
            except Exception as e:
                print(f"Error loading annotations: {e}")
                import traceback
                traceback.print_exc()
                
    def unnormalized_to_normalized(self, coords):
        coords = np.array(coords, dtype=float)
        original_shape = coords.shape
        
        # Reshape to 2D array if it's 1D
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
        
        # Normalize all x coordinates (even indices)
        coords[:, 0::2] /= self.original_width
        
        # Normalize all y coordinates (odd indices)
        coords[:, 1::2] /= self.original_height
        
        return coords.reshape(original_shape)
    
    def display_frame(self, frame, print_info=False):
        if frame is None:
            print("Error: Frame is None")
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, ch = frame.shape
        bytes_per_line = ch * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        self.graphics_view.scene.clear()  # Clear the scene before adding new items
        pixmap_item = self.graphics_view.scene.addPixmap(pixmap)
        self.graphics_view.fitInView(pixmap_item, Qt.KeepAspectRatio)
        
        frame_number = int(self.current_frame)  # Ensure frame number is an integer
        if frame_number in self.annotations:
            for annotation in self.annotations[self.current_frame]:
                # Draw bounding box
                x, y, w, h = annotation['bbox']
                
                scaled_x = x * width
                scaled_y = y * height
                scaled_w = w * width
                scaled_h = h * height
                
                scaled_x -= scaled_w / 2
                scaled_y -= scaled_h / 2

                rect_item = self.graphics_view.scene.addRect(scaled_x, scaled_y, scaled_w, scaled_h, QPen(QColor(255, 0, 0), 2))
                print(f"Drawing bbox_display on frame {frame_number}: {scaled_x:.4f}, {scaled_y:.4f}, {scaled_w:.4f}, {scaled_h:.4f}")

                # Draw keypoints
                keypoints = annotation['keypoints']
                visibilities = annotation.get('visibilities', [2] * (len(keypoints) // 2))  # Default to all visible
                self.graphics_view.keypoint_items = []  # Clear previous keypoint items
                for i in range(0, len(keypoints), 2):
                    kp_x, kp_y = keypoints[i] * width, keypoints[i+1] * height

                    visibility = visibilities[i // 2] if i // 2 < len(visibilities) else 2  # Default to visible if not specified

                    keypoint_item = QGraphicsEllipseItem(0, 0, 6, 6)
                    keypoint_item.setPos(kp_x - 3, kp_y - 3)
                    
                    if visibility == 2:  # Visible
                        keypoint_item.setBrush(QBrush(QColor(0, 255, 0)))
                        keypoint_item.setPen(QPen(QColor(0, 255, 0)))
                    elif visibility == 1:  # Invisible but labeled
                        keypoint_item.setBrush(QBrush(QColor(255, 255, 255)))
                        keypoint_item.setPen(QPen(QColor(255, 255, 255)))
                    else:  # Invisible
                        keypoint_item.setBrush(QBrush(Qt.NoBrush))  # Use QBrush(Qt.NoBrush) instead of Qt.NoBrush
                        pen = QPen(QColor(255, 0, 0))
                        pen.setStyle(Qt.DotLine)
                        keypoint_item.setPen(pen)
        
                    self.graphics_view.scene.addItem(keypoint_item)
                    self.graphics_view.keypoint_items.append(keypoint_item)
                                
            print(f"Drew {len(self.annotations[self.current_frame])} annotations")
        else:
            print(f"No annotations for frame {self.current_frame}")
        
        if print_info:
            print(f"Displaying frame {self.current_frame}, annotations present: {self.current_frame in self.annotations}")
            if self.current_frame in self.annotations:
                print(f"Annotations for frame {self.current_frame}: {self.annotations[self.current_frame]}")

    def get_current_frame(self):
        if self.video is not None:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.video.read()
            if ret:
                return frame
        return None

    def prev_frame(self):
        self.frame_slider.setValue(max(0, self.current_frame - 1))

    def next_frame(self):
        self.frame_slider.setValue(min(self.total_frames - 1, self.current_frame + 1))
        
    def convert_coordinates(self, x, y, w, h, to_normalized=True):
        if to_normalized == self.use_normalized_coordinates:
            return x, y, w, h
        
        frame = self.get_current_frame()
        if frame is None:
            return x, y, w, h
        
        height, width = frame.shape[:2]
        
        if to_normalized:
            return x / width, y / height, w / width, h / height
        else:
            return x * width, y * height, w * width, h * height

    
    def update_annotation_data(self, keypoint_index, x, y):
        if self.current_frame in self.annotations and self.annotations[self.current_frame]:
            annotation = self.annotations[self.current_frame][0]  # Assuming only one annotation per frame
            keypoints = annotation['keypoints']
            frame_width = self.graphics_view.scene.width()
            frame_height = self.graphics_view.scene.height()
            keypoints[keypoint_index*2] = x / frame_width
            keypoints[keypoint_index*2 + 1] = y / frame_height
            print(f"Updated annotation data for keypoint {keypoint_index}")

    def on_keypoint_selected(self, keypoint_index):
        self.selected_keypoint = keypoint_index
        print(f"Keypoint {keypoint_index} selected")

    def on_keypoint_moved(self, keypoint_index, x, y):
        if self.current_frame in self.annotations:
            annotation = self.annotations[self.current_frame][self.selected_annotation_index]
            keypoints = annotation['keypoints']
            frame_width = self.graphics_view.scene.width()
            frame_height = self.graphics_view.scene.height()
            
            old_x, old_y = keypoints[keypoint_index*2], keypoints[keypoint_index*2 + 1]
            new_x, new_y = x / frame_width, y / frame_height
            
            action = (self.current_frame, self.selected_annotation_index, keypoint_index, 
                      (old_x, old_y), (new_x, new_y), 'keypoint')
            self.add_to_undo_stack(action)
            
            keypoints[keypoint_index*2] = new_x
            keypoints[keypoint_index*2 + 1] = new_y
            print(f"Moved keypoint {keypoint_index} to ({x/frame_width:.4f}, {y/frame_height:.4f})")
            
            # Mark this frame as modified
            if 'modified' not in annotation:
                annotation['modified'] = set()
            annotation['modified'].add(keypoint_index)

    def on_keypoint_released(self, keypoint_index, x, y):
        self.on_keypoint_moved(keypoint_index, x, y)
        self.selected_keypoint = None
        print(f"Released keypoint {keypoint_index}")

    def toggle_add_annotation_mode(self):
        self.add_annotation_mode = not self.add_annotation_mode
        self.graphics_view.add_annotation_mode = self.add_annotation_mode
        if self.add_annotation_mode:
            self.add_annotation_button.setText('Cancel Add')
            self.graphics_view.setCursor(Qt.CrossCursor)
        else:
            self.add_annotation_button.setText('Add Annotation')
            self.graphics_view.setCursor(Qt.ArrowCursor)
        print(f"Add annotation mode toggled: {self.add_annotation_mode}")

    def save_annotations(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Annotations", "", "CSV Files (*.csv)")
        if file_name:
            try:
                with open(file_name, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    for frame, annotations in sorted(self.annotations.items()):
                        for annotation in annotations:
                            row = [frame, 1]  # Assuming track_id is always 1
                            bbox = annotation['bbox']
                            keypoints = annotation['keypoints']
                            if not self.use_normalized_coordinates:
                                bbox = self.convert_coordinates(*bbox, to_normalized=False)
                                keypoints = [coord for pair in zip(keypoints[::2], keypoints[1::2]) 
                                             for coord in self.convert_coordinates(*pair, 0, 0, to_normalized=False)[:2]]


                            row.extend(annotation['bbox'])
                            row.append(1.0)  # Confidence score, assuming it's always 1.0
                            row.extend(annotation['keypoints'])
                            
                            if 'confidences' in annotation:
                                row.extend(annotation['confidences'])
                            else:
                                row.extend([0.0] * 11)  # Default to 1.0 if not specified
                                
                            writer.writerow(row)
                print(f"Annotations saved successfully to {file_name}")
            except Exception as e:
                print(f"Error saving annotations: {e}")

    def reset_selection(self):
        self.selected_keypoint = None
        self.selected_annotation_index = None
        self.setCursor(Qt.ArrowCursor)
        self.graphics_view.setCursor(Qt.ArrowCursor)  # Ensure the label cursor is also reset

    def save_yolo_annotations(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory for YOLO Annotations")
        if directory:
            try:
                for frame, annotations in self.annotations.items():
                    if any(ann.get('modified') for ann in annotations):  # Check if any annotation in the frame is modified
                        print(f"Saving modified frame {frame}")
                        image_filename = f"frame_{frame:06d}.jpg"
                        annotation_filename = f"frame_{frame:06d}.txt"
                        
                        # Save the image
                        current_frame = self.get_frame(frame)
                        # image_filename_png = os.path.splitext(image_filename)[0] + ".png" # in case you want to save as png
                        if current_frame is not None:
                            image_path = os.path.join(directory, "images", image_filename)
                            # image_path = os.path.join(directory, "images", image_filename_png) # in case you want to save as png
                            os.makedirs(os.path.dirname(image_path), exist_ok=True)
                            cv2.imwrite(image_path, current_frame)
                        
                        # Save the annotation
                        annotation_path = os.path.join(directory, "labels", annotation_filename)
                        os.makedirs(os.path.dirname(annotation_path), exist_ok=True)
                        with open(annotation_path, 'w') as f:
                            for annotation in annotations:
                                if annotation.get('modified'):  # Only save modified annotations
                                    bbox = annotation['bbox']
                    
                                    x_center = bbox[0] + bbox[2]/2
                                    y_center = bbox[1] + bbox[3]/2
                                    w,h = bbox[2], bbox[3]
                
                                    # Start with class ID and bounding box
                                    line = f"0 {x_center} {y_center} {w} {h}"
                                
                                    # Add keypoints
                                    keypoints = annotation['keypoints']
                                    visibilities = annotation.get('visibilities', [2] * 11)  # Default to all visible

                                    for i in range(0, len(keypoints), 2):
                                        x,y = keypoints[i], keypoints[i+1]
                                        visibility = visibilities[i // 2]
                                        line += f" {x} {y} {visibility}"
                                    
                                    f.write(line + "\n")
                
                print(f"YOLO-style annotations saved in {directory}")
            except Exception as e:
                print(f"Error saving YOLO annotations: {e}")
                import traceback
                traceback.print_exc()

    def get_frame(self, frame_number):
        if self.video is not None:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.video.read()
            if ret:
                return frame
        return None

    def get_scaled_position(self, pos):
        view_geom = self.graphics_view.viewport().rect()
        scene_geom = self.graphics_view.sceneRect()
        
        # Convert view coordinates to scene coordinates
        scene_pos = self.graphics_view.mapToScene(pos)
        
        # Normalize the coordinates
        normalized_x = (scene_pos.x() - scene_geom.left()) / scene_geom.width()
        normalized_y = (scene_pos.y() - scene_geom.top()) / scene_geom.height()
        
        return QPointF(normalized_x, normalized_y)

    def find_nearest_keypoint(self, scaled_pos):
        min_distance = float('inf')
        nearest_keypoint = None
        nearest_annotation_index = None
        main_window = self.get_main_window()
        scene_rect = self.sceneRect()
        
        if main_window and main_window.current_frame in main_window.annotations:
            for i, annotation in enumerate(main_window.annotations[main_window.current_frame]):
                keypoints = annotation['keypoints']
                for j in range(0, len(keypoints), 2):
                    kp_x, kp_y = keypoints[j] * scene_rect.width(), keypoints[j+1] * scene_rect.height()
                    distance = ((scaled_pos.x() - kp_x) ** 2 + (scaled_pos.y() - kp_y) ** 2) ** 0.5
                    print(f"Keypoint {j//2}: ({kp_x:.4f}, {kp_y:.4f}), Distance: {distance:.4f}")

                    # if distance < min_distance and distance < 0.02:  # Adjust this threshold as needed
                    if distance < min_distance and distance < 10:  # Adjust this threshold as needed
                        min_distance = distance
                        nearest_keypoint = j // 2
                        nearest_annotation_index = i
                        
        print(f"Nearest keypoint: {nearest_keypoint}, Annotation index: {nearest_annotation_index}")
        return nearest_keypoint, nearest_annotation_index

    def get_image_position(self, pos):
        label_size = self.graphics_view.size()
        pixmap = self.graphics_view.pixmap()
        if pixmap:
            video_size = pixmap.size()
            video_ratio = video_size.width() / video_size.height()
            label_ratio = label_size.width() / label_size.height()
            
            if video_ratio > label_ratio:
                # Video is wider than label
                scale = label_size.width() / video_size.width()
                offset_y = (label_size.height() - video_size.height() * scale) / 2
                return QPointF(pos.x() / scale / video_size.width(), 
                            (pos.y() - offset_y) / scale / video_size.height())
            else:
                # Video is taller than label
                scale = label_size.height() / video_size.height()
                offset_x = (label_size.width() - video_size.width() * scale) / 2
                return QPointF((pos.x() - offset_x) / scale / video_size.width(), 
                            pos.y() / scale / video_size.height())
        return QPointF(pos) / label_size

    def undo(self):
        if self.undo_stack:
            action = self.undo_stack.pop()
            self.redo_stack.append(self.perform_action(action, undo=True))
            self.update_frame()

    def redo(self):
        if self.redo_stack:
            action = self.redo_stack.pop()
            self.undo_stack.append(self.perform_action(action, undo=False))
            self.update_frame()
    
    def perform_action(self, action, undo=False):
        frame, annotation_index, keypoint_index, old_value, new_value, action_type = action
        
        if frame not in self.annotations or annotation_index >= len(self.annotations[frame]):
            return action
        
        annotation = self.annotations[frame][annotation_index]
        
        if action_type == 'keypoint':
            if keypoint_index * 2 + 1 < len(annotation['keypoints']):
                old_x, old_y = annotation['keypoints'][keypoint_index*2:keypoint_index*2+2]
                if undo:
                    annotation['keypoints'][keypoint_index*2:keypoint_index*2+2] = old_value
                else:
                    annotation['keypoints'][keypoint_index*2:keypoint_index*2+2] = new_value
                return (frame, annotation_index, keypoint_index, (old_x, old_y), (old_value if undo else new_value), action_type)
        elif action_type == 'visibility':
            if 'visibilities' in annotation and keypoint_index < len(annotation['visibilities']):
                old_visibility = annotation['visibilities'][keypoint_index]
                if undo:
                    annotation['visibilities'][keypoint_index] = old_value
                else:
                    annotation['visibilities'][keypoint_index] = new_value
                return (frame, annotation_index, keypoint_index, old_visibility, (old_value if undo else new_value), action_type)
        
        return action

    def add_to_undo_stack(self, action):
        self.undo_stack.append(action)
        self.redo_stack.clear()  # Clear redo stack when a new action is performed
        
    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Undo):
            self.undo()
        elif event.matches(QKeySequence.Redo):
            self.redo()
        else:
            super().keyPressEvent(event)
            
    def zoom_in(self):
        self.zoom_factor *= 1.2
        self.update_frame()

    def zoom_out(self):
        self.zoom_factor /= 1.2
        self.update_frame()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AnnotationTool()
    ex.show()
    sys.exit(app.exec_())