import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
from datetime import datetime

class PersonTracker:
    def __init__(self, reference_img_path=None, display_scale=0.7, processing_scale=0.5):
        # Initialize models with optimized settings
        self.yolo_model = YOLO('yolov8n.pt')
        self.tracker = DeepSort(max_age=10, n_init=2)
        
        # Tracking data
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.target_id = None
        self.target_first_appearance = None
        self.target_first_frame = None
        self.target_first_timestamp = None
        self.real_world_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.current_real_time = None
        
        # Visualization settings
        self.display_scale = display_scale
        self.processing_scale = processing_scale
        self.flow_line_color = (0, 255, 255)  # Yellow
        self.bbox_color = (0, 255, 0)  # Green
        self.bbox_thickness = 2
        self.show_flow_line = True
        self.show_bbox = True
        self.text_color = (0, 255, 255)  # Yellow
        
        # Enhanced target identification parameters
        self.green_hue_range = (40, 80)  # Green color range in HSV
        self.white_value_threshold = 200  # Minimum value for white detection
        self.white_saturation_threshold = 30  # Maximum saturation for white
        self.target_score_threshold = 0.3  # Minimum score to identify target
        
        # Reference image handling
        self.reference_embedding = None
        if reference_img_path:
            roi_img = self._load_reference_image(reference_img_path)
            if roi_img is not None:
                self.reference_embedding = self._extract_features(roi_img)

    def _load_reference_image(self, path):
        """Load and process reference image with enhanced error handling"""
        try:
            ref_img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_2)
            if ref_img is None:
                print(f"Error: Could not load reference image at {path}")
                return None
                
            cv2.namedWindow("Select Target Region", cv2.WINDOW_NORMAL)
            roi = cv2.selectROI("Select Target Region", ref_img, showCrosshair=True)
            cv2.destroyWindow("Select Target Region")
            
            if roi == (0, 0, 0, 0):
                print("ROI selection was cancelled")
                return None
                
            x, y, w, h = map(int, roi)
            roi_img = ref_img[y:y+h, x:x+w]
            
            if roi_img.size == 0:
                print("Selected ROI is empty")
                return None
                
            cv2.imwrite("selected_reference_roi.jpg", roi_img)
            return roi_img
        except Exception as e:
            print(f"Error loading reference image: {e}")
            return None

    def _extract_features(self, image):
        """Enhanced feature extraction with multiple feature types"""
        if image is None or image.size == 0:
            return None
            
        try:
            small_img = cv2.resize(image, (64, 128))
            
            # Color histogram features
            hsv = cv2.cvtColor(small_img, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv], [0], None, [90], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [128], [0, 256])
            cv2.normalize(hist_h, hist_h)
            cv2.normalize(hist_s, hist_s)
            
            # Edge features (optional)
            gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1] * 255)
            
            return (hist_h, hist_s, edge_density)
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

    def detect_people(self, frame):
        """Detect people in frame using YOLO with optimized settings"""
        try:
            if self.processing_scale < 1.0:
                small_frame = cv2.resize(frame, (0,0), fx=self.processing_scale, fy=self.processing_scale)
            else:
                small_frame = frame
                
            results = self.yolo_model(small_frame, classes=[0], verbose=False, imgsz=320)
            
            detections = []
            scale = 1.0 / self.processing_scale
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = [int(coord * scale) for coord in box.xyxy[0].tolist()]
                    conf = float(box.conf[0])
                    detections.append(([x1, y1, x2-x1, y2-y1], conf, None))
            return detections
        except Exception as e:
            print(f"Detection error: {e}")
            return []

    def track_people(self, frame, detections):
        """Track detected people across frames"""
        try:
            return self.tracker.update_tracks(detections, frame=frame)
        except Exception as e:
            print(f"Tracking error: {e}")
            return []

    def find_target_person(self, frame, tracks):
        """Enhanced target identification with multiple criteria"""
        if self.target_id is not None:
            return self.target_id
            
        best_match = (None, 0)
        
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            if x1 >= x2 or y1 >= y2:
                continue
                
            try:
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                    
                # Method 1: Reference image matching (if available)
                if self.reference_embedding is not None:
                    current_features = self._extract_features(roi)
                    if current_features is None:
                        continue
                        
                    # Compare all features (histograms + edge density)
                    hist_similarity = sum(
                        cv2.compareHist(ref, curr, cv2.HISTCMP_CORREL)
                        for ref, curr in zip(self.reference_embedding[:2], current_features[:2])
                    ) / 2
                    
                    # Edge density similarity (1 - absolute difference)
                    edge_similarity = 1 - abs(self.reference_embedding[2] - current_features[2])
                    
                    # Combined similarity score
                    similarity = (hist_similarity * 0.8 + edge_similarity * 0.2)
                    
                    if similarity > best_match[1]:
                        best_match = (track.track_id, similarity)
                
                # Method 2: Color-based detection (green jacket + white shirt)
                else:
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    
                    # Check for green jacket (Hue ~60 in OpenCV)
                    green_mask = cv2.inRange(hsv_roi, 
                                           (self.green_hue_range[0], 40, 40),
                                           (self.green_hue_range[1], 255, 255))
                    green_pixels = cv2.countNonZero(green_mask)
                    green_ratio = green_pixels / (roi.shape[0] * roi.shape[1])
                    
                    # Check for white shirt (high Value, low Saturation)
                    white_mask = cv2.inRange(hsv_roi, 
                                           (0, 0, self.white_value_threshold),
                                           (180, self.white_saturation_threshold, 255))
                    white_ratio = cv2.countNonZero(white_mask) / (roi.shape[0] * roi.shape[1])
                    
                    # Check for blue elements (trolley/plants)
                    blue_mask = cv2.inRange(hsv_roi, (100, 50, 50), (140, 255, 255))
                    blue_ratio = cv2.countNonZero(blue_mask) / (roi.shape[0] * roi.shape[1])
                    
                    # Combined score with weights
                    combined_score = (green_ratio * 0.5 + 
                                    white_ratio * 0.3 + 
                                    blue_ratio * 0.2)
                    
                    if combined_score > self.target_score_threshold:
                        self.target_id = track.track_id
                        return self.target_id
                        
            except Exception as e:
                print(f"Error in target identification: {e}")
                continue
                
        # If using reference matching and we have a good match
        if self.reference_embedding and best_match[1] > 0.7:
            self.target_id = best_match[0]
            return self.target_id
            
        return None

    def process_video(self, video_path, output_path):
        """Enhanced video processing with better visualization"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video {video_path}")
                return
                
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_skip = 0 if fps < 15 else 1
            frame_count = 0
            target_found = False
            last_time = time.time()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_skip > 0 and frame_count % (frame_skip + 1) != 0:
                    frame_count += 1
                    continue
                    
                frame_count += 1
                current_timestamp = frame_count / fps
                self.current_real_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                detections = self.detect_people(frame)
                tracks = self.track_people(frame, detections)
                
                if not target_found:
                    target_id = self.find_target_person(frame, tracks)
                    if target_id is not None:
                        target_found = True
                        self.target_first_frame = frame_count
                        self.target_first_timestamp = current_timestamp
                        self.target_first_appearance = time.time()
                        print(f"\nTarget found at frame {frame_count} ({current_timestamp:.2f}s)")
                
                display_frame = frame.copy()
                
                # Enhanced timestamp display
                cv2.putText(display_frame, f"Target first appeared at: {self.real_world_time}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, self.text_color, 2)
                cv2.putText(display_frame, f"Current Time: {self.current_real_time}", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, self.text_color, 2)
                
                if target_found:
                    cv2.putText(display_frame, 
                               f"Target First Seen: {self.target_first_timestamp:.2f}s ",
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, self.text_color, 2)
                
                if target_found:
                    for track in tracks:
                        if track.track_id == self.target_id and track.is_confirmed():
                            x1, y1, x2, y2 = map(int, track.to_ltrb())
                            center = ((x1 + x2) // 2, (y1 + y2) // 2)
                            self.track_history[track.track_id].append(center)
                            
                            if self.show_bbox:
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), 
                                             self.bbox_color, self.bbox_thickness)
                                # Add ID text
                                cv2.putText(display_frame, f"Target {track.track_id}",
                                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.6, self.bbox_color, 2)
                            
                            if self.show_flow_line and len(self.track_history[track.track_id]) > 1:
                                points = np.array(self.track_history[track.track_id], np.int32)
                                if len(points) >= 4:
                                    points = cv2.approxPolyDP(points, 2, False)
                                cv2.polylines(display_frame, [points], False, 
                                              self.flow_line_color, 2, lineType=cv2.LINE_AA)
                
                out.write(display_frame)
                
                if time.time() - last_time > 0.033:
                    if self.display_scale != 1.0:
                        disp_frame = cv2.resize(display_frame, (0,0), 
                                              fx=self.display_scale, 
                                              fy=self.display_scale)
                    else:
                        disp_frame = display_frame
                    
                    # Add keyboard controls for toggling views
                    cv2.imshow('Person Tracking', disp_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('b'):
                        self.show_bbox = not self.show_bbox
                    elif key == ord('f'):
                        self.show_flow_line = not self.show_flow_line
                    
                    last_time = time.time()
            
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            if target_found:
                duration = time.time() - self.target_first_appearance
                print(f"\nTracking completed in {duration:.2f}s")
                print(f"Processing speed: {frame_count/duration:.2f} FPS")
                print(f"Target first appeared at: {self.target_first_timestamp:.2f}s (Frame {self.target_first_frame})")
            
        except Exception as e:
            print(f"Error in video processing: {e}")
            if 'cap' in locals(): cap.release()
            if 'out' in locals(): out.release()
            cv2.destroyAllWindows()