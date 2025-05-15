import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
from collections import defaultdict, deque
import os
from datetime import datetime
import tkinter as tk
import json
import pickle
import logging
import sys
import warnings
from PIL import Image, ImageDraw, ImageFont


warnings.filterwarnings('ignore')


if sys.platform.startswith('win'):
    import locale
    locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
    sys.stdout.reconfigure(encoding='utf-8')

class VehicleSpeedDetector:
    def __init__(self, video_path):
        self.video_path = video_path
        

        logging.getLogger("ultralytics").setLevel(logging.ERROR)
        

        try:

            self.model = YOLO('yolov8n.pt', torch_load_kwargs={'weights_only': False})
        except:

            original_torch_load = torch.load
            def torch_load_with_weights_only_false(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_torch_load(*args, **kwargs)
            torch.load = torch_load_with_weights_only_false
            self.model = YOLO('yolov8n.pt')
            torch.load = original_torch_load
            
        self.model.verbose = False  
        

        self.output_dir = "speed_records"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        

        try:
            self.font = ImageFont.truetype("simhei.ttf", 20)  
        except:
            try:
                self.font = ImageFont.truetype("simsun.ttc", 20)  
            except:
                self.font = ImageFont.load_default()
        

        self.tracks = defaultdict(list)
        self.speeds = defaultdict(list)
        self.current_speeds = defaultdict(float)
        self.vehicle_ids = {}
        self.vehicle_types = {}
        self.speed_history = defaultdict(lambda: deque(maxlen=10))
        

        self.learning_data = {
            'calibration_factors': deque(maxlen=100),  
            'speed_corrections': defaultdict(lambda: deque(maxlen=50)),  
            'vehicle_type_speeds': defaultdict(lambda: deque(maxlen=100)),  
            'track_quality': defaultdict(float),
            'environment_factors': defaultdict(float),  
            'confidence_scores': defaultdict(float),  
            'last_update': datetime.now()  
        }
        

        self.load_learning_data()
        

        self._ensure_learning_data_structure()
        

        self.class_names = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        

        self.calibration_factor = self.get_optimal_calibration_factor()
        self.reference_length = None    
        self.reference_pixels = None    
        self.perspective_matrix = None  
        self.perspective_points = []    
        

        self.roi_points = None
        

        self.calibration_mode = False
        self.calibration_points = []
        self.calibration_step = 1  
        

        self.vehicle_counter = 0
        

        self.video_writer = None
        

        self.min_track_points = 3  
        self.max_speed = 200.0     
        self.min_speed = 40.0       
        self.speed_window = 10     
        self.direction_threshold = 45  

    def _ensure_learning_data_structure(self):
        required_keys = [
            'calibration_factors',
            'speed_corrections',
            'vehicle_type_speeds',
            'track_quality',
            'environment_factors',
            'confidence_scores',
            'last_update'
        ]
        

        for key in required_keys:
            if key not in self.learning_data:
                if key == 'calibration_factors':
                    self.learning_data[key] = deque(maxlen=100)
                elif key in ['speed_corrections', 'vehicle_type_speeds']:
                    self.learning_data[key] = defaultdict(lambda: deque(maxlen=50))
                elif key in ['track_quality', 'environment_factors', 'confidence_scores']:
                    self.learning_data[key] = defaultdict(float)
                elif key == 'last_update':
                    self.learning_data[key] = datetime.now()

    def load_learning_data(self):
        try:
            load_path = os.path.join(self.output_dir, 'learning_data.pkl')
            if os.path.exists(load_path):
                with open(load_path, 'rb') as f:
                    loaded_data = pickle.load(f)
                    

                    if 'calibration_factors' in loaded_data:
                        self.learning_data['calibration_factors'] = deque(
                            loaded_data['calibration_factors'], maxlen=100)
                    
                    if 'speed_corrections' in loaded_data:
                        for k, v in loaded_data['speed_corrections'].items():
                            self.learning_data['speed_corrections'][k] = deque(v, maxlen=50)
                    
                    if 'vehicle_type_speeds' in loaded_data:
                        for k, v in loaded_data['vehicle_type_speeds'].items():
                            self.learning_data['vehicle_type_speeds'][k] = deque(v, maxlen=100)
                    
                    if 'track_quality' in loaded_data:
                        self.learning_data['track_quality'].update(loaded_data['track_quality'])
                    
                    if 'environment_factors' in loaded_data:
                        self.learning_data['environment_factors'].update(loaded_data['environment_factors'])
                    
                    if 'confidence_scores' in loaded_data:
                        self.learning_data['confidence_scores'].update(loaded_data['confidence_scores'])
                    
                    if 'last_update' in loaded_data:
                        self.learning_data['last_update'] = loaded_data['last_update']
                
        except Exception as e:
            print(f"Failed to load learning data: {e}")
            import traceback
            traceback.print_exc()

            self._ensure_learning_data_structure()

    def save_learning_data(self):
        try:
            self._ensure_learning_data_structure()
            

            save_data = {
                'calibration_factors': list(self.learning_data['calibration_factors']),
                'speed_corrections': {k: list(v) for k, v in self.learning_data['speed_corrections'].items()},
                'vehicle_type_speeds': {k: list(v) for k, v in self.learning_data['vehicle_type_speeds'].items()},
                'track_quality': dict(self.learning_data['track_quality']),
                'environment_factors': dict(self.learning_data['environment_factors']),
                'confidence_scores': dict(self.learning_data['confidence_scores']),
                'last_update': self.learning_data['last_update']
            }
            

            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            

            save_path = os.path.join(self.output_dir, 'learning_data.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(save_data, f)
            
        except Exception as e:
            import traceback
            traceback.print_exc()

    def get_optimal_calibration_factor(self):
        if self.learning_data['calibration_factors']:

            return np.median(self.learning_data['calibration_factors'])
        return None
        
    def update_learning_data(self, track_id, speed, vehicle_type):
        current_time = datetime.now()
        time_diff = (current_time - self.learning_data['last_update']).total_seconds()
        

        if speed > 0:

            confidence = self.calculate_confidence(track_id, speed, vehicle_type)
            self.learning_data['confidence_scores'][track_id] = confidence
            

            if confidence > 0.7:  
                self.learning_data['speed_corrections'][vehicle_type].append(speed)
                self.learning_data['vehicle_type_speeds'][vehicle_type].append(speed)
        

        track = self.tracks[track_id]
        if len(track) >= self.min_track_points:

            start_point = np.array(track[0][:2])
            end_point = np.array(track[-1][:2])
            total_distance = np.sqrt(np.sum((end_point - start_point) ** 2))
            
            actual_distance = 0
            for i in range(1, len(track)):
                p1 = np.array(track[i-1][:2])
                p2 = np.array(track[i][:2])
                actual_distance += np.sqrt(np.sum((p2 - p1) ** 2))
            

            straightness = total_distance / actual_distance if actual_distance > 0 else 0
            

            time_weight = np.exp(-time_diff / 3600)  
            self.learning_data['track_quality'][track_id] = straightness * time_weight
        

        self.update_environment_factors()
        

        self.learning_data['last_update'] = current_time
        
    def calculate_confidence(self, track_id, speed, vehicle_type):
        confidence = 1.0
        

        track_quality = self.learning_data['track_quality'].get(track_id, 0)
        confidence *= track_quality
        

        if vehicle_type in self.learning_data['vehicle_type_speeds']:
            type_speeds = self.learning_data['vehicle_type_speeds'][vehicle_type]
            if type_speeds:
                avg_speed = np.mean(type_speeds)
                std_speed = np.std(type_speeds)
                if std_speed > 0:

                    z_score = abs(speed - avg_speed) / std_speed
                    confidence *= np.exp(-z_score * z_score / 2)
        

        env_factor = self.learning_data['environment_factors'].get('overall', 1.0)
        confidence *= env_factor
        
        return confidence

    def update_environment_factors(self):

        pass

    def update_tracks(self, center_x, center_y, cls):
        if self.roi_points is not None:
            if cv2.pointPolygonTest(self.roi_points, (center_x, center_y), False) < 0:
                return None
        

        box_center = (center_x, center_y)
        track_id = None
        

        for vid, track in self.tracks.items():
            if len(track) > 0:
                last_pos = track[-1][:2]
                distance = np.sqrt((box_center[0] - last_pos[0])**2 + 
                                 (box_center[1] - last_pos[1])**2)
                if distance < 50:  
                    track_id = vid
                    break
        

        if track_id is None:
            track_id = f"vehicle_{self.vehicle_counter}"
            self.vehicle_counter += 1
            self.vehicle_types[track_id] = self.class_names.get(cls, 'Unknown')
        

        self.tracks[track_id].append((center_x, center_y, time.time()))
        
        return track_id

    def calibrate(self, frame):
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if self.calibration_step == 1:
                    self.calibration_points.append((x, y))
                    if len(self.calibration_points) == 2:

                        p1, p2 = self.calibration_points
                        self.reference_pixels = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                        

                        if self.reference_length is not None:
                            self.calibration_factor = self.reference_length / self.reference_pixels

                            self.learning_data['calibration_factors'].append(self.calibration_factor)
                            self.calibration_step = 2
                            self.calibration_points = []
                            print("select perspective transformation points(top left, top right, bottom right, bottom left)")
                elif self.calibration_step == 2:
                    self.perspective_points.append((x, y))
                    if len(self.perspective_points) == 4:

                        src_points = np.float32(self.perspective_points)
                        dst_points = np.float32([[0, 0], [frame.shape[1], 0], 
                                               [frame.shape[1], frame.shape[0]], 
                                               [0, frame.shape[0]]])
                        self.perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                        self.calibration_mode = False
                        cv2.destroyWindow('Calibration')
        
        cv2.namedWindow('Calibration')
        cv2.setMouseCallback('Calibration', mouse_callback)
        
        while self.calibration_mode:
            display_frame = frame.copy()
            

            if self.calibration_step == 1:
                for point in self.calibration_points:
                    cv2.circle(display_frame, point, 5, (0, 255, 0), -1)
                if len(self.calibration_points) == 2:
                    cv2.line(display_frame, self.calibration_points[0], 
                            self.calibration_points[1], (0, 255, 0), 2)
                cv2.putText(display_frame, "select reference object", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                for point in self.perspective_points:
                    cv2.circle(display_frame, point, 5, (0, 255, 0), -1)
                if len(self.perspective_points) >= 2:
                    for i in range(len(self.perspective_points)-1):
                        cv2.line(display_frame, self.perspective_points[i], 
                                self.perspective_points[i+1], (0, 255, 0), 2)
                cv2.putText(display_frame, "select perspective transformation points", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Calibration', display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    def set_reference_length(self, length_meters):
        self.reference_length = length_meters
        
    def set_roi(self, points):
        self.roi_points = np.array(points, np.int32)
        
    def calculate_speed(self, track, track_id=None):
        if len(track) < self.min_track_points or self.calibration_factor is None:
            return 0
            

        recent_track = track[-self.speed_window:] if len(track) > self.speed_window else track
        

        if len(recent_track) >= 2:
            start_point = np.array(recent_track[0][:2])
            end_point = np.array(recent_track[-1][:2])
            direction = np.arctan2(end_point[1] - start_point[1], 
                                 end_point[0] - start_point[0]) * 180 / np.pi
            

            if len(recent_track) >= 3:
                prev_direction = np.arctan2(recent_track[-2][1] - recent_track[-3][1],
                                          recent_track[-2][0] - recent_track[-3][0]) * 180 / np.pi
                if abs(direction - prev_direction) > self.direction_threshold:
                    return 0
        

        total_distance = 0
        for i in range(1, len(recent_track)):

            p1 = np.array([recent_track[i-1][0], recent_track[i-1][1], 1])
            p2 = np.array([recent_track[i][0], recent_track[i][1], 1])
            
            if self.perspective_matrix is not None:
                p1_transformed = np.dot(self.perspective_matrix, p1)
                p2_transformed = np.dot(self.perspective_matrix, p2)
                p1_transformed = p1_transformed[:2] / p1_transformed[2]
                p2_transformed = p2_transformed[:2] / p2_transformed[2]
                
                dx = p2_transformed[0] - p1_transformed[0]
                dy = p2_transformed[1] - p1_transformed[1]
            else:
                dx = recent_track[i][0] - recent_track[i-1][0]
                dy = recent_track[i][1] - recent_track[i-1][1]
            
            distance = np.sqrt(dx*dx + dy*dy)
            total_distance += distance
            

        total_distance_meters = total_distance * self.calibration_factor
        

        total_time = recent_track[-1][2] - recent_track[0][2]
        
        if total_time <= 0:
            return 0
            

        speed = (total_distance_meters / total_time) * 3.6
        

        if track_id and self.learning_data['speed_corrections']:
            vehicle_type = self.vehicle_types.get(track_id, 'Unknown')
            if vehicle_type in self.learning_data['vehicle_type_speeds']:
                type_speeds = self.learning_data['vehicle_type_speeds'][vehicle_type]
                if type_speeds:

                    weights = [self.learning_data['confidence_scores'].get(track_id, 0.5) 
                             for _ in range(len(type_speeds))]
                    weights = np.array(weights) / sum(weights)
                    avg_speed = np.average(type_speeds, weights=weights)
                    

                    if abs(speed - avg_speed) > avg_speed * 0.8:  

                        confidence = self.learning_data['confidence_scores'].get(track_id, 0.5)
                        speed = speed * (1 - confidence) + avg_speed * confidence
        

        if speed < self.min_speed or speed > self.max_speed:
            return 0
            
        return speed
        
    def get_smoothed_speed(self, track_id):
        if len(self.speed_history[track_id]) > 0:

            return np.median(self.speed_history[track_id])
        return 0
        
    def put_chinese_text(self, img, text, position, color):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        

        draw.text(position, text, font=self.font, fill=color[::-1])  
        

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        

        self.roi_points = np.array([(0, 0), (width, 0), (width, height), (0, height)], np.int32)
        

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_video_path = os.path.join(self.output_dir, f"processed_video_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        

        ret, frame = cap.read()
        if ret:
            print("enter reference object length：")
            reference_length = float(input())
            self.set_reference_length(reference_length)
            
            print("select reference object")
            self.calibration_mode = True
            self.calibrate(frame)
            

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            current_time = frame_count / fps
            

            with open(os.devnull, 'w', encoding='utf-8') as f:
                old_stdout = sys.stdout
                sys.stdout = f

                try:

                    if isinstance(frame, np.ndarray):

                        results = self.model(frame, verbose=False)
                    else:
                        results = self.model(np.array(frame), verbose=False)
                    

                    filtered_results = []
                    for result in results:
                        boxes = []
                        for box in result.boxes:
                            cls_val = int(box.cls[0])
                            if cls_val in [2, 3, 5, 7]:  
                                boxes.append(box)
                        result.boxes = boxes
                        filtered_results.append(result)
                    
                    results = filtered_results
                    
                except Exception as e:
                    sys.stdout = old_stdout
                    continue
                sys.stdout = old_stdout
            

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    if conf > 0.5:  

                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        

                        track_id = self.update_tracks(center_x, center_y, cls)
                        

                        if track_id in self.tracks and len(self.tracks[track_id]) >= self.min_track_points:
                            current_speed = self.calculate_speed(self.tracks[track_id], track_id)
                            if current_speed > 0:
                                self.current_speeds[track_id] = current_speed
                                self.speed_history[track_id].append(current_speed)

                                self.speeds[track_id].append(current_speed)
                                

                                self.update_learning_data(track_id, current_speed, 
                                                        self.vehicle_types.get(track_id, 'Unknown'))
                        

                        smoothed_speed = self.get_smoothed_speed(track_id)
                        

                        if smoothed_speed > 140.0:  
                            box_color = (0, 0, 255)  
                            speed_text = f"over speeding: {smoothed_speed:.1f} km/h"
                            print("vehicle over speeding! vehicle id: ", track_id)
                        else:
                            box_color = (0, 255, 0)  
                            speed_text = f"{smoothed_speed:.1f} km/h"
                        

                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                        

                        frame = self.put_chinese_text(frame, speed_text, (x1, y1 - 25), (255, 255, 255))
            

            cv2.polylines(frame, [self.roi_points], True, (0, 255, 0), 2)
            

            cv2.imshow('Vehicle Speed Detection', frame)
            

            self.video_writer.write(frame)
            

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        self.video_writer.release()
        cv2.destroyAllWindows()
        

        self.save_learning_data()
        

        self.save_speed_statistics()
        

        root = tk.Tk()
        root.withdraw()
        root.destroy()

    def save_speed_statistics(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"vehicle_speeds_{timestamp}.txt")
        

        speed_stats = self.get_speed_statistics()
        

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("max speed：\n")
            f.write("-" * 50 + "\n")
            

            sorted_stats = sorted(speed_stats.items(), key=lambda x: x[1], reverse=True)
            
            for track_id, max_speed in sorted_stats:
                f.write(f"vehicle id: {track_id}\n")
                f.write(f"vehicle type: {self.vehicle_types.get(track_id, 'Unknown')}\n")
                f.write(f"max speed: {max_speed:.1f} km/h\n")
                f.write("-" * 50 + "\n")
        

        root = tk.Tk()
        root.withdraw()  
        root.destroy()
        
    def get_speed_statistics(self):
        speed_stats = {}
        for track_id, speeds in self.speeds.items():
            if speeds:

                valid_speeds = [s for s in speeds if s > 0 and s <= 200]
                if valid_speeds:
                    speed_stats[track_id] = max(valid_speeds)
        return speed_stats

def main():

    detector = VehicleSpeedDetector('video.mp4')
    

    detector.process_video()
    

    detector.save_speed_statistics()
    

    speed_stats = detector.get_speed_statistics()

    root = tk.Tk()
    root.withdraw()  
    root.destroy()

if __name__ == "__main__":
    main() 