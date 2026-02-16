import cv2
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import time
import numpy as np

class SportsTracker:
    def __init__(self, det_model_path='yolo11n.pt', pose_model_path='yolo11n-pose.pt'):
        """
        Initializes the detection and keypoint models.
        Satisfies Assignment Requirements 1 & 2.
        """
        self.detector = YOLO(det_model_path)  # Player Detection 
        self.pose_estimator = YOLO(pose_model_path)  # Keypoint Detection 
        
        self.inference_times = []
        self.conf_scores = []
        self.detection_counts = []

    def process_collection(self, input_dir, output_dir):
       
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        video_files = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi'))]
        
        for video_file in video_files:
            video_path = os.path.join(input_dir, video_file)
            print(f"Processing: {video_file}")
            self._run_inference(video_path, output_dir, video_file)
            
        self.plot_performance_metrics(output_dir)

    def _run_inference(self, video_path, output_dir, filename):
        cap = cv2.VideoCapture(video_path)
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
        
        ret, frame = cap.read()
        if ret:
            
            
            start_time = time.time()
            results = self.pose_estimator.predict(frame, conf=0.5)
            end_time = time.time()
            
            inference_time = (end_time - start_time) * 1000  # ms
            self.inference_times.append(inference_time)
            
            if results[0].boxes:
                confs = results[0].boxes.conf.cpu().numpy()
                if len(confs) > 0:
                    self.conf_scores.extend(confs)
                    self.detection_counts.append(len(confs))
                else:
                    self.detection_counts.append(0)
            else:
                 self.detection_counts.append(0)
            
            
            annotated_frame = results[0].plot()
            output_path = os.path.join(output_dir, f"result_{filename.split('.')[0]}.jpg")
            cv2.imwrite(output_path, annotated_frame)
            
        cap.release()

    def plot_performance_metrics(self, output_dir):
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.inference_times, marker='o', linestyle='-', color='b')
        plt.title('Inference Speed per Video')
        plt.xlabel('Video Index')
        plt.ylabel('Inference Time (ms)')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'inference_speed_comparison.png'))
        plt.close()

        
        plt.figure(figsize=(10, 6))
        plt.hist(self.conf_scores, bins=10, color='g', alpha=0.7)
        plt.title('Confidence Score Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'confidence_score_distribution.png'))
        plt.close()

        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(self.detection_counts)), self.detection_counts, color='r', alpha=0.7)
        plt.title('Detections per Video')
        plt.xlabel('Video Index')
        plt.ylabel('Number of Detections')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'detections_per_frame.png'))
        plt.close()
        
        print(f"Performance plots saved to {output_dir}")

    def evaluate_model(self, data_yaml):
        
        print(f"Starting evaluation on dataset: {data_yaml}")
        try:
            metrics = self.pose_estimator.val(data=data_yaml)
            
            print("\nEvaluation Complete!")
            print(f"Results saved to: {metrics.save_dir}")
            print(f"Mean Average Precision (mAP50-95): {metrics.box.map}")
            print(f"Precision: {metrics.box.mp}")
            print(f"Recall: {metrics.box.mr}")
            
        except Exception as e:
            print(f"\n[ERROR] Evaluation failed: {e}")
            print("Ensure you have a valid .yaml dataset file referenced.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sports Series Computer Vision Assignment')
    parser.add_argument('--mode', type=str, default='process', choices=['process', 'evaluate'], 
                        help='Mode: "process" for video inference, "evaluate" for model metrics (requires dataset)')
    parser.add_argument('--data', type=str, default='coco8-pose.yaml', 
                        help='Path to dataset.yaml for evaluation mode')
    
    args = parser.parse_args()

    
    INPUT_VIDEOS = 'video_clips' 
    OUTPUT_RESULTS = 'output'
    
    tracker = SportsTracker()
    
    if args.mode == 'process':
        tracker.process_collection(INPUT_VIDEOS, OUTPUT_RESULTS)
    elif args.mode == 'evaluate':
        tracker.evaluate_model(args.data)