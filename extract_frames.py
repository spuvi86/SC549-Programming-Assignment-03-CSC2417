import cv2
import os
import glob

def extract_frames(video_dir, output_dir, frames_per_video=10):
    """
    Extracts a fixed number of frames from each video in the directory.
    
    Args:
        video_dir (str): Directory containing video files.
        output_dir (str): Directory to save extracted images.
        frames_per_video (int): Number of frames to extract per video.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    video_extensions = ['*.mp4', '*.avi', '*.mov']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_dir, ext)))
        
    print(f"Found {len(video_files)} videos in {video_dir}")
    
    for video_path in video_files:
        filename = os.path.basename(video_path)
        name_no_ext = os.path.splitext(filename)[0]
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            print(f"Skipping {filename}: No frames found.")
            continue
            
        # Calculate interval to get evenly spaced frames
        interval = max(1, total_frames // frames_per_video)
        
        count = 0
        saved_count = 0
        
        while cap.isOpened() and saved_count < frames_per_video:
            ret, frame = cap.read()
            if not ret:
                break
                
            if count % interval == 0:
                output_filename = f"{name_no_ext}_frame_{count}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, frame)
                saved_count += 1
                
            count += 1
            
        cap.release()
        print(f"Extracted {saved_count} frames from {filename}")
        
    print(f"\nExtraction complete. Images saved to: {output_dir}")

if __name__ == "__main__":
    VIDEO_DIR = 'video_clips'
    OUTPUT_DIR = os.path.join('datasets', 'sports', 'images', 'val')
    
    extract_frames(VIDEO_DIR, OUTPUT_DIR, frames_per_video=5)
