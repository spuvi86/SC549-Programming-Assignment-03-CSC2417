from ultralytics import YOLO
import os
import glob

def generate_labels(images_dir, labels_dir, model_path='yolo11n-pose.pt'):
    """
    Generates pseudo-labels for images using a pre-trained YOLO model.
    Saves validation labels in YOLO format (.txt).
    
    Args:
        images_dir (str): Directory containing images.
        labels_dir (str): Directory to save generated labels.
        model_path (str): Path to the YOLO model.
    """
    model = YOLO(model_path)
    
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)
        
    image_files = glob.glob(os.path.join(images_dir, '*.jpg'))
    print(f"Found {len(image_files)} images in {images_dir}")
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        save_path = os.path.join(labels_dir, txt_filename)
        
        # Run inference
        results = model.predict(img_path, conf=0.25, verbose=False) # Use lower conf to catch more potential objects
        
        # Save results in YOLO format
        results[0].save_txt(save_path)
        
    print(f"Generated {len(image_files)} label files in {labels_dir}")

if __name__ == "__main__":
    IMAGES_DIR = os.path.join('datasets', 'sports', 'images', 'val')
    LABELS_DIR = os.path.join('datasets', 'sports', 'labels', 'val')
    
    generate_labels(IMAGES_DIR, LABELS_DIR)
