import os
import sys

def extract_frames(video_path, output_dir, frames_per_second=3):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")

    # Calculate frame sampling interval based on the frames_per_second parameter
    interval = fps / frames_per_second
    
    count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Sample frames based on the target sampling interval
        if count >= saved_count * interval:
            frame_name = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_name, frame)
            saved_count += 1
            
        count += 1

    cap.release()
    print(f"Done! Extracted {saved_count} frames to {output_dir}")

if __name__ == "__main__":
    video_file = "video.mp4"
    output_folder = "frames"
    
    try:
        import cv2
    except ImportError:
        print("Error: opencv-python not found. Please install it using:")
        print("pip install opencv-python")
        sys.exit(1)

    if not os.path.exists(video_file):
        print(f"Error: {video_file} not found in current directory.")
    else:
        extract_frames(video_file, output_folder)
