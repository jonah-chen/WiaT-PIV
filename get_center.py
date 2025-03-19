import cv2

def main():
    video_file = "test.MOV"
    cap = cv2.VideoCapture(video_file)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_file}")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read a frame from the video.")
        return

    # Get frame dimensions: height, width, and number of channels.
    height, width, channels = frame.shape

    # Calculate the center pixel coordinates
    center_x = width // 2
    center_y = height // 2

    print(f"Center pixel coordinates of {video_file}: (x, y) = ({center_x}, {center_y})")

    cap.release()

if __name__ == "__main__":
    main()
