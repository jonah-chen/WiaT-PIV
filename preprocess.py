import cv2
import argparse
import os

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess a MOV file by cropping frames to 600<y<900 and 300<x<700."
    )
    parser.add_argument("input_file", help="Path to the input MOV file.")
    parser.add_argument(
        "output_file",
        nargs="?",
        default="cropped.mov",
        help="Path to the output cropped video file (default: cropped.mov).",
    )
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    # Open the input video
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_file}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define cropping coordinates: y from 600 to 900, x from 300 to 700
    y_min, y_max = 600, 960
    x_min, x_max = 540-240, 540+240

    # Check if cropping region is within the frame dimensions
    if y_max > frame_height or x_max > frame_width:
        print("Error: Cropping region exceeds frame dimensions of the input video.")
        cap.release()
        return

    # Define new dimensions after cropping
    new_width = x_max - x_min
    new_height = y_max - y_min

    # Define the codec and create VideoWriter object
    # 'mp4v' codec is generally compatible; adjust if necessary.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (new_width, new_height))

    print(f"Processing video: {input_file}")
    print(f"Original dimensions: {frame_width}x{frame_height}, FPS: {fps}")
    print(f"Cropping region: x [{x_min}, {x_max}) and y [{y_min}, {y_max}), new dimensions: {new_width}x{new_height}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop the frame to the desired region
        cropped_frame = frame[y_min:y_max, x_min:x_max]
        out.write(cropped_frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"Processed {frame_count} frames.")
    print(f"Cropped video saved to {output_file}")

if __name__ == "__main__":
    main()
