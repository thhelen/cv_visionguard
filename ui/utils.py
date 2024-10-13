import cv2
import os

def save_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1
        if frame_count %20==0:
            image_name = f"frame_{frame_count:04d}.png"
            image_path = os.path.join(output_folder, image_name)

            cv2.imwrite(image_path, frame)

    cap.release()
    print(f"{frame_count} frames saved to {output_folder}")

if __name__ == "__main__":
    video_path = "/Users/shunya/Project/visionguard/ui/videos/IMG_0311.MOV"  # Replace with the path to your video file
    output_folder = "frames"  # Replace with the desired output folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    save_frames(video_path, output_folder)
