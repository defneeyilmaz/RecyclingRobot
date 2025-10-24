import os
import sys
from datetime import datetime
import cv2
import numpy as np
import pyrealsense2 as rs


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def open_realsense_camera():
    """Initialize Intel RealSense camera pipeline."""
    pipeline = rs.pipeline()
    config = rs.config()

    # Automatically find connected device
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_name = device.get_info(rs.camera_info.name)
    print(f"✅ Found RealSense device: {device_name}")

    # Enable color stream (and optionally depth)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # optional

    # Start streaming
    pipeline.start(config)
    return pipeline


def main(save_dir: str = "captures"):
    ensure_dir(save_dir)

    try:
        pipeline = open_realsense_camera()
    except Exception as e:
        print(f"❌ RealSense camera could not be initialized: {e}")
        return

    print("🛈 Controls:")
    print("   - [Space] = Capture and save image")
    print("   - [q]     = Quit")

    while True:
        frames = pipeline.wait_for_frames()

        # Get color frame
        color_frame = frames.get_color_frame()
        if not color_frame:
            print("Could not get color frame.")
            continue

        # Convert to numpy array for OpenCV
        color_image = np.asanyarray(color_frame.get_data())

        # Show the frame
        cv2.imshow("Intel RealSense - Press [Space] to capture, [q] to quit", color_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32:  # Space
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"capture_{ts}.jpg"
            path = os.path.join(save_dir, filename)
            success = cv2.imwrite(path, color_image)
            if success:
                print(f"Saved: {path}")
            else:
                print("❌ Failed to save image!")

    # Cleanup
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Camera closed.")


if __name__ == "__main__":
    main()
