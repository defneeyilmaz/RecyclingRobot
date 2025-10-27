import time
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO


def main(
        model_name: str = "best.pt",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        show_labels: bool = True,
        show_fps: bool = True
):
    # Load YOLO model
    model = YOLO(model_name)

    # --- Configure RealSense pipeline ---
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable color stream
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    prev_time = time.time()
    frame_count = 0
    fps = 0.0

    window_name = "YOLOv11+ RealSense (press 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            # Wait for a coherent pair of frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert image to numpy array
            frame = cv2.cvtColor(
                np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Run YOLO inference
            results = model.predict(
                source=frame_bgr,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                verbose=False
            )

            plotted = results[0].plot(labels=show_labels)

            # Calculate FPS every 10 frames
            frame_count += 1
            if frame_count >= 10:
                now = time.time()
                fps = frame_count / (now - prev_time)
                prev_time = now
                frame_count = 0

            if show_fps:
                text = f"FPS: {fps:.1f}"
                cv2.putText(plotted, text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 255, 0), 2, cv2.LINE_AA)

            # Show frame
            cv2.imshow(window_name, plotted)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import numpy as np

    main(
        model_name="best.pt",
        imgsz=640,
        conf=0.25,
        iou=0.45,
        show_labels=True,
        show_fps=True
    )
else:
    print(__name__)
