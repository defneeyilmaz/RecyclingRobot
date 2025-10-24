import time
import cv2
from inference import get_model

def main(
        model_id: str = "iue_garbage_batch-1-2-p0c5a/2",  # your Roboflow model
        api_key: str = "rUcFcwUxYmLAGWg9FkG9",  # replace with your Roboflow key
        cam_index: int = 0,
        imgsz: int = 640,
        conf: float = 0.25,
        show_labels: bool = True,
        show_fps: bool = True
):
    # Initialize Roboflow model
    model = get_model(model_id=model_id, api_key=api_key)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Camera could not be opened (index={cam_index}).")

    prev_time = time.time()
    frame_count = 0
    fps = 0.0

    window_name = f"Roboflow Model - {model_id} (Press 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame could not be read. Exiting.")
                break

            # Run inference
            results = model.infer(frame, confidence=conf)

            # Draw detections
            for prediction in results['predictions']:
                x, y, w, h = int(prediction['x']), int(prediction['y']), int(prediction['width']), int(prediction['height'])
                class_name = prediction['class']
                conf_score = prediction['confidence']

                # Draw bounding box
                cv2.rectangle(frame, (x - w // 2, y - h // 2),
                              (x + w // 2, y + h // 2), (0, 255, 0), 2)

                if show_labels:
                    label = f"{class_name} ({conf_score:.2f})"
                    cv2.putText(frame, label, (x - w // 2, y - h // 2 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Compute FPS
            frame_count += 1
            if frame_count >= 10:
                now = time.time()
                fps = frame_count / (now - prev_time)
                prev_time = now
                frame_count = 0

            if show_fps:
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
