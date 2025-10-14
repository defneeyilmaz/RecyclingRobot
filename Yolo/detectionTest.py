import time
import cv2
from ultralytics import YOLO


def main(
        model_name: str = "yolo12n.pt",  # n/s/m/l/x seçebilirsin
        cam_index: int = 0,  # 0: varsayılan kamera, harici kamera için 1 vb.
        imgsz: int = 640,  # giriş boyutu (düşürürsen hızlanır)
        conf: float = 0.25,  # güven eşiği
        iou: float = 0.45,  # NMS IoU eşiği
        show_labels: bool = True,  # sınıf ve skor yazıları
        show_fps: bool = True  # sol üstte FPS göstergesi
):
    model = YOLO(model_name)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera açılamadı (index={cam_index}).")

    prev_time = time.time()
    frame_count = 0
    fps = 0.0

    window_name = "YOLOv12 - Real-time (press 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Kare okunamadı, döngüden çıkılıyor.")
                break

            results = model.predict(
                source=frame,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                verbose=False
            )

            plotted = results[0].plot(labels=show_labels)

            frame_count += 1
            if frame_count >= 10:
                now = time.time()
                fps = frame_count / (now - prev_time)
                prev_time = now
                frame_count = 0

            if show_fps:
                text = f"FPS: {fps:.1f}"
                cv2.putText(plotted, text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow(window_name, plotted)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(
        model_name="yolo12n.pt",
        cam_index=0,
        imgsz=640,
        conf=0.25,
        iou=0.45,
        show_labels=True,
        show_fps=True
    )
else :
    print(__name__)
