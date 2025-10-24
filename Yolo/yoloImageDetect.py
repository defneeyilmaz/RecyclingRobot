import cv2
from ultralytics import YOLO

def main(
        model_name: str = "best.pt",  # model path
        image_path: str = "image.jpg",  # path to your image
        imgsz: int = 640,  # input size
        conf: float = 0.25,  # confidence threshold
        iou: float = 0.45,  # IoU threshold for NMS
        show_labels: bool = True,  # show labels on boxes
        save_result: bool = True  # save result image
):
    # Load model
    model = YOLO(model_name)

    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Görsel bulunamadı: {image_path}")

    # Run inference
    results = model.predict(
        source=frame,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=False
    )

    # Plot detections
    plotted = results[0].plot(labels=show_labels)

    # Show result
    window_name = "YOLOv12 - Image Detection"
    cv2.imshow(window_name, plotted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally save
    if save_result:
        out_path = "result.jpg"
        cv2.imwrite(out_path, plotted)
        print(f"Sonuç kaydedildi: {out_path}")


if __name__ == "__main__":
    main(
        model_name="best.pt",
        image_path=r"E:\PythonProjects\RecyclingRobot\Yolo\datasets\iue_yolo_ready\test\images\capture_20251022_151308_914721_jpg.rf.06111de6059b2d4743c850c43d295970.jpg",  # change this to your image
        imgsz=640,
        conf=0.25,
        iou=0.45,
        show_labels=True,
        save_result=True
    )
