import cv2
from tests.test_python import test_predict_img
from ultralytics import YOLO

def main(
        model_name: str = "",  # model path
        image_path: str = "",  # path to your image
        imgsz: int = 640,  # input size
        conf: float = 0.25,  # confidence threshold
        iou: float = 0.45,  # IoU threshold for NMS
        show_labels: bool = True,  # show labels on boxes
        save_result: bool = True,  # save result image
        out_path: str = "result.jpg",
        display_window: bool = False,
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

    # Show result # Disabled for multiple photos
    if display_window:
        window_name = model_name
        cv2.imshow(window_name, plotted)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Optionally save
    if save_result:
        cv2.imwrite(out_path, plotted)
        print(f"Sonuç kaydedildi: {out_path}")


if __name__ == "__main__":
    testimgpath = r"E:\PythonProjects\RecyclingRobot\utils\captures\capture_20251024_145710_123671.jpg"
    main(
        model_name=r"E:\PythonProjects\RecyclingRobot\Yolo\runs\detect\yolo12s-162a\weights\best.pt",
        image_path=testimgpath,  # change this to your image
        show_labels=True,
        conf=0.45,
        save_result=True,
        out_path="Results/yolov12sresult.jpg"
    )
    main(
        model_name=r"E:\PythonProjects\RecyclingRobot\Yolo\runs\detect\yolo12n-162av1\weights\best.pt",
        image_path=testimgpath,  # change this to your image
        show_labels=True,
        conf=0.45,
        save_result=True,
        out_path="Results/yolov12nresult.jpg"
    )
