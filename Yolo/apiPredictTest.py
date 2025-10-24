import cv2
from inference import get_model

def predict_image(
        image_path: str,
        model_id: str = "iue_garbage_batch-1-2-p0c5a/2",
        api_key: str = "rUcFcwUxYmLAGWg9FkG9",
        conf: float = 0.25,
        show_labels: bool = True,
        save_output: bool = True
):
    # Load Roboflow model
    model = get_model(model_id=model_id, api_key=api_key)

    # Read input image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Run inference
    results = model.infer(image, confidence=conf)

    # ✅ results is a list, we take the first item
    predictions = results[0].predictions

    # Draw predictions
    for pred in predictions:
        x, y = int(pred.x), int(pred.y)
        w, h = int(pred.width), int(pred.height)
        class_name = pred.class_name
        conf_score = pred.confidence

        # Draw bounding box
        cv2.rectangle(image, (x - w // 2, y - h // 2),
                      (x + w // 2, y + h // 2), (0, 255, 0), 2)

        if show_labels:
            label = f"{class_name} ({conf_score:.2f})"
            cv2.putText(image, label, (x - w // 2, y - h // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show result
    cv2.imshow("Prediction Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save output
    if save_output:
        output_path = "predicted_output.jpg"
        cv2.imwrite(output_path, image)
        print(f"✅ Saved prediction to {output_path}")


if __name__ == "__main__":
    test_image = r"E:\PythonProjects\RecyclingRobot\Yolo\datasets\iue_yolo_ready\test\images\capture_20251022_151308_914721_jpg.rf.06111de6059b2d4743c850c43d295970.jpg"
    predict_image(test_image)
