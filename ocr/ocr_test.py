import cv2
import easyocr

def main():
    # Initialize camera (0 = default webcam)
    cap = cv2.VideoCapture(0)
    reader = easyocr.Reader(['en','tr'])  # You can add more languages like ['en', 'tr']

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for speed (optional)
        resized = cv2.resize(frame, (640, 480))

        # Perform OCR on the frame
        results = reader.readtext(resized)

        # Draw boxes and text
        for (bbox, text, confidence) in results:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))
            cv2.rectangle(resized, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(resized, text, (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Display frame
        cv2.imshow("Live OCR", resized)

        # Print extracted texts
        if results:
            print("Detected texts:", [text for (_, text, _) in results])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
