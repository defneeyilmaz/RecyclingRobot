from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/classify/train2/weights/best.pt")

def classify(path):
    results = model.predict(source=path)
    for r in results:
        print(r.probs.top1)  # predicted class index
        print(r.names[r.probs.top1])  # predicted class name
        print(r.probs)  # probabilities for all classes

classify("Test/test_cardboard001.jpg")
classify("Test/test_cardboard002.jpg")
classify("Test/test_cardboard003.jpg")
classify("Test/test_cardboard004.jpg")