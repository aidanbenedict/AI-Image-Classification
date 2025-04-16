from ultralytics import YOLO

# Load the best saved model (automatically saved in runs/detect/train/weights/)
model = YOLO("runs/detect/train5/weights/best.pt")

# Predict on a single image
results = model.predict(
    source="bigmac.jpeg",  # Path to your test image
    conf=0.1,    # Confidence threshold (adjust to filter weak detections)
    save=True,   # Save the results
    show_labels=True,  # Show class labels
    show_conf=True,    # Show confidence scores
    imgsz=320          # Match the training image size
)

# Show results inline (optional)
results[0].show()

import matplotlib.pyplot as plt

# Load and plot the saved prediction
predicted_img = plt.imread("runs/detect/predict")
plt.imshow(predicted_img)
plt.axis("off")
plt.show()