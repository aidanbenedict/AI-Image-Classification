from roboflow import Roboflow
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt

# Check if MPS (Apple Silicon GPU) is available
print(f"MPS acceleration available: {torch.backends.mps.is_available()}")

# Load data
rf = Roboflow(api_key="nlJqijhN8BbaqBNzFACQ")
project = rf.workspace("material-identification").project("garbage-classification-3")
dataset = project.version(2).download(
    "yolov8")

# Load smallest model
model = YOLO("yolov8n.pt")  # Nano version

# Fast training configuration
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=10,      # Only 10 epochs for quick testing
    batch=8,        # Smaller batches to avoid memory issues
    imgsz=320,      # Smaller images = faster processing
    device="cpu",   # Use Apple Silicon GPU (falls back to CPU if unavailable)
    workers=0,      # Disable multiprocessing (safer on macOS)
    augment=False,  # Disable time-consuming augmentations
    verbose=True    # Show progress updates
)

# Quick evaluation
metrics = model.val()
print(f"Quick mAP score: {metrics.box.map:.2f}")

# Test prediction
input_image = "Desktop/images.jpeg"
results = model.predict(input_image, save=True, imgsz=320)

# Show results
print("\nPrediction results:")
results[0].show()

# Plot using saved image
predicted_image = f"runs/detect/predict/{input_image.split('/')[-1]}"
img = plt.imread(predicted_image)
plt.imshow(img)
plt.axis('off')
plt.show()

print("\nTraining complete! Check the plots above for results.")