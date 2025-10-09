import torch

# Model loading
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # Can be 'yolov5n' - 'yolov5x6', or 'custom'

# Inference on images
img = "https://ultralytics.com/images/zidane.jpg"  # Can be a file, Path, PIL, OpenCV, numpy, or list of images

# Run inference
results = model(img)

# Display results
results.print()  # Other options: .show(), .save(), .crop(), .pandas(), etc. Explore these in the Predict mode documentation.

# Extract and display bounding boxes, classes, and confidence scores
df = results.pandas().xyxy[0]  # DataFrame with xmin, ymin, xmax, ymax, confidence, class, name
print(df)

# Show the image with bounding boxes and labels
# results.show()
results.save()