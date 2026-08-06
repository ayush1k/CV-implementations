import cv2
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # Load a pretrained YOLO
image = cv2.imread("dog.jpeg")  # Read an image
results = model(image)  # Perform inference on the image
annotated_image = results[0].plot()
cv2.imshow("Annotated Image", annotated_image)  # Display the annotated image
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()  # Close all OpenCV windows