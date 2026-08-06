import cv2
from ultralytics import YOLO
cap = cv2.VideoCapture(0)  # Open the default camera (0)
# Load a model
model = YOLO("yolov8n.pt")  # Load a pretrained YOLO
while True:
    ret, frame = cap.read()  # Read a frame from the camera
    if not ret:
        break  # Exit the loop if the frame is not read correctly
    results = model(frame)  # Perform inference on the frame
    annotated_frame = results[0].plot()  # Annotate the frame with detection results
    cv2.imshow("Live Camera Feed", annotated_frame)  # Display the annotated frame
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit the loop if 'q' is pressed
        break
cap.release()  # Release the camera
cv2.destroyAllWindows()  # Close all OpenCV windows 