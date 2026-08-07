import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt")  # Load a pretrained YOLOv8 model
cap = cv2.VideoCapture('bottles.mp4')  # Open the video file

unique_ids = set()  # Set to store unique object IDs
while True:
    ret, frame = cap.read()  # Read a frame from the video
    results = model.track(frame, classes=[39], persist=True, verbose=False)  # Track objects of class 39 (bottle)
    annotated_frame = results[0].plot()  # Annotate the frame with tracking results

    if results[0].boxes and results[0].boxes.id is not None:  # Check if there are any tracked objects
        ids = results[0].boxes.id.numpy()  # Get the IDs of the tracked objects
        for oid in ids:
            unique_ids.add(oid)  # Add the object ID to the set of unique IDs
        cv2.putText(annotated_frame, f'Unique Bottles: {len(unique_ids)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Display the count of unique bottles
        cv2.imshow('Object Counting', annotated_frame)  # Show the annotated frame

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit the loop if 'q' is pressed
        break 

    cap.release()  # Release the video capture object
    cv2.destroyAllWindows()  # Close all OpenCV windows