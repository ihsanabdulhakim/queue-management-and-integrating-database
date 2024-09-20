import os
import cv2
import shutil
import numpy as np
from ultralytics import YOLO, solutions
import firebase_admin
from firebase_admin import credentials, firestore

# Firebase initialization
cred = credentials.Certificate('thejsonAPIfromfirecloudgoogle.json') #JSON Format file API that you gain from google cloud console firestore
firebase_admin.initialize_app(cred)
db = firestore.client()

# Function to ensure directory exists for saving cropped images
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to clear the directory
def clear_dir(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

# Check if a point is inside a polygon
def is_point_in_polygon(x, y, polygon):
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), (x, y), False) >= 0

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Open the video file
cap = cv2.VideoCapture("./video.avi")
assert cap.isOpened(), "Error reading video file"

# Set video properties
w, h = 1920, 1080
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Set up video writer to save the processed video
video_writer = cv2.VideoWriter("resultofqueue-firebase.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Define queue region for tracking
factors = 550
factors2 = 180
queue_region = [(20, 10), (850 + factors2, 10), (850 + factors2, 360 + factors), (20, 360 + factors)]

# Initialize QueueManager for handling queue region and IDs
queue = solutions.QueueManager(
    classes_names=model.names,
    reg_pts=queue_region,
    line_thickness=3,
    fontsize=1.0,
    region_color=(255, 144, 31),
)

# Directory to save cropped images
cropped_dir = "cropped_images"
ensure_dir(cropped_dir)
clear_dir(cropped_dir)  # Clear the directory

# Set to keep track of seen IDs
seen_ids = set()
frame_number = 0

# Create a named window and set it to be resizable
cv2.namedWindow('queuemanagement', cv2.WINDOW_NORMAL)

while cap.isOpened():
    success, im0 = cap.read()
    frame_number += 1

    if success:
        # Resize the frame to 1920x1080
        im0 = cv2.resize(im0, (w, h))

        # Create a copy of the frame for processing
        im0_copy = im0.copy()

        # Track objects in the frame
        results = model.track(im0_copy, show=False, persist=True, verbose=False)

        # Process the queue and add ID information to the frame
        out = queue.process_queue(im0_copy, results)

        # Create a green transparent overlay
        overlay = im0_copy.copy()
        alpha = 0.5  # Transparency factor
        cv2.fillPoly(overlay, np.array([queue_region], dtype=np.int32), (0, 255, 0))
        cv2.addWeighted(overlay, alpha, im0_copy, 1 - alpha, 0, im0_copy)

        # Iterate through the tracked objects in results
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls)
                    obj_id = int(box.id)

                    # Check if the object is a person and is a new ID
                    if result.names[cls_id] == 'person' and obj_id not in seen_ids:
                        # Calculate the timestamp
                        timestamp = frame_number / fps
                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # Check if the center of the bounding box is inside the queue region
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        if is_point_in_polygon(center_x, center_y, queue_region):
                            print(f"person #{obj_id} appeared at {timestamp:.2f} seconds with bounding box, Coordinates ({x1},{y1}) and ({x2},{y2})")

                            # Crop the person from the original frame
                            person_crop = im0[y1:y2, x1:x2]

                            # Save the cropped image
                            filename = os.path.join(cropped_dir, f"person_{obj_id}_frame_{frame_number}.jpg")
                            cv2.imwrite(filename, person_crop)

                            # Insert data into Firebase Firestore
                            doc_ref = db.collection('table-queue-management').document(f'person_{obj_id}_frame_{frame_number}')
                            doc_ref.set({
                                'obj_id': obj_id,
                                'timestamps': timestamp,
                                'x1': x1,
                                'y1': y1,
                                'x2': x2,
                                'y2': y2
                            })

                            # Add the ID to seen IDs
                            seen_ids.add(obj_id)

        # Write the processed frame to the output video
        video_writer.write(im0_copy)

        # Show the frame with queue region in a new window
        cv2.imshow('queuemanagement', im0_copy)

        # Resize the window to match the video frame size
        cv2.resizeWindow('queuemanagement', w, h)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    print("Video frame is empty or video processing has been successfully completed.")
    break

# Release the video capture and writer objects
cap.release()
video_writer.release()
cv2.destroyAllWindows()
