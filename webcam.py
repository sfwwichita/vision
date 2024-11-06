# import numpy as np
# import cv2
# import cvlib as cv
# from cvlib.object_detection import draw_bbox

# #Currently trying to get python version 3.11 working with pyenv so that we can install tensorflow

# cap = cv2.VideoCapture()

# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     # Detect objects and draw on screen
#     bbox, label, conf = cv.detect_common_objects(frame)
#     output_image = draw_bbox(frame, bbox, label, conf)

#     cv2.imshow('output',output_image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
from goprocam import GoProCamera, constants

# gopro = GoProCamera.GoPro(constants.gpcontrol)

# gopro.overview()

# gopro.stream()

# gopro.downloadLastMedia(gopro.take_photo(2), custom_filename='Realtry1.JPG')
# gopro.delete("last")

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

#RTLS tag - Real time location tracking - sewio tag

# Load class names from coco file
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Load video
#video_path = "Realtry1.JPG"

cap = cv2.VideoCapture(0)

# Initialize variables for frame extraction
frame_rate = 1  # Extract one frame per minute
frame_count = 0

lst = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, dsize=(1000, 700))

    frame_count += 1

    lst_inner = []

    if frame_count % frame_rate == 0:
        # Preprocess the frame for YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (10, 10, 10), True, crop=False)
        net.setInput(blob)
        outs = net.forward(net.getUnconnectedOutLayersNames())

        # Process YOLO output
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    # Extract detection details
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    width = int(detection[2] * frame.shape[1])
                    height = int(detection[3] * frame.shape[0])

                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    # Draw a box and label on the frame
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    label = f"{classes[class_id]}: {confidence:.2f}"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    lst_inner.append([x, y, label])

        # Display the frame with detections
        cv2.imshow("DISPLAYING: FRAME | Detections", frame)
        cv2.waitKey(40)
    if (len(lst_inner) != 0):
        lst.append(lst_inner)

cap.release()
cv2.destroyAllWindows()
print(lst)