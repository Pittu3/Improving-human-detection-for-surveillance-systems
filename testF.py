import cv2
import numpy as np
import time
import math
from ultralytics import YOLO  # Import YOLOv8 model from the ultralytics package

# Define video source (either webcam or video file)
videoSource = 0
cap = cv2.VideoCapture(videoSource)
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

time.sleep(2)  # Give the camera some time to warm up

# Load YOLOv8 model
model = YOLO("E:/Project/final/yolov8n/yolov8m-face.pt")  # Replace with the correct model path

# Initialize global variables
people = []
personId = 0
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
appendThresh = 80
font = cv2.FONT_HERSHEY_PLAIN

class Person:
    def __init__(self, personId, location):
        self.id = personId
        self.curLocation = location
        self.trajectory = []
        self.flag = 0
        if personId < len(colors):
            self.color = colors[personId]
        else:
            self.color = colors[personId % len(colors)]

    def addPointToTrajectory(self, location):
        self.trajectory.append(location)

def plotTrajectories(frame):
    global people
    for person in people:
        prev_point = None  # Store the previous point
        traj = person.trajectory[-5:] if len(person.trajectory) >= 5 else person.trajectory
        for i in traj:
            x, y = int(i[0]), int(i[1])  # Convert to integers
            color = person.color
            frame = cv2.circle(frame, (x, y), 3, color, cv2.FILLED)
            if prev_point is not None:
                frame = cv2.line(frame, prev_point, (x, y), color, 1)
            prev_point = (x, y)
    return frame

def calcDist(a, b):
    x1, y1 = a
    x2, y2 = b
    return math.sqrt(((x2-x1)*(x2-x1)) + ((y2-y1)*(y2-y1)))

def trackerHandle(curCoords, frame):
    global people, personId

    if people:
        if curCoords:
            if len(people) != len(curCoords):
                if len(people) > len(curCoords):  # Existing person exited
                    people = people[:len(curCoords)]
                elif len(people) < len(curCoords):  # New person added
                    for i in curCoords[len(people):]:
                        createPerson(i)

            for person, coord in zip(people, curCoords):
                dist = calcDist(person.curLocation, coord)
                if dist < appendThresh:
                    person.trajectory.append(coord)
                person.curLocation = coord

        else:
            people = []

    elif curCoords:
        for coord in curCoords:
            createPerson(coord)

    return plotTrajectories(frame)

def createPerson(currentCoord):
    global people, personId
    person = Person(personId, currentCoord)
    people.append(person)
    personId += 1

cv2.namedWindow("Person Face Tracking", cv2.WINDOW_NORMAL)

while True:
    startTime = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))

    # Detect objects in the frame using YOLOv8
    results = model.predict(frame, save=False, conf=0.5)  # Get predictions from YOLOv8

    curCoords = []
    for result in results:  # Loop over each result (each frame's predictions)
        for bbox in result.boxes.xywh:  # xywh: [x_center, y_center, width, height]
            x_center, y_center, width, height = bbox.tolist()
            x = int(x_center - width / 2)
            y = int(y_center - height / 2)
            w = int(width)
            h = int(height)
            
            curCoords.append([x_center, y_center])  # Append detected face center coordinates

            # Draw a rectangle around each detected person
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Person', (x, y - 10), font, 1, (0, 255, 0), 1)

    # Handle the tracking of people
    frame = trackerHandle(curCoords, frame)

    # Count the number of people detected (using the current coordinates)
    peopleFaceCount = len(curCoords)
    print(peopleFaceCount)
    # Count the number of people detected
    peopleFaceCount = len(curCoords)  # Use curCoords length for detected people, or continue using len(people)

    # Display the people face count on the frame
    # Adjusted position and font size
    cv2.putText(frame, f"People Face Count: {peopleFaceCount}", (10, 50), font, 2, (0, 0, 255), 2)
    # Calculate FPS
    endTime = time.time()
    fps = 1 / (endTime - startTime) if endTime != startTime else 60
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), font, 1.5, (0, 0, 255), 2)

    # Show the output frame
    cv2.imshow("Person Face Tracking", frame)

    # Press 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

