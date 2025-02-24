
import cv2
import numpy as np
import time
import math

# Define video source (either webcam or video file)
videoSource = "E:/Project/final/multi-person-detection-and-tracking-main/sample.mp4"  # Use 0 for webcam, or replace with a file path like "mainGateTest.mp4"

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

time.sleep(2)  # Give the camera some time to warm up

# Load YOLO model
net = cv2.dnn.readNet("E:/Project/final/multi-person-detection-and-tracking-main/yolov3.weights", "E:/Project/final/multi-person-detection-and-tracking-main/yolov3.cfg")

# Load COCO names (class labels)
with open("E:/Project/final/multi-person-detection-and-tracking-main/coco.names", "r") as f:
    classes = f.read().strip().split('\n')

# Initialize global variables
people = []
personId = 0
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
enteredPeople = 0
exitedPeople = 0
doorThresh = 50
doorCoord = (670, 400)
appendThresh = 80
font = cv2.FONT_HERSHEY_PLAIN

class Person:
    def __init__(self, personId, location):
        self.id = personId
        self.curLocation = location
        self.trajectory = []
        self.state = []
        self.flag = 0
        if personId < len(colors):
            self.color = colors[personId]
        else:
            self.color = colors[personId % len(colors)]

    def addPointToTrajectory(self, location):
        self.trajectory.append(location)

def plotTrajectories(frame):
    global people, doorCoord
    for person in people:
        prev_point = None  # Store the previous point
        traj = person.trajectory[-5:] if len(person.trajectory) >= 5 else person.trajectory
        for i in traj:
            x, y = i[0], i[1]
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
    global people, personId, enteredPeople, exitedPeople, doorCoord

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

cv2.namedWindow("Person Tracking", cv2.WINDOW_NORMAL)

while True:
    startTime = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))
    frame = cv2.circle(frame, doorCoord, 5, (255, 0, 0))

    # Detect objects in the frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] == 'person':
                center_x, center_y, w, h = (detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype(int)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
    curCoords = []
    for i in indices:
        box = boxes[i]
        curCoords.append([box[0]+int(box[2]/2), box[1]+int(box[3]/2)])
        x, y, w, h = box[0], box[1], box[2], box[3]
    # Your code continues...


        # Draw a rectangle around each person
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Person', (x, y - 10), font, 1, (0, 255, 0), 1)

    frame = trackerHandle(curCoords, frame)

    peopleCount = len(people)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    endTime = time.time()
    fps = 1 / (endTime - startTime) if endTime != startTime else 60

    cv2.putText(frame, f"People Count: {peopleCount}", (550, 30), font, 1.5, (0, 0, 255), 2)
    # cv2.putText(frame, f"Entered Count: {enteredPeople}", (550, 60), font, 1.5, (0, 0, 255), 2)
    # cv2.putText(frame, f"Exited Count: {exitedPeople}", (550, 90), font, 1.5, (0, 0, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), font, 1.5, (0, 0, 255), 2)
    cv2.imshow("Person Tracking", frame)

cap.release()
cv2.destroyAllWindows()
