# # import cv2
# # import numpy as np
# # import time
# # import math

# # # Define video source (either webcam or video file)
# # videoSource = "C:/Users/yuvaraju/OneDrive/Desktop/multi-person-detection-and-tracking-main/sample.mp4"
# # cap = cv2.VideoCapture(videoSource)

# # if not cap.isOpened():
# #     print("Error: Could not open video source.")
# #     exit()

# # time.sleep(2)  # Give the camera some time to warm up

# # # Load YOLOv8 model
# # from ultralytics import YOLO
# # model = YOLO('yolov8n.pt')

# # # Initialize global variables
# # people = []
# # personId = 0
# # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
# # enteredPeople = 0
# # exitedPeople = 0
# # doorLineY = 300  # Define the y-coordinate of the virtual door
# # appendThresh = 80
# # font = cv2.FONT_HERSHEY_PLAIN

# # # Cooldown timer to avoid multiple entries/exits
# # cooldown_time = 10  # 10 frames
# # cooldown_tracker = {}

# # class Person:
# #     def __init__(self, personId, location):
# #         self.id = personId
# #         self.curLocation = location
# #         self.trajectory = []
# #         self.crossedDoor = False
# #         self.cooldown = cooldown_time  # Cooldown period before re-counting
# #         self.color = colors[personId % len(colors)]

# #     def addPointToTrajectory(self, location):
# #         self.trajectory.append(location)

# # def plotTrajectories(frame):
# #     global people
# #     for person in people:
# #         prev_point = None
# #         traj = person.trajectory[-5:] if len(person.trajectory) >= 5 else person.trajectory
# #         for i in traj:
# #             x, y = int(i[0]), int(i[1])  # Convert to integers
# #             color = person.color
# #             frame = cv2.circle(frame, (x, y), 3, color, cv2.FILLED)
# #             if prev_point is not None:
# #                 frame = cv2.line(frame, prev_point, (x, y), color, 1)
# #             prev_point = (x, y)
# #     return frame

# # def calcDist(a, b):
# #     x1, y1 = a
# #     x2, y2 = b
# #     return math.sqrt(((x2 - x1)**2) + ((y2 - y1)**2))

# # def checkEntryExit(person, frame):
# #     global enteredPeople, exitedPeople, doorLineY, cooldown_tracker

# #     # Check cooldown
# #     if person.id in cooldown_tracker and cooldown_tracker[person.id] > 0:
# #         cooldown_tracker[person.id] -= 1
# #         return

# #     # Check if the person has a valid trajectory with at least two points
# #     if len(person.trajectory) > 1:
# #         prev_y = person.trajectory[-2][1]
# #         curr_y = person.trajectory[-1][1]

# #         # Check if the person crosses the door line
# #         if prev_y < doorLineY <= curr_y and not person.crossedDoor:
# #             enteredPeople += 1
# #             person.crossedDoor = True
# #             cooldown_tracker[person.id] = cooldown_time  # Start cooldown
# #         elif prev_y > doorLineY >= curr_y and not person.crossedDoor:
# #             exitedPeople += 1
# #             person.crossedDoor = True
# #             cooldown_tracker[person.id] = cooldown_time  # Start cooldown

# #         # Reset the state if the person moves away from the door line by a buffer distance
# #         if abs(curr_y - doorLineY) > 50:  # 50 is a buffer to reset crossing
# #             person.crossedDoor = False


# # def trackerHandle(curCoords, frame):
# #     global people, personId

# #     if people:
# #         if curCoords:
# #             for i, coord in enumerate(curCoords):
# #                 if i < len(people):
# #                     dist = calcDist(people[i].curLocation, coord)
# #                     if dist < appendThresh:
# #                         people[i].addPointToTrajectory(coord)
# #                     people[i].curLocation = coord
# #                     checkEntryExit(people[i], frame)
# #                 else:
# #                     createPerson(coord)
# #         else:
# #             people = []
# #     elif curCoords:
# #         for coord in curCoords:
# #             createPerson(coord)

# #     return plotTrajectories(frame)

# # def createPerson(currentCoord):
# #     global people, personId
# #     person = Person(personId, currentCoord)
# #     people.append(person)
# #     personId += 1

# # def displayDistances(frame):
# #     for i in range(len(people)):
# #         for j in range(i + 1, len(people)):
# #             dist = calcDist(people[i].curLocation, people[j].curLocation)
# #             if dist < 100:  # Customize the threshold distance
# #                 p1 = tuple(map(int, people[i].curLocation))
# #                 p2 = tuple(map(int, people[j].curLocation))
# #                 cv2.line(frame, p1, p2, (0, 0, 255), 2)
# #                 mid_point = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
# #                 cv2.putText(frame, f"{int(dist)}", mid_point, font, 1, (0, 255, 255), 2)
# #     return frame

# # cv2.namedWindow("Person Tracking", cv2.WINDOW_NORMAL)

# # while True:
# #     ret, frame = cap.read()
# #     if not ret:
# #         break

# #     frame = cv2.resize(frame, (800, 600))
# #     cv2.line(frame, (0, doorLineY), (800, doorLineY), (255, 0, 0), 2)  # Draw door line

# #     # Detect objects in the frame using YOLOv8
# #     results = model.predict(frame, verbose=False)
# #     curCoords = []

# #     for result in results:
# #         for box in result.boxes.xyxy:
# #             x1, y1, x2, y2 = map(int, box[:4])
# #             center_x = (x1 + x2) // 2
# #             center_y = (y1 + y2) // 2
# #             curCoords.append([center_x, center_y])
# #             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
# #             cv2.putText(frame, 'Person', (x1, y1 - 10), font, 1, (0, 255, 0), 1)

# #     frame = trackerHandle(curCoords, frame)
# #     frame = displayDistances(frame)

# #     peopleCount = len(people)

# #     cv2.putText(frame, f"People Count: {peopleCount}", (550, 30), font, 1.5, (0, 0, 255), 2)
# #     cv2.putText(frame, f"Entered Count: {enteredPeople}", (550, 60), font, 1.5, (0, 0, 255), 2)
# #     cv2.putText(frame, f"Exited Count: {exitedPeople}", (550, 90), font, 1.5, (0, 0, 255), 2)
# #     cv2.imshow("Person Tracking", frame)

# #     key = cv2.waitKey(1) & 0xFF
# #     if key == ord('q'):
# #         break

# # cap.release()
# # cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import time
# import math

# # Define video source (either webcam or video file)
# videoSource = "C:/Users/yuvaraju/OneDrive/Desktop/multi-person-detection-and-tracking-main/sample.mp4"
# cap = cv2.VideoCapture(videoSource)

# if not cap.isOpened():
#     print("Error: Could not open video source.")
#     exit()

# time.sleep(2)  # Give the camera some time to warm up

# # Load YOLOv8 model
# from ultralytics import YOLO
# model = YOLO('yolov8n.pt')

# # Initialize global variables
# people = []
# personId = 0
# colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
# enteredPeople = 0
# exitedPeople = 0
# doorLineY = 300  # Define the y-coordinate of the virtual door
# appendThresh = 80
# font = cv2.FONT_HERSHEY_PLAIN

# # Cooldown timer to avoid multiple entries/exits
# cooldown_time = 10  # 10 frames
# cooldown_tracker = {}

# class Person:
#     def __init__(self, personId, location):
#         self.id = personId
#         self.curLocation = location
#         self.trajectory = []
#         self.crossedDoor = False
#         self.cooldown = cooldown_time  # Cooldown period before re-counting
#         self.color = colors[personId % len(colors)]

#     def addPointToTrajectory(self, location):
#         self.trajectory.append(location)

# def plotTrajectories(frame):
#     global people
#     for person in people:
#         prev_point = None
#         traj = person.trajectory[-5:] if len(person.trajectory) >= 5 else person.trajectory
#         for i in traj:
#             x, y = int(i[0]), int(i[1])  # Convert to integers
#             color = person.color
#             frame = cv2.circle(frame, (x, y), 3, color, cv2.FILLED)
#             if prev_point is not None:
#                 frame = cv2.line(frame, prev_point, (x, y), color, 1)
#             prev_point = (x, y)
#     return frame

# def calcDist(a, b):
#     x1, y1 = a
#     x2, y2 = b
#     return math.sqrt(((x2 - x1)**2) + ((y2 - y1)**2))

# def checkEntryExit(person, frame):
#     global enteredPeople, exitedPeople, doorLineY, cooldown_tracker

#     # Check cooldown
#     if person.id in cooldown_tracker and cooldown_tracker[person.id] > 0:
#         cooldown_tracker[person.id] -= 1
#         return

#     # Check if the person has a valid trajectory with at least two points
#     if len(person.trajectory) > 1:
#         prev_y = person.trajectory[-2][1]
#         curr_y = person.trajectory[-1][1]

#         # Check if the person crosses the door line
#         if prev_y < doorLineY <= curr_y and not person.crossedDoor:
#             enteredPeople += 1
#             person.crossedDoor = True
#             cooldown_tracker[person.id] = cooldown_time  # Start cooldown
#         elif prev_y > doorLineY >= curr_y and not person.crossedDoor:
#             exitedPeople += 1
#             person.crossedDoor = True
#             cooldown_tracker[person.id] = cooldown_time  # Start cooldown

#         # Reset the state if the person moves away from the door line by a buffer distance
#         if abs(curr_y - doorLineY) > 50:  # 50 is a buffer to reset crossing
#             person.crossedDoor = False

# def trackerHandle(curCoords, frame):
#     global people, personId

#     if people:
#         if curCoords:
#             for i, coord in enumerate(curCoords):
#                 if i < len(people):
#                     dist = calcDist(people[i].curLocation, coord)
#                     if dist < appendThresh:
#                         people[i].addPointToTrajectory(coord)
#                     people[i].curLocation = coord
#                     checkEntryExit(people[i], frame)
#                 else:
#                     createPerson(coord)
#         else:
#             people = []
#     elif curCoords:
#         for coord in curCoords:
#             createPerson(coord)

#     return plotTrajectories(frame)

# def createPerson(currentCoord):
#     global people, personId
#     person = Person(personId, currentCoord)
#     people.append(person)
#     personId += 1

# def displayDistances(frame):
#     for i in range(len(people)):
#         for j in range(i + 1, len(people)):
#             dist = calcDist(people[i].curLocation, people[j].curLocation)
#             if dist < 100:  # Customize the threshold distance
#                 p1 = tuple(map(int, people[i].curLocation))
#                 p2 = tuple(map(int, people[j].curLocation))
#                 cv2.line(frame, p1, p2, (0, 0, 255), 2)
#                 mid_point = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
#                 cv2.putText(frame, f"{int(dist)}", mid_point, font, 1, (0, 255, 255), 2)
#     return frame

# cv2.namedWindow("Person Tracking", cv2.WINDOW_NORMAL)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.resize(frame, (800, 600))
#     cv2.line(frame, (0, doorLineY), (800, doorLineY), (255, 0, 0), 2)  # Draw door line

#     # Detect objects in the frame using YOLOv8
#     results = model.predict(frame, verbose=False)
#     curCoords = []
#     for result in results:
#         for box in result.boxes.xyxy:
#             class_name = result.names[int(result.boxes.cls[0])]  # Get class name
#             if class_name == 'person':  # Check if it's a person
#                 x1, y1, x2, y2 = map(int, box[:4])
#                 center_x = (x1 + x2) // 2
#                 center_y = (y1 + y2) // 2
#                 curCoords.append([center_x, center_y])
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, 'Person', (x1, y1 - 10), font, 1, (0, 255, 0), 1)

#     frame = trackerHandle(curCoords, frame)
#     frame = displayDistances(frame)

#     peopleCount = len(people)

#     cv2.putText(frame, f"People Count: {peopleCount}", (550, 30), font, 1.5, (0, 0, 255), 2)
#     cv2.putText(frame, f"Entered Count: {enteredPeople}", (550, 60), font, 1.5, (0, 0, 255), 2)
#     cv2.putText(frame, f"Exited Count: {exitedPeople}", (550, 90), font, 1.5, (0, 0, 255), 2)
#     cv2.imshow("Person Tracking", frame)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
import cv2
import numpy as np
import time
import math
from ultralytics import YOLO  # Import YOLOv8 model from the ultralytics package

# Define video source (either webcam or video file)
videoSource = "C:/Users/yuvaraju/OneDrive/Desktop/multi-person-detection-and-tracking-main/sample.mp4"
cap = cv2.VideoCapture(videoSource)  # Corrected for video file input
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

time.sleep(2)  # Give the camera some time to warm up

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # You can replace 'yolov8n.pt' with a larger model like 'yolov8m.pt'

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

# Person class to track individual people
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

# Function to plot the trajectories of tracked people
def plotTrajectories(frame):
    global people, doorCoord
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

# Function to calculate Euclidean distance between two points
def calcDist(a, b):
    x1, y1 = a
    x2, y2 = b
    return math.sqrt(((x2-x1)*(x2-x1)) + ((y2-y1)*(y2-y1)))

# Function to handle people tracking logic
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

# Function to create a new person and assign a unique ID
def createPerson(currentCoord):
    global people, personId
    person = Person(personId, currentCoord)
    people.append(person)
    personId += 1

# Main loop for processing video frames
cv2.namedWindow("Person Tracking", cv2.WINDOW_NORMAL)

while True:
    startTime = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))
    frame = cv2.circle(frame, doorCoord, 5, (255, 0, 0))

    # Detect objects in the frame using YOLOv8
    results = model.predict(frame, save=False, conf=0.5)  # Get predictions from YOLOv8

    curCoords = []
    for result in results:
        for bbox in result.boxes.xywh:
            x_center, y_center, width, height = bbox.tolist()
            x = int(x_center - width / 2)
            y = int(y_center - height / 2)
            w = int(width)
            h = int(height)
            
            curCoords.append([x_center, y_center])

            # Draw a rectangle around each detected person
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Person', (x, y - 10), font, 1, (0, 255, 0), 1)

    # Handle the tracking of people
    frame = trackerHandle(curCoords, frame)

    # Count the number of people detected
    peopleCount = len(people)

    # Display stats
    cv2.putText(frame, f"People Count: {peopleCount}", (550, 30), font, 1.5, (0, 0, 255), 2)
    cv2.putText(frame, f"Entered Count: {enteredPeople}", (550, 60), font, 1.5, (0, 0, 255), 2)
    cv2.putText(frame, f"Exited Count: {exitedPeople}", (550, 90), font, 1.5, (0, 0, 255), 2)

    # Calculate FPS
    endTime = time.time()
    fps = 1 / (endTime - startTime) if endTime != startTime else 60
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), font, 1.5, (0, 0, 255), 2)

    # Show the output frame
    cv2.imshow("Person Tracking", frame)

    # Press 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
