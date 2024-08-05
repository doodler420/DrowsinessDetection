from imutils.video import VideoStream
from imutils import face_utils
import threading
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import simpleaudio as sa
from collections import deque


def sound_alarm(path):
    global alarm_status
    global alarm_status2
    global status_update
    while alarm_status:
        print('call')
        status_update= True
        wave_obj = sa.WaveObject.from_wave_file(path)
        play_obj = wave_obj.play()
        play_obj.wait_done()
        status_update= False
    if alarm_status2:
        print('call')
        status_update = True
        wave_obj = sa.WaveObject.from_wave_file(path)
        play_obj = wave_obj.play()
        play_obj.wait_done()
        status_update = False

def eye_aspect_ratio(eye):
    # Calculate Euclidean distances between sets of vertical eye landmarks
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    # Calculate Euclidean distance between the horizontal eye landmarks
    C = np.linalg.norm(eye[0] - eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(shape):
    # Calculate Euclidean vertical distances between mouth landmarks
    A = np.linalg.norm(shape[52] - shape[58])
    B = np.linalg.norm(shape[53] - shape[57])
    C = np.linalg.norm(shape[51] - shape[59])
    # Calculate horizontal distance of the mouth
    D = np.linalg.norm(shape[54] - shape[48])
    # Compute the mouth aspect ratio
    mar = (A + B + C) / (3.0* D)
    return mar

def final_ear(shape):
    # Get the indices for the left and right eye landmarks
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    # Compute the final eye aspect ratio
    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def final_mar(shape):
    mar = mouth_aspect_ratio(shape)
    return mar

#Argument parser for the camera and alarm sound file
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
ap.add_argument("-a", "--alarm", type=str, default="ale.wav", help="path alarm .WAV file")
args = vars(ap.parse_args())

EAR_THRESH = 0.258
consec_frames_eye = int(cv2.VideoCapture(0).get(cv2.CAP_PROP_FPS)) * 3
consec_frames_mouth = int(cv2.VideoCapture(0).get(cv2.CAP_PROP_FPS)) * 6
YAWN_THRESH = 0.485
alarm_status = False
alarm_status2 = False
status_update = False
COUNTER = 0
COUNTER2 = 0
LEARNING_RATE = 1.5  # Adaptive learning rate for threshold adjustment
print("-> Loading the predictor and detector...")
#detector = dlib.get_frontal_face_detector()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    #Faster but less accurate
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
#vs= VideoStream(usePiCamera=True).start()       //For Raspberry Pi
# Get the frame rate of the camera
frame_rate = int(vs.stream.get(cv2.CAP_PROP_FPS))
# Initialize deque to store recent eye and mouth aspect ratios based on the frame rate of the camera so that the values update every 1 second
ear_history = deque(maxlen=frame_rate * 10)  # Store 10 seconds of data
mar_history = deque(maxlen=frame_rate * 10)  # Store 10 seconds of data
time.sleep(1.0)

while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #rects = detector(gray, 0)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=3, 
                                      minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
    # for rect in rects:
    #     (x, y, w, h) = face_utils.rect_to_bb(rect)  # Convert dlib rectangle to (x, y, w, h) format
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye[1]
        rightEye = eye[2]
        mar = final_mar(shape)
         # Update the deque with recent eye and mouth aspect ratios
        ear_history.append(ear)
        mar_history.append(mar)
        #  # Calculate dynamic eye aspect ratio threshold
        # ear_thresh = adaptive_threshold(np.array(ear_history), bandwidth=BANDWIDTH)
        # # Calculate dynamic mouth aspect ratio threshold
        # mar_thresh = adaptive_threshold(np.array(mar_history), bandwidth=BANDWIDTH)

        # # Adjust bandwidth based on the difference between predefined and current thresholds
        # BANDWIDTH = adjust_bandwidth(EAR_THRESH, ear_thresh, BANDWIDTH)
        # ear_thresh = adaptive_threshold(np.array(ear_history), bandwidth=BANDWIDTH)
        # #plotting and testing
        # plot_density_function(np.array(ear_history), bandwidth=0.5)
        
        # BANDWIDTH = adjust_bandwidth(YAWN_THRESH, mar_thresh, BANDWIDTH)
        # mar_thresh = adaptive_threshold(np.array(mar_history), bandwidth=BANDWIDTH)
        # # Calculate dynamic eye aspect ratio threshold
        # ear_thresh = np.mean(ear_history) - ALPHA * np.std(ear_history)
        # # Calculate dynamic mouth aspect ratio threshold
        # mar_thresh = np.mean(mar_history) - ALPHA * np.std(mar_history)
        # Calculate dynamic eye aspect ratio threshold
        ear_thresh = EAR_THRESH - LEARNING_RATE * (np.mean(ear_history) - EAR_THRESH)
        
        # Calculate dynamic mouth aspect ratio threshold
        mar_thresh = YAWN_THRESH + LEARNING_RATE * (YAWN_THRESH - np.mean(mar_history))

        if ear < ear_thresh:
            COUNTER += 1
            if COUNTER >= consec_frames_eye:
                cv2.putText(frame, "Open Your Eyes!", (10, 30),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
                if not alarm_status and not status_update:
                    alarm_status = True
                    if args["alarm"] != "":
                        sound_thread = threading.Thread(target=sound_alarm, args=(args["alarm"],))
                        sound_thread.daemon = True
                        sound_thread.start()
                
        else:
            COUNTER = 0
            alarm_status = False

        if mar > mar_thresh:
            COUNTER2 += 1
            if COUNTER2 >= consec_frames_mouth:
                cv2.putText(frame, "You are yawning", (10, 60),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
                if not alarm_status2 and not status_update:
                    alarm_status2 = True
                    if args["alarm"] != "":
                        sound_thread = threading.Thread(target=sound_alarm, args=(args["alarm"],))
                        sound_thread.daemon = True
                        sound_thread.start()

        else:
            COUNTER2 = 0
            alarm_status2 = False

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR_T: {:.2f}".format(ear_thresh), (300, 90),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "MAR_T: {:.2f}".format(mar_thresh), (300, 120),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
        
        
        # Draw contours around the eyes
        leftEye = final_ear(shape)[1]
        rightEye = final_ear(shape)[2]
        cv2.drawContours(frame, [leftEye], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEye], -1, (0, 255, 0), 1)

        # Draw contour around the mouth
        mouth = shape[48:68]
        cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)


    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("z"):
        break

cv2.destroyAllWindows()
vs.stop()