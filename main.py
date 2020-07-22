from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import imutils
import time
import dlib
import cv2
import playsound
from closed_eye_detector import *

face_detector = dlib.get_frontal_face_detector()
face_landmark_predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# indexes of the facial landmarks for the left and right eye, resp
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

EYE_RATIO_THRESH=0.28
EYE_NBR_FRAMES = 25
# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
ALARM_ON = False

print("[INFO] starting video stream thread...")
video_stream = VideoStream(src=0).start()
time.sleep(1.0)

while True:
    frame = video_stream.read()
    frame = imutils.resize(frame,width=600,height=480)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # detect faces
    face_rects = face_detector(gray,0)
    # Loop over all faces
    for face in face_rects:
        # determine the facial keypoints for the face region
        face_landmarks = face_landmark_predictor(gray,face)
        # then convert them (x, y)-coordinates to a NumPy array
        face_landmarks = face_utils.shape_to_np(face_landmarks)

        # extract left and right eyes
        leftEye = face_landmarks[lStart:lEnd]
        rightEye = face_landmarks[rStart:rEnd]

        # Compute eye ratio
        left_ratio = eye_aspect_ratio(leftEye)
        right_ratio = eye_aspect_ratio(rightEye)

        # Take average of eye aspect
        total_ratio=(left_ratio+right_ratio)/2.0
        cv2.putText(frame, "score: {:.2f}".format(total_ratio), (400, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if total_ratio < EYE_RATIO_THRESH:
            COUNTER += 1
            cv2.putText(frame, "Eye: Closing", (30, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255,0), 2)
            # if the eyes were closed for a sufficient number of
            # then sound the alarm
            if COUNTER >= EYE_NBR_FRAMES:
                # if the alarm is not on, turn it on
                if not ALARM_ON:
                    ALARM_ON = True
                    # start a thread to have the alarm
                    # sound played in the background
                    t = Thread(target=playsound.playsound,
                               args=("Industrial_Alarm.wav",))
                    t.deamon = True
                    t.start()
                # draw an alarm on the frame
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            COUNTER = 0
            ALARM_ON = False
            cv2.putText(frame, "Eye: Opening", (30, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Frame",frame)
    key =cv2.waitKey(1) & 0xFF

    if(key==ord("q")):
        break
cv2.destroyAllWindows()
video_stream.stop()