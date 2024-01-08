"""
import cv2
import mediapipe as mp

camera_video = cv2.VideoCapture(0)
camera_video.set(3,1280)
camera_video.set(4,960)

cv2.namedWindow('Dem ngon tay', cv2.WINDOW_NORMAL)

mp_hands = mp.solutions.hands

hands= mp_hands.Hands(static_image_mode=True, model_complexity=1, max_num_hands=2, min_detection_confidence=0.75, min_tracking_confidence=0.5)
hands_video = mp.hands.Hands(static_image_mode=True, model_complexity=1, max_num_hands=2,min_detection_confidence=0.75, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

while camera_video.isOpened():

    ok, frame = camera_video.read()

    if not ok:
        continue

    frame = cv2.flip(frame, 1)

    cv2.imshow('Dem ngon tay', frame)

    k = cv2.waitKey(1) & 0xFF

    if (k ==27):
        break

camera_video.release()
cv2.destroyAllWindows()