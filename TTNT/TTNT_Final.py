import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, model_complexity=1, max_num_hands=2, min_detection_confidence=0.75,
                       min_tracking_confidence=0.5)
hands_videos = mp_hands.Hands(static_image_mode=False, model_complexity=1, max_num_hands=2,
                              min_detection_confidence=0.75, min_tracking_confidence=0.5)

camera_video = cv2.VideoCapture(0)
camera_video.set(3, 1280)
camera_video.set(4, 960)

cv2.namedWindow('COUNTING FINGERS', cv2.WINDOW_NORMAL)

def detectHandsLandmarks(image, hands, draw=True, display=True):
    output_image = image.copy()
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks and draw:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image=output_image, landmark_list=hand_landmarks,
                                      connections=mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255),
                                                                                   thickness=2, circle_radius=2),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0),
                                                                                     thickness=2, circle_radius=2))

    if display:
        plt.figure(figsize=[15, 15])
        plt.subplot(121)
        plt.imshow(image[:, :, ::-1])
        plt.title("Original Image")
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output")
        plt.axis('off')
    else:
        return output_image, results


def countFingersAndDisplay(image, results):
    height, width, _ = image.shape
    output_image = image.copy()
    count = {'Right': 0, 'Left': 0}
    fingers_tips_ids = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                        mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]

    for hand_index, hand_info in enumerate(results.multi_handedness):
        hand_label = hand_info.classification[0].label
        hand_landmarks = results.multi_hand_landmarks[hand_index]
        fingers_count = 0

        for tip_index in fingers_tips_ids:
            finger_name = tip_index.name.split("_")[0]
            if (hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index - 2].y):
                fingers_count += 1

        thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
        thumb_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP - 2].x

        if (hand_label == 'Right' and (thumb_tip_x < thumb_mcp_x)) or (
                hand_label == 'Left' and (thumb_tip_x > thumb_mcp_x)):
            fingers_count += 1

        count[hand_label] += fingers_count

    cv2.putText(output_image, f"Left: {count['Left']} | Right: {count['Right']}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (124, 252, 0), 3)

    return output_image

while camera_video.isOpened():

    ok, frame = camera_video.read()
    if not ok:
        continue
    frame = cv2.flip(frame, 1)
    frame, results = detectHandsLandmarks(frame, hands_videos, display=False)
    if results.multi_hand_landmarks:
        frame = countFingersAndDisplay(frame, results)
    cv2.imshow('COUNTING FINGERS', frame)
    k = cv2.waitKey(1) & 0xFF
    if (k == 27):
        break

camera_video.release()
cv2.destroyAllWindows()
