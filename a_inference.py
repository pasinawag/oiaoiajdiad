import torch
import cv2
import mediapipe as mp
import numpy as np
from model import SignLanguageModel

# Load the model
model = SignLanguageModel()
model.load_state_dict(torch.load('./model.pth'))
model.eval()

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.6)

# Define the label dictionary for 5 classes
labels_dict = {0: 'A', 1: 'E', 2: 'I', 3: 'O', 4: 'U'}

confidence_threshold = 0.95  # 95% confidence threshold
unknown_threshold = 0.7  # Threshold for considering the input as unrecognized

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Process hand landmarks and normalize
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Define bounding box coordinates for the hand
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Model prediction
        input_tensor = torch.tensor([data_aux], dtype=torch.float32)
        with torch.no_grad():
            prediction = model(input_tensor)
            # Get the predicted class and confidence score
            confidence, predicted_class = torch.max(torch.nn.functional.softmax(prediction, dim=1), dim=1)

        # Check if the confidence exceeds the threshold for recognition
        if confidence.item() >= confidence_threshold:
            predicted_character = labels_dict[predicted_class.item()]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, f"{predicted_character}: {confidence.item() * 100:.2f}%",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
        elif confidence.item() >= unknown_threshold:
            # If the confidence is between unknown and threshold, it's uncertain
            cv2.putText(frame, "Unrecognized", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            # If confidence is too low, label it as unrecognized
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(frame, "Unrecognized", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)

    # Show the video frame
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
