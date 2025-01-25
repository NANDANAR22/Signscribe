import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe for Hand Detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# Function to detect a basic gesture (A, B, or C)
def detect_gesture(landmarks):
    # Gesture "A" (e.g., thumb closed, other fingers extended)
    if landmarks[4][1] < landmarks[3][1] and landmarks[8][1] < landmarks[7][1] and landmarks[12][1] < landmarks[11][1]:
        return "A"

    # Gesture "B" (e.g., all fingers extended, but thumb bent inward)
    elif landmarks[4][1] > landmarks[3][1] and landmarks[8][1] < landmarks[7][1] and landmarks[12][1] < landmarks[11][1]:
        return "B"

    # Gesture "C" (e.g., thumb and fingers forming a "C")
    elif landmarks[4][1] < landmarks[3][1] and landmarks[8][1] > landmarks[7][1]:
        return "C"

    return "Unknown"

# Start webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract hand landmarks and convert them to a list
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
            
            # Use a rule-based method to classify the gesture
            gesture = detect_gesture(landmarks)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Display the recognized gesture
            cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the result in the OpenCV window
    cv2.imshow("Sign Language Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
