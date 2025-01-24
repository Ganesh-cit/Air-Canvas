import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Parameters for drawing
canvas = None
pen_color = (0, 255, 0)  # Color of the pen in BGR
brush_thickness = 5
eraser_thickness = 25

# Variable to store previous fingertip position
prev_x, prev_y = 0, 0

# Initialize webcam
cap = cv2.VideoCapture(0)
erase_mode=False
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame to avoid mirrored view
        frame = cv2.flip(frame, 1)

        # Create a canvas if it does not exist
        if canvas is None:
            canvas = np.zeros_like(frame)

        # Convert to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # If hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the index fingertip position (landmark 8)
                index_finger_tip = hand_landmarks.landmark[8]
                height, width, _ = frame.shape
                x, y = int(index_finger_tip.x * width), int(index_finger_tip.y * height)

                # Check if this is the first frame
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y
                if abs(x - prev_x) < 50 and abs(y - prev_y) < 50:  # Avoid large jumps
                    cv2.line(canvas, (prev_x, prev_y), (x, y), pen_color, brush_thickness)
                    prev_x, prev_y = x, y
                if erase_mode==True:
                    cv2.circle(canvas, (x, y), eraser_thickness, (0, 0, 0), -1)  # Draw on the canvas


                # Update the previous coordinates





        # Combine canvas and webcam feed
        frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

        # Display result
        cv2.imshow("Air Canvas", frame)

        # Key press handling
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key==ord("s"):

            cv2.imwrite('Captured images/captured_frame.jpg', frame)
            print("Frame captured and saved as 'captured_frame.jpg'.")

        elif key == ord("e"):
            if erase_mode==False:
                erase_mode=True
            else:
                erase_mode=False




        elif key == ord("c"):
            canvas = np.zeros_like(frame)  # Clear canvas

# Cleanup
cap.release()
cv2.destroyAllWindows()
