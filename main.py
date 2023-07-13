# import the necessary packages
from imutils.video import WebcamVideoStream, FPS
from vid_prereqs import MediaPipe, Facetracker
import imutils
import cv2
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
import time
import os

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tensorflow.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tensorflow.config.experimental.set_memory_growth(gpu, True)

# Disable GPU garbage collection
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'

model = load_model('Models/action_v4.h5')
facetracker = load_model('Models/facetracker_v3.h5')
actions = np.array(['idle_0','idle_1','idle_2','mouse_on','mouse_off', 'sleep','using_mouse_L','using_mouse_R','using_mouse_LR'])
expressions = ['Happy', 'Neutral', 'Sad']

# created a *threaded* video stream, allow the camera sensor to warm up
# and start the FPS counter


start_time = time.time()
# FPS update time in seconds
display_time = 2
fc = 0
FPS = 0

# Class
ft = Facetracker()
mp_obj = MediaPipe()

# 1. New detection variables
sequence = []
predictions = []
threshold = 0.5
frame_count = 0
last_predicted_motion = ""
use_mouse_movements = False
prediction_action = False

# Scroll
scroll_y = []


print("Turning on Action Recognition...")
cap = cv2.VideoCapture(0)

# Action Recognition

with mp_obj.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        # grab the frame from the threaded video stream and resize it
        ret, frame = cap.read()

        # 540x720
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.resize(frame, (720, 540))

        # Perform detection
        image, results = mp_obj.mediapipe_detection(frame)

        # Draw landmarks on the image 
        mp_obj.draw_styled_landmarks(image, results)

        # 2. Prediction logic
        keypoints = mp_obj.extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        frame_count += 1
        
        # 3. Check if sequence is complete and perform prediction
        if frame_count == 30:
            frame_count = 0
            
            # Predict and update predictions
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            action = np.argmax(res)
            predictions.append(actions[action])
            predictions = predictions[-2:]
            prediction_action = True
            print(predictions)
            
            if np.max(res) > threshold:
                last_predicted_motion = actions[np.argmax(res)]
            else:
                continue
        
        
        if prediction_action == True:
            if str(np.unique(predictions)) == "['mouse_on']":
                use_mouse_movements = True

            elif str(np.unique(predictions)) == "['mouse_off']":
                use_mouse_movements = False

        if use_mouse_movements:
            scroll_y = mp_obj.mouse_movements(image, results, scroll_y)
            scroll_y, reset_flag = mp_obj.scroll(scroll_y)
            if reset_flag:
                scroll_y = []

        if not use_mouse_movements:
            if str(np.unique(predictions)) == "['sleep']":
                break


        image = cv2.flip(image, 1)

        cv2.putText(image, f"Last Motion: {last_predicted_motion}", (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
        cv2.putText(image, f"Mouse: {use_mouse_movements}", (10, 125),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Check FPS
        fc+=1
        TIME = time.time() - start_time

        if (TIME) >= display_time :
            FPS = fc / (TIME)
            fc = 0
            start_time = time.time()

        fps_disp = "FPS: "+str(FPS)[:5]
	
        # Add FPS count on frame
        cv2.putText(image, fps_disp, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # # display the frame
        cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF

        # check if the 'q' key is pressed to quit
        if key == ord('q'):
            break

    # do a bit of cleanup
    cap.release()
    cv2.destroyAllWindows()
