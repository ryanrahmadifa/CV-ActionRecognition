import cv2
import datetime
from threading import Thread
import mouse
import mediapipe as mp
import numpy as np

class FPS:
	def __init__(self):
		# store the start time, end time, and total number of frames
		# that were examined between the start and end intervals
		self._start = None
		self._end = None
		self._numFrames = 0
	def start(self):
		# start the timer
		self._start = datetime.datetime.now()
		return self
	def stop(self):
		# stop the timer
		self._end = datetime.datetime.now()
	def update(self):
		# increment the total number of frames examined during the
		# start and end intervals
		self._numFrames += 1
	def elapsed(self):
		# return the total number of seconds between the start and
		# end interval
		return (self._end - self._start).total_seconds()
	def fps(self):
		# compute the (approximate) frames per second
		return self._numFrames / self.elapsed()
	

class WebcamVideoStream:
	def __init__(self, src=3):
		# initialize the video camera stream and read the first frame
		# from the stream
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()
		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False
		
	def start(self):
		# start the thread to read frames from the video stream
		Thread(target=self.update, args=()).start()
		return self
	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return
			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()
	def read(self):
		# return the frame most recently read
		return self.frame
	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True

class Facetracker:
    def __init__(self):
          pass
    
    def predict(self, frame, yhat, class_names):
        if np.any(yhat[0]):
            sample_coords = yhat[1][0]
            predicted_class = np.argmax(yhat[0][0])
            confidence = yhat[0][0][predicted_class]

            # Controls the main rectangle
            cv2.rectangle(frame,
                        tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)),
                        tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)),
                        (255, 0, 0), 1)

            # Controls the label rectangle
            cv2.rectangle(frame,
                        tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int), [0, -20])),
                        tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int), [150, 0])),
                        (255, 0, 0), -1)

            # Controls the text rendered
            class_name = class_names[predicted_class]
            confidence_rate = f'{confidence:.2f}'
            text = f'{class_name} ({confidence_rate})'
            cv2.putText(frame, text,
                        (int(sample_coords[0] * 450), int(sample_coords[1] * 450) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

class MediaPipe:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.model = self.mp_holistic.Holistic()

    def mediapipe_detection(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False  # Image is no longer writeable
        results = self.model.process(image)  # Make prediction
        image.flags.writeable = True  # Image is now writeable
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
        return image, results
    
    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, lh, rh])
    
    def draw_styled_landmarks(self, image, results):
        # # Draw face connections
        # self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION,
        #                                self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        #                                self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
        #                                )

        # Draw left hand connections
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                       self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                       )
        # Draw right hand connections
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                       self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                       )

    def mouse_movements(self, image, results, scroll_y):
        # Extract index finger tip coordinates
        if results.right_hand_landmarks:
            index_finger_tip = results.right_hand_landmarks.landmark[self.mp_holistic.HandLandmark.INDEX_FINGER_TIP]
            thumb_finger_tip = results.right_hand_landmarks.landmark[self.mp_holistic.HandLandmark.THUMB_TIP]
            pinky_finger_mcp = results.right_hand_landmarks.landmark[self.mp_holistic.HandLandmark.PINKY_MCP]
            ring_finger_dip = results.right_hand_landmarks.landmark[self.mp_holistic.HandLandmark.RING_FINGER_DIP]
	    
            right_finger_positions = {}

            for finger_name, finger_landmark in [
                ('index', index_finger_tip),
                ('thumb', thumb_finger_tip),
                ('pinky', pinky_finger_mcp),
                ('ring', ring_finger_dip)
            ]:
                finger_x = int(finger_landmark.x * image.shape[1])
                finger_y = int(finger_landmark.y * image.shape[0])
                finger_pos = (finger_x, finger_y)
                
                right_finger_positions[finger_name] = finger_pos

            # Draw circle at the index finger tip
            radius = 8
            color = (0, 255, 0)  # Green color (BGR format)
            thickness = 3  # Outline thickness
            cv2.circle(image, (right_finger_positions['index'][0], right_finger_positions['index'][1]), radius, color, thickness)


            # Cursor movement
            ## Calculate the trackpad position and size
            trackpad_width = 320 # 1920/6
            trackpad_height = 180 # 1080/6
            trackpad_x = 40
            trackpad_y = 190
            total_trackpad_x = trackpad_x + trackpad_width
            total_trackpad_y = trackpad_y + trackpad_height

            ## Draw the trackpad rectangle
            cv2.rectangle(image, (trackpad_x, trackpad_y), (total_trackpad_x, total_trackpad_y), (0, 0, 255), 2)


            ## Calculate the scaled finger position for the trackpad, make sure to mirror it on the y axis
            trackpad_finger_x = int((right_finger_positions['index'][0])-360)*-5
            trackpad_finger_y = int((right_finger_positions['index'][1])-190)*5


            ## Check whether the finger position is inside the trackpad
            if trackpad_x <= np.array((right_finger_positions['index'][0])) <= total_trackpad_x and trackpad_y <= np.array((right_finger_positions['index'][1])) <= total_trackpad_y:

                ## Move the mouse to the trackpad position
                mouse.move(trackpad_finger_x, trackpad_finger_y, absolute=True, duration=0.03)
            else:
                pass

            ## Euclidean distance
            left_click_dist = np.linalg.norm(np.array(right_finger_positions['thumb']) - np.array(right_finger_positions['pinky']))
            right_click_dist = np.linalg.norm(np.array(right_finger_positions['thumb']) - np.array(right_finger_positions['ring']))

            if left_click_dist < 20:
                mouse.click(button='left')

            elif right_click_dist < 20:
                mouse.click(button='right')

            # ADD SCROLLING WITH LEFT HAND
        if results.left_hand_landmarks:
            index_finger_tip = results.left_hand_landmarks.landmark[self.mp_holistic.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = results.left_hand_landmarks.landmark[self.mp_holistic.HandLandmark.MIDDLE_FINGER_TIP]
	    
            left_finger_positions = {}

            for finger_name, finger_landmark in [
                ('index', index_finger_tip),
                ('middle', middle_finger_tip)
            ]:
                finger_x = int(finger_landmark.x * image.shape[1])
                finger_y = int(finger_landmark.y * image.shape[0])
                finger_pos = (finger_x, finger_y)
                
                left_finger_positions[finger_name] = finger_pos

            scroll_dist = np.linalg.norm(np.array(left_finger_positions['index']) - np.array(left_finger_positions['middle']))
            
            
	    
            if scroll_dist < 20:
                # Scroll up or down based on the vertical movement of the fingers
                y = np.array(left_finger_positions['index'][1])  # Store the original y-coordinate
                scroll_y.append(y)
            else:
                scroll_y = []
        return scroll_y
                
    def scroll(self, scroll_y):
        if len(scroll_y) == 3:

            # Calculate the vertical distance traveled by the fingers
            vertical_distance = scroll_y[2] - scroll_y[0]
            
            # Scroll amount based on the vertical distance traveled
            scroll_amount = np.floor(vertical_distance * 0.25 )  # Adjust the scroll speed as needed
            
            # Perform the scroll action
            mouse.wheel(scroll_amount)
	    
            scroll_y = []
        # Return the updated scroll_y list and a flag indicating reset
            return [], True
    
        else:
	    
            # Return the original scroll_y list and a flag indicating no reset
            return scroll_y, False