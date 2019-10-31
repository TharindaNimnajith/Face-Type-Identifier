import cv2
import dlib
from imutils import face_utils #importing image utilities library

face_detector = dlib.get_frontal_face_detector() #classifier for detecting faces in a given image
landmarks_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') #classifier for detecting 68 landmarks in a face

camera = cv2.VideoCapture(0) #0 for default camera

#infinite loop
#while(1): -> is also correct
while(True): 
    ret, image = camera.read() #reading a single frame and save it to image
    #ret is the boolean return on whether the camera works or not
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #color conversion bgr -> gray scale
    
    rectangle = face_detector(gray)

    #exception handling with a try - except blocks
    try:
        #x1 = rectangle[0].left()
        #y1 = rectangle[0].top()
        #x2 = rectangle[0].right()
        #y2 = rectangle[0].bottom()

        #cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        points = landmarks_detector(gray, rectangle[0]) #passing the gray face to the landmarks detector
        points = face_utils.shape_to_np(points) #converting the 68 points into a numpy array
        #points is a numpy 2D array[68][2] with 64 rows(68 points) and 2 columns(x, y coordinates for the each point)

        k = 0
        for (x, y) in points:
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1) #-1 for filled circle, otherwise 1, 2, 3, ... for the width of the curve
            cv2.putText(image, str(k), (x - 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
            k += 1;
            
        cv2.imshow('LIVE', image)
        cv2.waitKey(1) #1ms pause, allowing time to camera to get another frame

    #except: 
    except Exception as e:
        #pass
        print(e)
