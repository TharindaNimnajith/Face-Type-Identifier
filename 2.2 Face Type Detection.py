import pickle
from sklearn import neighbors
import cv2
import dlib
from imutils import face_utils

data_file    = open('train_data.pickle', 'rb')   #rb = read in bytes
target_file  = open('train_target.pickle', 'rb')

train_data   = pickle.load(data_file)   #load = read
train_target = pickle.load(target_file)

##print(train_data)
##print(train_target)
##print(len(train_data))
##print(len(train_target))

classifier = neighbors.KNeighborsClassifier()
classifier.fit(train_data, train_target)

camera = cv2.VideoCapture(0)

def predict_face(image, points):
    face_dictionary = {0:'Diamond', 1:'Oblong', 2:'Oval', 3:'Round', 4:'Square', 5:'Triangle'}
    
    points7 = [0 for x in range(7)]

    points7[0] = points[2][0]
    points7[1] = points[3][0]
    points7[2] = points[4][0]
    points7[3] = points[5][0]
    points7[4] = points[6][0]
    points7[5] = points[7][0]
    points7[6] = points[8][0]

    d1 = points7[6] - points7[0]
    d2 = points7[6] - points7[1]
    d3 = points7[6] - points7[2]
    d4 = points7[6] - points7[3]
    d5 = points7[6] - points7[4]
    d6 = points7[6] - points7[5]

    D1 = d2 / float(d1) * 100
    D2 = d3 / float(d1) * 100
    D3 = d4 / float(d1) * 100
    D4 = d5 / float(d1) * 100
    D5 = d6 / float(d1) * 100

    result = classifier.predict([[D1, D2, D3, D4, D5]])
    result = result[0]
    #print(face_dictionary[result])
    cv2.putText(image, face_dictionary[result], (320, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
face_detector = dlib.get_frontal_face_detector()
landmarks_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')    

while(True): 
    ret, image = camera.read()    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
    rectangle = face_detector(gray)

    image[:50, :] = [0, 255, 0]
    #image[0:][0:50] = [0, 255, 0] #green color
    #from 0 to the end of the image in x axis
    #from 0 to 30 of the image in y axis
    
    cv2.putText(image, 'FACE TYPE : ', (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    try:
        points = landmarks_detector(gray, rectangle[0])
        points = face_utils.shape_to_np(points)

        k = 0
        for (x, y) in points:
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
            cv2.putText(image, str(k), (x - 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
            k += 1;

        predict_face(image, points)
        cv2.imshow('LIVE', image)
        cv2.waitKey(1)

    except Exception as e:
        print(e)
