import cv2
from imutils import face_utils
import dlib
import pickle  #can write arrays directly into files
import numpy as np

face_detector = dlib.get_frontal_face_detector()
landmarks_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

train_data   = [[0 for x in range(5)] for y in range(60)] #train_data is a 2D array[60][5] -> 60 rows - 60 faces, 5 columns - 5 distances
train_target = [0 for x in range(60)] #train_target is a 1D array[60] -> names of the each face type: diamond, oval etc.
image_shapes = ['Diamond', 'Oblong', 'Oval', 'Round', 'Square', 'Triangle']

count = 0

def append_data(points, shape):
    face_dictionary = {'Diamond':0, 'Oblong':1, 'Oval':2, 'Round':3, 'Square':4, 'Triangle':5}
    
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
    
    global count
    global train_data
    global train_target 

    train_data[count][0] = round(D1, 4)
    train_data[count][1] = round(D2, 4)
    train_data[count][2] = round(D3, 4)
    train_data[count][3] = round(D4, 4)
    train_data[count][4] = round(D5, 4)
    
    train_target[count] = face_dictionary[shape]
    
    count += 1
    
##def append_data(points, shape):
##    face_dictionary = {'Diamond' : 0, 'Oblong' : 1, 'Oval' : 2, 'Round' : 3, 'Square' : 4, 'Triangle' : 5}
##    
##    points7 = [0 for x in range(7)]
##   
##    for i in range(0, 6):
##        points7[i] = points[i + 2][0]
##
##    d = [0 for x in range(6)]
##    for i in range(0,6):
##        d[i + 1] = points7[6] - points7[i]
##
##    D = [0 for x in range(5)]
##    for i in range(0, 5):
##        D[i + 1] = d[i + 2] / float(d[1]) * 100
##
##    global count, train_data, train_target
##
##    for i in range(0, 5):
##        train_data[count][i] = round(D[i + 1], 4)
##
##    train_target[count] = face_dictionary[shape]
##    
##    count += 1
    
for num in range(0, 10):
    for shape in image_shapes:
        image     = cv2.imread('Face Shapes/' + shape + '/' + str(num + 1) + '.jpeg')
        gray      = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rectangle = face_detector(gray)
        points    = landmarks_detector(gray, rectangle[0])
        points    = face_utils.shape_to_np(points)
        append_data(points, shape)

train_data   = np.array(train_data)
train_target = np.array(train_target)

print(train_data)
print()
print(train_target)

train_data_file   = open('train_data.pickle', 'wb')   #wb - write in bytes (file modes - write / read / append etc.)
train_target_file = open('train_target.pickle', 'wb')

pickle.dump(train_data, train_data_file)     #dump = write
pickle.dump(train_target, train_target_file)

train_data_file.close()
train_target_file.close()
