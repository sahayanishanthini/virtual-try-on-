import cv2
import time
import numpy as np
import argparse
from math import hypot

cat = 'tshirt'
# cat = 'jeans'
# cat = 'shirt'
# cat = 'knee'
# cat = 'bottom'

cap = cv2.VideoCapture(0)

def nothing(x):
    pass

protoFile = "mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "mpi/pose_iter_160000.caffemodel"
nPoints = 15
POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

while True:
    
    dress_image = cv2.imread("tshirt.png")
    
    frame = cv2.imread("man.jpg") #for image of a person
    
    frame = cv2.resize(frame,(408, 612))
    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    threshold = 0.1

    rows, cols, _ = frame.shape

    dress_mask = np.zeros((rows, cols), np.uint8)
    dress_mask.fill(0)

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    
    t = time.time()
    # input image dimensions for the network
    inWidth = 368
    inHeight = 368
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                            (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()
    # print("time taken by network : {:.3f}".format(time.time() - t))

    H = output.shape[2]
    W = output.shape[3]

    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
        # print(i)
        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold : 
            # cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            # cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        
        else :
    
    if cat == 'tshirt':
    
        shoulder_width = int(hypot((points[2][0]-20) - (points[5][0]+20), points[2][1] - points[5][1]))      
        
        dress_height = int(points[8][0])
        
        full_dress = cv2.resize(dress_image, (shoulder_width, dress_height))
        full_dress_gray = cv2.cvtColor(full_dress, cv2.COLOR_BGR2GRAY)
        _, dress_mask = cv2.threshold(full_dress_gray, 25, 255, cv2.THRESH_BINARY_INV)

        dress_area = frameCopy[points[2][1]-20: (points[2][1]-20) + dress_height,
                    points[2][0]-10: points[2][0]-10 + shoulder_width]

        dress_area_no_dress = cv2.bitwise_and(dress_area, dress_area, mask=dress_mask)

        final_dress = cv2.add(dress_area_no_dress, full_dress)

        frameCopy[points[2][1]-20: (points[2][1]-20) + dress_height,
                    points[2][0]-10: points[2][0]-10 + shoulder_width] = final_dress

    

    cv2.imshow('image', frameCopy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
