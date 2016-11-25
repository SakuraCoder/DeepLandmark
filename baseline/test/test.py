import os, sys
import time
from functools import partial
import cv2
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from common import getCNNs
from common import getDataFromTxt, logger
from common import shuffle_in_unison_scary, createDir, processImage


TXT = 'dataset/train/testImageList.txt'
template = '''################## Summary #####################
Test Number: %d
Time Consume: %.03f s
FPS: %.03f
LEVEL - %d
Mean Error:
    Left Eye       = %f
    Right Eye      = %f
    Nose           = %f
    Left Mouth     = %f
    Right Mouth    = %f
Failure:
    Left Eye       = %f
    Right Eye      = %f
    Nose           = %f
    Left Mouth     = %f
    Right Mouth    = %f
'''

def evaluateError(landmarkGt, landmarkP, bbox):
    e = np.zeros(5)
    for i in range(5):
        e[i] = norm(landmarkGt[i] - landmarkP[i])
    e = e / bbox.w
    print 'landmarkGt'
    print landmarkGt
    print 'landmarkP'
    print landmarkP
    print 'error', e
    return e

def getResult(img,bbox):
    # F
    F,EN,NM = getCNNs(level = 1)
    f_bbox = bbox.subBBox(-0.05, 1.05, -0.05, 1.05)
    f_face = img[f_bbox.top:f_bbox.bottom+1,f_bbox.left:f_bbox.right+1]


    f_face = cv2.resize(f_face, (39, 39))
    en_face = f_face[:31, :]
    nm_face = f_face[8:, :]

    f_face = f_face.reshape((1, 1, 39, 39))
    f_face = processImage(f_face)
    f = F.forward(f_face)
  
    # EN
    # en_bbox = bbox.subBBox(-0.05, 1.05, -0.04, 0.84)
    # en_face = img[en_bbox.top:en_bbox.bottom+1,en_bbox.left:en_bbox.right+1]

    en_face = cv2.resize(en_face, (31, 39)).reshape((1, 1, 31, 39))
    #en_landmark = (landmarkGt[:3, :] - landmark_pre[:3,:]).reshape((6))
    en_face = processImage(en_face)
    en = EN.forward(en_face)

    #EN_imgs.append(en_face)
    #EN_landmarks.append(en_landmark)

    # NM
    # nm_bbox = bbox.subBBox(-0.05, 1.05, 0.18, 1.05)
    # nm_face = img[nm_bbox.top:nm_bbox.bottom+1,nm_bbox.left:nm_bbox.right+1]

    nm_face = cv2.resize(nm_face, (31, 39)).reshape((1, 1, 31, 39))
    #nm_landmark = (landmarkGt[2:, :] - landmark_pre[2:,:]).reshape((6))
    nm_face = processImage(nm_face)
    nm = NM.forward(nm_face)
    #NM_imgs.append(nm_face)
    #NM_landmarks.append(nm_landmark)
    landmark = np.zeros((5, 2))
    landmark[0] = (f[0]+en[0]) / 2
    landmark[1] = (f[1]+en[1]) / 2
    landmark[2] = (f[2]+en[2]+nm[0]) / 3
    landmark[3] = (f[3]+nm[1]) / 2
    landmark[4] = (f[4]+nm[2]) / 2
    return landmark
     
def E():

    data = getDataFromTxt(TXT)
    error = np.zeros((len(data), 5))
    for i in range(len(data)):
        imgPath, bbox, landmarkGt = data[i]
        img = cv2.imread(imgPath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        assert(img is not None)
        logger("process %s" % imgPath)

        landmarkP = getResult(img, bbox)

        # real landmark
        landmarkP = bbox.reprojectLandmark(landmarkP)
        landmarkGt = bbox.reprojectLandmark(landmarkGt)
        error[i] = evaluateError(landmarkGt, landmarkP, bbox)
    return error

nameMapper = ['F_test', 'level1_test', 'level2_test', 'level3_test']

if __name__ == '__main__':
  
    t = time.clock()
    error = E()
    t = time.clock() - t

    N = len(error)
    fps = N / t
    errorMean = error.mean(0)
    # failure
    failure = np.zeros(5)
    threshold = 0.05
    for i in range(5):
        failure[i] = float(sum(error[:, i] > threshold)) / N
    # log string
    s = template % (N, t, fps, 1, errorMean[0], errorMean[1], errorMean[2], \
        errorMean[3], errorMean[4], failure[0], failure[1], failure[2], \
        failure[3], failure[4])
    print s
    logfile = 'log/level1.log'
    with open(logfile, 'w') as fd:
        fd.write(s)
