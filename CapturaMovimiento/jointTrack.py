import numpy as np
import cv2

RED = (55,75,255)

LRangeBlue = np.array([100,0,0])
URangeBlue = np.array([255,255,100])

dimx = 640
dimy = 480

face_csc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
body_csc = cv2.CascadeClassifier('haarcascade_fullbody.xml')
lower_body_csc = cv2.CascadeClassifier('haarcascade_lowerbody.xml')
upper_body_csc = cv2.CascadeClassifier("haarcascade_upperbody.xml")

class Joint:
    def __init__(self,x=0,y=0):
        self.x=x
        self.y=y

def gatherPoints(frame,lowerRange,upperRange):
    mask = cv2.inRange(frame,lowerRange,upperRange)
    active = []
    for row in range(0,dimx,10):
        for col in range(0,dimy,10):
            if mask[col,row] == 255:
                active.append([row,col])
    return mask,active

def findJoint_AverageMethod(maskList):
    ave = averagePoint(maskList)
    joint = Joint(x=ave[0],y=ave[1])
    return  joint

def findJoint_FaceMethod(gray):
    cara = face_csc.detectMultiScale(gray)
    for(x,y,w,h) in cara:
        return Joint(x+int(w/2),y+int(h/2))
    return Joint()

def drawJoint(joint,frame):
    if joint.x!=0 or joint.y!=0:
        cv2.circle(frame,(joint.x,joint.y),10,RED,2)
def averagePoint(list):
    totalx = 0
    totaly = 0
    count = 0

    for item in list:
        totalx += item[0]
        totaly += item[1]
        count+=1
    if count>0:
        return [int(totalx/count),int(totaly/count)]
    else:
        return [0,0]

'''
LRangeBlk = np.array([0,0,0])
URangeBlk = np.array([2,2,2])

LRangeBlu = np.array([75,0,0])
URangeBlu = np.array([255,150,100])

LRangeOra = np.array([0,0,255])
URangeOra = np.array([200,200,255])

LRangeHSVBlu = np.array([100,0,0])
URangeHSVBlu = np.array([120,255,100])

jointBlk = Joint(lowerRange=LRangeBlk,upperRange=URangeBlk,type='color',colorName='black')
Joint.totalJointList.append(joinBlk)

jointOra = Joint(lowerRange=LRangeHSVBlu,upperRange=URangeHSVBlu)
'''
cam = cv2.VideoCapture(0)

while(True):
    ret,frame = cam.read()
    frame = cv2.resize(frame,(dimx,dimy))

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    jointHead = findJoint_FaceMethod(gray)
    maskBlue, maskListBlue = gatherPoints(frame, LRangeBlue, URangeBlue)
    jointBlue = findJoint_AverageMethod(maskListBlue)

    drawJoint(jointBlue, frame)
    drawJoint(jointHead, frame)

    cv2.imshow('Skeleton 1.0', frame)
    if ord('q') == cv2.waitKey(1) & 0xFF:
        break
cam.release()
cv2.destroyAllWindows()