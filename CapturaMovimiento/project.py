import cv2
import pygame
import numpy as np

#inicia las funciones de Pygame usadadas en el codigo.
pygame.init()

#se enciende la camara web con cam
cam  = cv2.VideoCapture(0)

#si esta prendido la camara, retorna True
if cam.isOpened():
    ret,frame=cam.read()
else:
    ret = False

ret,frame1 = cam.read()
ret,frame2 = cam.read()


pista = pygame.mixer.Sound("E:/UNSA CS/2019/Arq Comp/Proyecto Final/CapturaMovimiento/pista1.wav")
pista.play()

while ret:

    d = cv2.absdiff(frame1,frame2)

    gray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray,(5,5),0)

    ret,th = cv2.threshold(blur,20,255,cv2.THRESH_BINARY)


    dilated = cv2.dilate(th, np.ones((3,3), np.uint8), iterations=10)

    erored = cv2.erode(dilated, np.ones((3,3), np.uint8), iterations=10)

    #assert isinstance(dilated, object)
    c, h = cv2.findContours(erored, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(frame1, c, -1, (0,155,255),2)

    #frame3 = frame1

    cv2.imshow("Original", frame2)
    cv2.imshow("Salida", frame1)

    key = cv2.waitKey(1)
    if key == 27:
        break
    if key == 43:
        pista.set_volume(pista.get_volume()+0.1)
        print(pista.get_volume())
    if key == 45:
        pista.set_volume(pista.get_volume()-0.1)
        print(pista.get_volume())

    frame1 = frame2
    ret,frame2 = cam.read()

cam.release()
pista.stop()
cv2.destroyAllWindows()
