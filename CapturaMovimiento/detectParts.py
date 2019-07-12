import cv2

face_csc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
low_csc = cv2.CascadeClassifier('haarcascade_lowerbody.xml')
body_csc = cv2.CascadeClassifier('haarcascade_fullbody.xml')
#img = cv2.imread('bailarin.JPG')
#img = cv2.imread('op.jpeg')
#img = cv2.imread('kok.jpeg')
img = cv2.imread('lolol.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face = face_csc.detectMultiScale(gray,1.3,5)
low = low_csc.detectMultiScale(gray,1.1,5)
bodies = body_csc.detectMultiScale(gray,1.1,5)

for (x,y,w,h) in face:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    #roi_gray = gray[y:y+h, x:x+w]
    #roi_color = img[y:y+h, x:x+w]
for (x,y,w,h) in low:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    #roi_gray = gray[b:b+d, a:a+c]
    #roi_color = img[b:b+d, a:a+c]

for (x,y,w,h) in bodies:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #roi_gray = gray[y:y+h, x:x+w]
    #roi_color = img[y:y+h, x:x+w]

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
