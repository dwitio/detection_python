import cv2

cam = cv2.VideoCapture(0)
cam.set(3, 640) 
cam.set(4, 480) 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
fullbody_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_smile.xml')
upperbody_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')

while(True):   
    retV, frame = cam.read()   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    body = fullbody_cascade.detectMultiScale(gray, 1.1, 4)
    print(len(faces))
    
    for (x,y,w,h) in faces:
         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
         roi_gray = gray[y:y+h, x:x+w]
         roi_color = frame[y:y+h, x:x+w]
         eyes = eye_cascade.detectMultiScale(roi_gray)
         for (ex,ey,ew,eh) in eyes:
             cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
 
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)

    for (x, y, w, h) in body:
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (0, 0, 0), 5)

    cv2.imshow('WEBCAM',frame)
    close = cv2.waitKey(1) & 0xFF 
    if close == 27 or close == ord ('n') :
        break


cam.release()
cv2.destroyAllWindows()
