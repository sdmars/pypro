import cv2,time
#vfile="C:\\Users\SAYAN\Desktop\mlpro\\vdo1.mp4"
vdo=cv2.VideoCapture(0)
xml_file="C:\\Users\SAYAN\Desktop\mlpro\\front_face_detector.xml"
face_cascade=cv2.CascadeClassifier(xml_file)
out=cv2.VideoWriter('output.mp4',-1,20.0,(640,480))
while True:
    check,frame=vdo.read()
    #convert into a bw image
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=8)

    for x,y,w,h in faces:
        frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

    cv2.imshow("frames",frame)
    out.write(frame)
    #gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #cv2.imshow("frames",frame)
    key=cv2.waitKey(1)

    if key==ord('e'):
        break

vdo.release()
out.release()
cv2.destroyAllWindows()
