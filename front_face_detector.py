import cv2
xml_file="C:\\Users\SAYAN\Desktop\mlpro\\front_face_detector.xml"
face_cascade=cv2.CascadeClassifier(xml_file)

img_file="C:\\Users\SAYAN\Desktop\mlpro\\dev.jpg"
img=cv2.imread(img_file)

gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=face_cascade.detectMultiScale(gray_img,scaleFactor=1.05,minNeighbors=5)

print(type(faces))
print(faces)

for x,y,w,h in faces:
    img=cv2.rectangle(img,(x,y),(x+w,y+h),(124,78,1),3)

cv2.imshow("Gray",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
