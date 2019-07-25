import cv2
import numpy as np
import wx
from pynput.mouse import Button,Controller

app=wx.App(False)
kernalOpen=np.ones((5,5))
kernalClose=np.ones((20,20))
b_min=np.array([20,100,100])
b_max=np.array([30,255,255])
mouse=Controller()
(sx,sy)=wx.GetDisplaySize()
flag=False
cam=cv2.VideoCapture(0)
while True:
    ret,frame=cam.read()
    (camx,camy)=(frame.shape[0],frame.shape[1])


    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,b_min,b_max)

    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernalOpen)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernalClose)


    conts,_=cv2.findContours(maskClose,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    #sort the contours
    #sconts=sorted(conts,key=cv2.contourArea)
    #big2=sconts[-2:]
    #we can draw the contours
    #cv2.drawContours(frame,big2,-1,(0,0,255),3)
    #let us draw rectangles for them
    #x1,x2,y1,y2,w1,w2,h1,h2=0,0,0,0,0,0,0,0
    big2=[]
    for cnt in conts:
        if cv2.contourArea(cnt)>=100:
            big2+=[cnt]
    n = len(big2)
    if n==2 :

        #cv2.drawContours(frame, big2, -1, (0, 255, 0), 3)
        x1, y1, w1, h1 = cv2.boundingRect(big2[0])
        x2, y2, w2, h2 = cv2.boundingRect(big2[1])
        cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
        cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2
        cv2.line(frame, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2)
        cx, cy = (cx1 + cx2) // 2, (cy1 + cy2) // 2
        cv2.circle(frame, (cx, cy), 7, (255, 0, 0), -1)
        (mold0, mold1) = (sx - (cx * sx / camx), cy * sy / camy)
        mouse.position = (mold0, mold1)
        flag=True
    elif n==1 and flag==True:
        if cv2.contourArea(big2[0])>300:
            x1, y1, w1, h1 = cv2.boundingRect(big2[0])
            cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 3)
            mouse.press(Button.left)
            flag=False

    #cv2.imshow("masked",mask)
    #cv2.imshow('mopen',maskOpen)
    #cv2.imshow('mclose',maskClose)
    cv2.imshow("img",frame)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
cv2.destroyAllWindows()
cam.release()
