import wx
import cv2
import numpy as np
import time
from pynput.mouse import Button,Controller



cam=cv2.VideoCapture(0)
#yellow
lower=np.array([20,100,100])
upper=np.array([30,255,255])
#blue
#lower=np.array([110,50,50])
#upper=np.array([130,255,255])
kernalOpen=np.ones((5,5))
kernalClose=np.ones((20,20))


mouse=Controller()
app=wx.App(False)
(sx,sy)=wx.GetDisplaySize()
(camx,camy)=(480,320)

hold=False
firstClick=False
start=0
end=-10

while True:
    ret,img=cam.read()
    img=cv2.resize(img,(camx,camy))
    img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(img_hsv,lower,upper)
    maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernalOpen)
    maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernalClose)
    maskFinal=maskClose
    #cv2.imshow('hsv',img_hsv)
    conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    sconts=[c for c in conts if cv2.contourArea(c)>=500]
    n=len(sconts)
    if n==2:
        # for move mouse pointer
        firstClick=True
        x1,y1,w1,h1=cv2.boundingRect(sconts[0])
        x2,y2,w2,h2=cv2.boundingRect(sconts[1])
        # drawing rectangle over the objects
        cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1),(255,0,0),5)
        cv2.rectangle(img,(x2,y2),(x2+w2,y2+h2),(255,0,0),5)
        #centre coordinate of first object
        cx1=x1+w1//2
        cy1=y1+h1//2

        cx2=x2+w2//2
        cy2=y2+h2//2

        #drawing line
        cv2.line(img,(cx1,cy1),(cx2,cy2),(0,255,0),3)

        cx=(cx1+cx2)//2
        cy=(cy1+cy2)//2
        #cx,cy=cx1,cy1


        # Drawing the point (red dot) 
        cv2.circle(img, (cx,cy),2,(0,0,255),2)
        #adding the pointer
        

        (mold0,mold1)=(sx-(cx*sx/camx), cy*sy/camy)
        #mloc0=(cx-mold0)*0.5
        #mloc1=(cy-mold1)*0.5
        #mloc0,mloc1=mold0*0.8,mold1*0.8
        #mouse.release(Button.left)
        #mouse.release(Button.left)



        if hold:
            hold=False
            mouse.release(Button.left)
            end=time.time()
            print("end - start =",end-start)
            if 0.08<=end-start<=2:

                mouse.press(Button.left)
                mouse.release(Button.left)
                mouse.press(Button.left)
                mouse.release(Button.left)

        mouse.position=(mold0*0.8,mold1*0.8)

        
    
    elif n==1 and firstClick:

        x1, y1, w1, h1 = cv2.boundingRect(sconts[0])
        cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 5)
        (cx,cy)=(x1+w1//2,y1+h1//2)
        if not hold:
            #mouse.press(Button.left)
            #mouse.release(Button.left)
            mouse.press(Button.left)
            start=time.time()
            hold=True
        #mouse.release(Button.left)
        #mouse.press(Button.left)
        (posx,posy)=(sx-(cx*sx/camx), cy*sy/camy)
        mouse.position=(posx*0.8,posy*0.8)
        #mouse.release(Button.left)




    cv2.imshow('from cam',img)
    
    key=cv2.waitKey(5)
    if key==ord('q'):
        cv2.destroyAllWindows()
        break
cam.release()
