import cv2
import time
import os
import handdetectmodule as hdm
import numpy as np

#import images
folderpath="C:/devdev/paintings/headers"
mylist=os.listdir(folderpath)
#print(mylist)
color=(255,0,255)

brushthickness=10
eraserthickness=60

xp,yp=0,0

imgcanvas=np.zeros((480, 640,3), np.uint8)

overlayList=[]

#read the file
for impath in mylist:
    image=cv2.imread(f'{folderpath}/{impath}')
    overlayList.append(image)
#print(len(overlayList))
header=overlayList[0] 

cap=cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Resize the header image to match the frame width
header = cv2.resize(header, (frame_width, int(frame_height * 0.4)))  # Assuming header height is 40% of frame height

detector=hdm.Handdetect(detectconfi=0.85)

while True:
    #import image
    success, img=cap.read()
    #flip image: mirror image
    '''h,w,c=img.shape
    print(h)
    print(w)
    print(c)''' # to check height, width and channels of original camera
    img=cv2.flip(img,1)
    #finding landmarks
    img=detector.findhands(img)

    lmlist=detector.findposition(img, False)
    if len(lmlist)!=0:

        #print(lmlist)
        #tip of index and middle fingers
        x1,y1=lmlist[8][1:]
        x2,y2=lmlist[12][1:]

        fingers=detector.fingersup()
        #print(fingers)
        #print(f"Index finger tip coordinates (x1, y1): ({x1}, {y1})")

        #if select mode=2 fingers up
        if fingers[1] and fingers[2]:
            xp,yp=0,0
            #visually saying selection
            cv2.rectangle(img, (x1,y1-30), (x2,y2+30),(255,255,0),-1)
            print("Selection mode")
            #print(f"x1 value: {x1}")
            if y1<286:
                if 141<x1<267:
                    header=overlayList[0]
                    color=(255,0,255)
                elif 268<x1<300:
                    header=overlayList[1]
                    color=(255,0,0)
                elif 375<x1<500:
                    header=overlayList[2];
                    color=(0,255,0)
                elif 510<x1<630:
                    header=overlayList[3]
                    color=(0,0,0)
            
            cv2.rectangle(img, (x1,y1-30), (x2,y2+30),color,-1)

        #if draw mode=1 finger up(index finger)
        if fingers[1] and fingers[2]==False:
            #visually saying drawing

           
            cv2.circle(img, (x1,y1), 15, (255,255,0), -1)
            print("drawing mode")
            if xp==0  and yp==0:
                xp,yp=x1,y1
            if color==(0,0,0):
                cv2.line(img,(xp,yp),(x1,y1), color, eraserthickness)
                cv2.line(imgcanvas,(xp,yp),(x1,y1), color, eraserthickness)
            else:
                cv2.line(img,(xp,yp),(x1,y1), color, brushthickness)
                cv2.line(imgcanvas,(xp,yp),(x1,y1), color, brushthickness)
            
            xp,yp=x1,y1
    imggraw=cv2.cvtColor(imgcanvas, cv2.COLOR_BGR2GRAY)
        #convert to binary
    _, imgInv=cv2.threshold(imggraw,60,255, cv2.THRESH_BINARY_INV)
        #we are converting img in gray
        #whereever we are coloring in black canvas is getting inversed
        #on camera img, that place turns black and we see colors
    imgInv=cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img=cv2.bitwise_and(img, imgInv) #we are adding both images
    img=cv2.bitwise_or(img, imgcanvas)

        #overlaying the image
        #w=286
        #h=1920
    # Ensure header is resized if necessary to match img dimensions
    header_resized = cv2.resize(header, (img.shape[1], header.shape[0]))

    # Overlay header image
    img[0:header_resized.shape[0], 0:header_resized.shape[1]] = header_resized
    img=cv2.addWeighted(img,0.5,imgcanvas,0.5,0)

    # Display the image
    cv2.imshow("Image", img)
    cv2.imshow("Image2", imgcanvas)

    if cv2.waitKey(1) & 0xFF == ord('v'):
        break