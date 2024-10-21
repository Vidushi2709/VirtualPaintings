import cv2
import streamlit as st
import handdetectmodule as hdm
import numpy as np
import tempfile
import os

# Streamlit interface
st.title('Virtual Painter')
st.write('Use the brush or eraser to paint in the air!')

# Setting up the canvas
brushthickness = st.sidebar.slider('Brush Thickness', 5, 30, 10)
eraserthickness = st.sidebar.slider('Eraser Thickness', 30, 100, 60)

# Load header images for selection
folderpath = "C:/devdev/paintings/headers"
mylist = os.listdir(folderpath)
overlayList = [cv2.imread(f'{folderpath}/{impath}') for impath in mylist]
header = overlayList[0]

# Video capture setup
cap = cv2.VideoCapture(0)
detector = hdm.Handdetect(detectconfi=0.85)

# Create a temporary file to store the processed video frames
temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')

# Setting up OpenCV video capture with Streamlit
imgcanvas = np.zeros((480, 640, 3), np.uint8)
xp, yp = 0, 0
color = (255, 0, 255)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img = detector.findhands(img)
    lmlist = detector.findposition(img, False)

    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]

        fingers = detector.fingersup()

        # Selection mode: 2 fingers up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.rectangle(img, (x1, y1 - 30), (x2, y2 + 30), (255, 255, 0), -1)

            if y1 < 286:
                if 141 < x1 < 267:
                    header = overlayList[0]
                    color = (255, 0, 255)
                elif 268 < x1 < 300:
                    header = overlayList[1]
                    color = (255, 0, 0)
                elif 375 < x1 < 500:
                    header = overlayList[2]
                    color = (0, 255, 0)
                elif 510 < x1 < 630:
                    header = overlayList[3]
                    color = (0, 0, 0)

            cv2.rectangle(img, (x1, y1 - 30), (x2, y2 + 30), color, -1)

        # Drawing mode: 1 finger up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, (255, 255, 0), -1)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if color == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), color, eraserthickness)
                cv2.line(imgcanvas, (xp, yp), (x1, y1), color, eraserthickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), color, brushthickness)
                cv2.line(imgcanvas, (xp, yp), (x1, y1), color, brushthickness)

            xp, yp = x1, y1

    # Prepare image for display
    imggraw = cv2.cvtColor(imgcanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imggraw, 60, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgcanvas)

    # Overlay header image
    header_resized = cv2.resize(header, (img.shape[1], header.shape[0]))
    img[0:header_resized.shape[0], 0:header_resized.shape[1]] = header_resized

    # Convert frame to bytes for display in Streamlit
    frame_bytes = cv2.imencode('.jpg', img)[1].tobytes()

    # Streamlit component to display the image
    st.image(frame_bytes, channels="BGR")

    if st.sidebar.button('Stop'):
        break

cap.release()
