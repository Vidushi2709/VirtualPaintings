import cv2
import time
import mediapipe as mp
hands_module = mp.solutions.hands #type: ignore
drawing_utils = mp.solutions.drawing_utils #type: ignore
class Handdetect():
    def __init__(self, mode=False, maxhands=2, detectconfi=0.5, trackconfi=0.5):
        self.mode = mode
        self.maxhands = maxhands
        self.detectconfi = detectconfi
        self.trackconfi = trackconfi
        self.mpHands = hands_module
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxhands,
                                        min_detection_confidence=self.detectconfi,
                                        min_tracking_confidence=self.trackconfi)
        self.mpdraw = drawing_utils
        self.results = None 
        self.tipIds=[4,8,12,16,20]

    def findhands(self, img, draw=True):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findposition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results and self.results.multi_hand_landmarks:  # Check if self.results is not None
            if handNo < len(self.results.multi_hand_landmarks):  # Ensure handNo is within the range
                hand_landmarks = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lmList.append((id, cx, cy))
                    if draw:
                        cv2.circle(img, (cx, cy), 1, (255, 0, 255), cv2.FILLED)
        return self.lmList 
    def fingersup(self):
        fingers=[] #empty list
       
        #thumb if tip of thumb is up or no
        if self.lmList[self.tipIds[0]][1]<self.lmList[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        #for 4 fingers
        for id in range(1,5):
            if self.lmList[self.tipIds[id]][2]<self.lmList[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers;
                      
def main():
    pTime = 0
    cTime = 0
    detect = Handdetect()
    vidz = cv2.VideoCapture(0)

    while True:
        success, img = vidz.read()
        if not success or img is None:
            print("Failed to capture image")
            break
        
        img = detect.findhands(img)
        lmList = detect.findposition(img,True)
        if len(lmList) != 0:  
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
        cv2.imshow("hands", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    vidz.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
