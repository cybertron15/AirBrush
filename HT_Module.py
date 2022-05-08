import cv2 as cv
import mediapipe as mp
import time

class Hand_Tracker():
    def __init__(self,mode=False, max_hands=2, detetctionCon=0.5, trackingCon=0.5):
        """initializes all the neccessary things"""
        self.mode=mode
        self.max_hands=max_hands
        self.detetctionCon=detetctionCon
        self.trackingCon=trackingCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detetctionCon,
            min_tracking_confidence=self.trackingCon)
        self.mpDraw = mp.solutions.drawing_utils

    
    def get_hands(self,RGB_img,img,draw_hands=True):
        """detects, traces and draws hands"""
        self.results = self.hands.process(RGB_img)
        h, w, c = img.shape

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw_hands:
                    # draws the hands on the image with connecting lines
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
        
        return img

    def find_positions(self, img, multipleHands=True, handNo=0, draw=True):
        """gets the indivisual points present of the hand"""
        # creating a landmark list to store the landmarks
        self.landmark_list = []
        if self.results.multi_hand_landmarks:
            if multipleHands:
                # detects points on multiple hands
                for handLms in self.results.multi_hand_landmarks:
                    self._get_landmarks(handLms,img,self.landmark_list,draw)
            else:
                # detects points on single hand at a time
                handLms = self.results.multi_hand_landmarks[handNo]
                self._get_landmarks(handLms,img,self.landmark_list,draw)
        
        return self.landmark_list
    
    
    def _get_landmarks(self,hand_landmarks,img,landmark_list,draw):
        h, w, c = img.shape
        for id,landmarks in enumerate(hand_landmarks.landmark):
            
            cx, cy = int(landmarks.x*w), int(landmarks.y*h)

            # appending values to landmark_list
            landmark_list.append([id,cx,cy])

            if draw:
                if id == 4 or id == 8:
                    # draws on individual points
                    cv.circle(img,(cx,cy),15,(255,255,0),-1)

    def count_fingers(self):
        """counts how many fingers are raised and returns a list of length 5
        where each element of the list corosponds to the fingers of right hand.
        it strats with the thumb and ends with pinkie. for raised finger the 
        value will be 1 else it will be 0"""
        fingers = []

        # checking thumb
        if self.landmark_list[4][1]>self.landmark_list[3][1]:
            fingers.append(1) #apeending 1 to fingers list if 
        else:
            fingers.append(0)

        # checking fingers
        for i in range(1,5):
            tips = 4*(i+1) # tips are in multiples of four so we use this logic to get each tip
            joints = tips-2 # join landmark will be always 2 positions before tip landmark of any finger

            #getting the coordinates of the landmarks we need
            y_tip = self.landmark_list[tips][2]
            y_joint = self.landmark_list[joints][2]

            # checking if y_tip is greater then y_joints which indicates the tip is below the joint
            # this means the finger is closed
            if y_tip>y_joint:
                fingers.append(0)
            else:
                fingers.append(1)

        return fingers

def main():
    capture = cv.VideoCapture(0)

    tracker = Hand_Tracker()

    previous_time = 0
    current_time = 0

    while True:
        isTrue, img = capture.read()
        RGB_img = cv.cvtColor(img,cv.COLOR_BGR2RGB)

        # FPS logic
        current_time = time.time()
        fps = 1/(current_time-previous_time)
        previous_time = current_time

        img = tracker.get_hands(RGB_img,img,True)
        landmarks = tracker.find_positions(img,draw=False)
        # if len(landmarks)!=0:
            # print(landmarks[4])
    
        # fliping the image so it doesn't get displayed inverted
        img = cv.flip(img,1)

        # displaying FPS
        cv.putText(img,f"FPS:{str(int(fps))}",(20,50),cv.FONT_HERSHEY_SIMPLEX,2,(156,222,27),2)

   
        cv.imshow("win",img)
        if cv.waitKey(1) == ord('q'):
            break

if __name__ == "__main__":
    main()
