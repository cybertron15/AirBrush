import cv2 as cv
import os
import numpy as np
import UI_components as UIC
import HT_Module as htm

capture = cv.VideoCapture(0)
UI_components = UIC.load_UI_components("Images")
tracker = htm.Hand_Tracker(detetctionCon=0.85,trackingCon=0.7)
disabled_buttons = ["brushbuttonon","eraserbuttonon"]
# video Loop
while True:
    _,img = capture.read() # reading image from wecam stream
    img = cv.flip(img,1) # flipping the image
    RGB_image = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    hands = tracker.get_hands(RGB_image,img,draw_hands=True)
    landmarks = tracker.find_positions(hands,draw=False)
    
    #looping through UI components dict
    for keys,values in UI_components.items():

        #skipping if we have the mentioned componenets
        if keys in disabled_buttons:
            continue

        resized = values[2]
        bg_mask = values[4]
        component = values[0]

        if resized:
            component = cv.resize(component,resized)

        x_pos,y_pos = values[1][0],values[1][1]
        height,width,channels = component.shape
        masked_componenet = values[3]
            
        roi = img[y_pos:y_pos+height , x_pos:x_pos+width]#getting ROI
        masked_roi = cv.bitwise_and(roi,roi,mask=bg_mask)# masking ROI(removing UI element part and keeping rest)

        # adding both the images and placeing it on the exact same place where we took ROI from
        img[y_pos:y_pos+height , x_pos:x_pos+width] = cv.add(masked_componenet,masked_roi)
    

    if landmarks:
        fingers = tracker.count_fingers()
        x1,y1 = landmarks[12][1],landmarks[12][2]
        x2,y2 = landmarks[8][1],landmarks[8][2]

        dist = np.hypot(x2-x1,y2-y1)
        
        if fingers[1] and not fingers[2]:
            cv.putText(img,"Draw mode",(10,470),cv.FONT_HERSHEY_PLAIN,1.2,(255,225,255),2)
            cv.circle(img,(x2,y2),5,(130,255,110),-1)

            
        if fingers[1] and fingers[2]:
            cv.putText(img,"Selection mode",(10,470),cv.FONT_HERSHEY_PLAIN,1.2,(255,225,255),2)
            dist = np.hypot(x2-x1,y2-y1)
            cv.circle(img,(x2,y2),5,(10,20,210),2)
            
            # print(UI_components["eraserbuttonon"][1])
            # print(UI_components["eraserbuttonon"][0].shape)

            if 5+56>x2>5 and 150+49>y2>150:
                disabled_buttons[0] = "brushbuttonoff"
                disabled_buttons[1] = "eraserbuttonon"
            
            if 5+56>x2>5 and 210+49>y2>210:
                disabled_buttons[0] = "brushbuttonon"
                disabled_buttons[1] = "eraserbuttonoff"
        
    
    cv.imshow("img",img)

    if cv.waitKey(1) == ord('q'):
            break

