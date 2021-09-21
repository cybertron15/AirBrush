import cv2 as cv
import os
import numpy as np
import UI_components as UIC
import HT_Module as htm

capture = cv.VideoCapture(0)
frameW,frameH = int(capture.get(3)),int(capture.get(4))

UI_components = UIC.load_UI_components("Images")
tracker = htm.Hand_Tracker(detetctionCon=0.85,trackingCon=0.7)
disabled_buttons = ["brushbuttonon","eraserbuttonon","numrecogon","clearon"]
canvas = np.zeros((frameH,frameW,3),np.uint8)
xp,yp=0,0

start,end = [],[]
scan = []
scanned_can = None

# video Loop
while True:
    _,img = capture.read() # reading image from wecam stream
    img = cv.flip(img,1) # flipping the image
    RGB_image = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    hands = tracker.get_hands(RGB_image,img,draw_hands=False)
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

        
        # adjusting diffrent modes
        if fingers[1] and not fingers[2]:# if only index finger is up and middle is down
            cv.putText(img,"Draw mode",(10,470),cv.FONT_HERSHEY_PLAIN,1.2,(255,225,255),2)
            cv.circle(img,(x2,y2),5,(130,255,110),-1)
            
            # to prevent the line from starting from thr top right corner
            if xp==0 and yp == 0:
                xp,yp=x2,y2
            
            #drawing
            if "brushbuttonoff" in disabled_buttons:
                cv.line(canvas,(xp,yp),(x2,y2),(50,25,110),10)
                if not start:
                    start.append(x2)
                    start.append(y2)

            #ereasing
            if "eraserbuttonoff" in disabled_buttons:
                cv.line(canvas,(xp,yp),(x2,y2),(0,0,0),50)
            
            xp,yp=x2,y2
            

        if fingers[1] and fingers[2]: #if index and middle both fingers are up

            if start and not end:
                end.append(x2)
                end.append(y2)
                scan.append([start,end])
                start = []
                end = []


            #resetting the drawing position
            xp,yp = 0,0

            # Displaying Selection mode on screen
            cv.putText(img,"Selection mode",(10,470),cv.FONT_HERSHEY_PLAIN,1.2,(255,225,255),2)
            cv.circle(img,(x2,y2),5,(10,20,210),2)

            #selecting buttons
            if 5+56>x2>5 and 150+49>y2>150: # selecting brush button
                disabled_buttons[0] = "brushbuttonoff"
                disabled_buttons[1] = "eraserbuttonon"
                disabled_buttons[2] = "numrecogon"
                disabled_buttons[3] = "clearon"
            
            if 5+56>x2>5 and 210+49>y2>210: #selecting eraser button
                disabled_buttons[0] = "brushbuttonon"
                disabled_buttons[1] = "eraserbuttonoff"
                disabled_buttons[2] = "numrecogon"
                disabled_buttons[3] = "clearon"
            
            if 5+56>x2>5 and 270+49>y2>270: #selecting Number recogniser button
                disabled_buttons[0] = "brushbuttonon"
                disabled_buttons[1] = "eraserbuttonon"
                disabled_buttons[2] = "numrecogoff"
                disabled_buttons[3] = "clearon"

            if 5+56>x2>5 and 330+49>y2>330: #selecting clear screen button
                disabled_buttons[0] = "brushbuttonon"
                disabled_buttons[1] = "eraserbuttonon"
                disabled_buttons[2] = "numrecogon"
                disabled_buttons[3] = "clearoff"

                canvas = np.zeros((frameH,frameW,3),np.uint8) #this will clear the screen

        
    #drawing on screen
    gray_canvas = cv.cvtColor(canvas,cv.COLOR_BGR2GRAY)
    _, inv_canvas_tresh = cv.threshold(gray_canvas,50,255,cv.THRESH_BINARY_INV)
    # cv.imshow("can",inv_canvas_tresh)
    if scan:
        print(scan)
        for index, coordinates in enumerate(scan):
            x1,x2 = coordinates[0][0], coordinates[1][0]
            y1,y2 = coordinates[0][1], coordinates[1][1]
            print(y1,y2)
            print(x1,x2)
            cv.imshow(f"{index}",inv_canvas_tresh[y1:y2,x1:x2])

    inv_canvas_tresh = cv.cvtColor(inv_canvas_tresh,cv.COLOR_GRAY2BGR)
    img = cv.bitwise_and(img,inv_canvas_tresh)
    img = cv.bitwise_or(img,canvas)

    cv.imshow("img",img)
    # cv.imshow("canvas",canvas)

    if cv.waitKey(1) == ord('q'):
        break

