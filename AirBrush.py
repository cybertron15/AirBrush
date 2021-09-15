import cv2 as cv
import os
import UI_components as UIC

capture = cv.VideoCapture(0)
UI_components = UIC.load_UI_components("Images")

# video Loop
while True:
    _,img = capture.read() # reading image from wecam stream
    img = cv.flip(img,1) # flipping the image

    #looping through UI components dict
    for keys,values in UI_components.items():

        #skipping if we have the mentioned componenets
        if keys in ["brushbuttonon","eraserbuttonon"]:
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
        

    cv.imshow("img",img)

    if cv.waitKey(1) == ord('q'):
            break

