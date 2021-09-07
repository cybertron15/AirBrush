import cv2 as cv

capture = cv.VideoCapture(0)
hands = cv.imread(r"Images\virtuals paint.png")
hands = cv.resize(hands,(640,71))
h,w,c = hands.shape
# hands = cv.resize(hands,(640,480))

while True:
    _,img = capture.read() # reading image from wecam stream
    img = cv.flip(img,1) # flipping the image
    roi = img[0:h,0:w]
    added_roi = cv.addWeighted(roi,0.1,hands,0.9,0)
    img[0:h,0:w] = added_roi
    cv.imshow("img",img)

    if cv.waitKey(1) == ord('q'):
            break

