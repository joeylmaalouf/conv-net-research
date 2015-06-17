import numpy as np
import cv2

cap = cv2.VideoCapture(0)
ret,frame = cap.read()

refPt=[(382, 273), (753, 568)]
track_window = (refPt[0][0],refPt[0][1],refPt[1][0]-refPt[0][0],refPt[1][1]-refPt[0][1])
roi = frame[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]

hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret ,frame = cap.read()

    if ret == True and cap.isOpened():
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        pts = cv2.cv.BoxPoints(ret)
        pts = np.int0(pts)
        try:
            cv2.polylines(frame,[pts],True, 255,2)
            cv2.imshow('img2',frame)
        except:
            pass
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",frame)

    else:
        break

cv2.destroyAllWindows()
cap.release()