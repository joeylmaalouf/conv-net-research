import numpy as np
import cv2


class object_filter():
    def __init_(self):
        pass
    def camshift_tracking(self):

        self.cap = cv2.VideoCapture(0)
        self.ret,self.frame = self.cap.read()
        self.refPt=self.mouse_response()
        print self.refPt
        # self.refPt=[(382, 273), (753, 568)]
        track_window = (self.refPt[0][0],self.refPt[0][1],self.refPt[1][0]-self.refPt[0][0],self.refPt[1][1]-self.refPt[0][1])
        self.roi = self.frame[self.refPt[0][1]:self.refPt[1][1], self.refPt[0][0]:self.refPt[1][0]]
        hsv_roi =  cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
        cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

        while(1):
            self.ret , self.frame = self.cap.read()

            if self.ret == True and self.cap.isOpened():
                hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
                dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

                
                # ret, track_window = cv2.meanShift(dst, track_window, term_crit)
                # x,y,w,h = track_window
                # self.refPt=[(x,y), (x+w,y+h)]
                # cv2.rectangle(self.frame, (x,y), (x+w,y+h), 255,2)
                # cv2.imshow('img2',self.frame)

                ret, track_window = cv2.CamShift(dst, track_window, term_crit)
                pts = cv2.cv.BoxPoints(ret)
                pts = np.int0(pts)
                cv2.polylines(self.frame,[pts],True, 255,2)
                cv2.imshow('img2',self.frame)
                

                key = cv2.waitKey(60) & 0xff
                if key == 27:
                    break
                if key == ord("s"):
                    if len(self.refPt) == 2:
                        roi = self.frame[self.refPt[0][1]:self.refPt[1][1], self.refPt[0][0]:self.refPt[1][0]]
                        cv2.imwrite('roi.png',roi)
                if key == ord("r"):
                    self.camshift_tracking()
                else:
                    cv2.imwrite(chr(key)+".jpg",self.frame)

            else:
                break

        cv2.destroyAllWindows()
        cap.release()




    def mouse_response(self):
        self.image=self.frame
        self.clone=self.image
        self.refPt = []
        cv2.namedWindow("Target")
        cv2.setMouseCallback("Target", self.click_and_crop)

        while True:
            self.ret,self.frame = self.cap.read()
            self.image=self.frame
            if len(self.refPt) == 2:
                cv2.rectangle(self.image, self.refPt[0], self.refPt[1], (0, 255, 0), 2)
            cv2.imshow("Target", self.image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):
                self.image = self.clone
                cv2.imshow("Target", self.image)
                self.refPt=[]
                cv2.destroyAllWindows()
                self.image_cropping(self.image)
            if key == ord("d"):
                if len(self.refPt) == 2:
                    roi = self.clone[self.refPt[0][1]:self.refPt[1][1], self.refPt[0][0]:self.refPt[1][0]]
                    cv2.imwrite('roi.png',roi)
                    cv2.destroyAllWindows()
                    return self.refPt

            elif key == ord("q"):
                   cv2.destroyAllWindows()
                   break

        

    def click_and_crop(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.refPt = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            self.refPt.append((x, y))

object_filter=object_filter()
object_filter.camshift_tracking()