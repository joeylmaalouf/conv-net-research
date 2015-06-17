import cv2
import cv2.cv as cv 
from PIL import Image
import numpy as np
import sys 

class image_edit():

	def __init_(self):
		pass

	def image_cropping(self, image, side=100):
		self.image=image
		self.image=cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
		self.side=side
		self.refPt = []

		self.clone = self.image.copy()
		cv2.namedWindow("Target")
		cv2.setMouseCallback("Target", self.click_and_crop)

		while True:
			cv2.imshow("Target", self.image)
			key = cv2.waitKey(1) & 0xFF

			if key == ord("r"):
				self.image = self.clone
				cv2.imshow("Target", self.image)
				self.refPt=[]
				cv2.destroyAllWindows()
				self.image_cropping(self.image)
			elif key == ord("c"):
				break
			elif key == ord("q"):
				   cv2.destroyAllWindows()
				   break

		if len(self.refPt) == 2:
			roi = self.clone[self.refPt[0][1]:self.refPt[1][1], self.refPt[0][0]:self.refPt[1][0]]
			cv2.imshow("ROI", roi)

			gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
			contours,hierarchy = cv2.findContours(gray, 1, 2)
			cnt = contours[0]
			M = cv2.moments(cnt)
			cx = int(M['m10']/M['m00'])
			cy = int(M['m01']/M['m00'])

			try:	
				f_roi= self.clone[self.refPt[0][1]+cy-self.side:self.refPt[0][1]+cy+self.side, self.refPt[0][0]+cx-self.side:self.refPt[0][0]+cx+self.side]
				cv2.imshow("F_ROI", f_roi)
				cv2.imwrite('F_ROI.png',f_roi)
				cv2.waitKey(0)
			except:
				print "cropping area is too smaqll"
				self.image_cropping(self.image)

		cv2.destroyAllWindows()

	def click_and_crop(self, event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			self.refPt = [(x, y)]
		elif event == cv2.EVENT_LBUTTONUP:
			self.refPt.append((x, y))
			cv2.rectangle(self.image, self.refPt[0], self.refPt[1], (0, 255, 0), 2)
			print self.refPt

if __name__=="__main__":
	image_edit=image_edit()
	image = cv2.imread('testing.png',-1)
	image_edit.image_cropping(image)