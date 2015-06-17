import cv2
import cv2.cv as cv 
from PIL import Image
import numpy as np
import sys 

class image_edit():

	def __init_(self):
		pass
	def image_cropping(self,image):
		self.image=image
		self.clone=self.image
		self.refPt = []
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
			if key == ord("c"):
				if len(self.refPt) == 2:
					roi = self.clone[self.refPt[0][1]:self.refPt[1][1], self.refPt[0][0]:self.refPt[1][0]]
					cv2.imwrite('roi.png',roi)
					return self.refPt
			elif key == ord("q"):
				   cv2.destroyAllWindows()
				   break

		cv2.destroyAllWindows()

	def click_and_crop(self, event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			self.refPt = [(x, y)]
		elif event == cv2.EVENT_LBUTTONUP:
			self.refPt.append((x, y))
			cv2.rectangle(self.image, self.refPt[0], self.refPt[1], (0, 255, 0), 2)

if __name__=="__main__":
	image_edit=image_edit()
	image = cv2.imread('testing.png',-1)
	image_edit.image_cropping(image)