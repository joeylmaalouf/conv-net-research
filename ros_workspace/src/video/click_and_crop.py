
# import argparse
import cv2

refPt = []
cropping = False
side=100

def click_and_crop(event, x, y, flags, param):
	global refPt, cropping

	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True

	elif event == cv2.EVENT_LBUTTONUP:
		refPt.append((x, y))
		cropping = False
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", image)

def image_cropping(image):
	
	global refPt
	global cropping
	global side

	clone = image.copy()
	cv2.namedWindow("image")
	cv2.setMouseCallback("image", click_and_crop)


	while True:
		# display the image and wait for a keypress
		cv2.imshow("image", image)
		key = cv2.waitKey(1) & 0xFF

		if key == ord("r"):
			image = clone.copy()
			refPt=[]

		elif key == ord("c"):
			break
		try:
			cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		except:
			pass

	if len(refPt) == 2:
		roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
		cv2.imshow("ROI", roi)

		gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
		contours,hierarchy = cv2.findContours(gray, 1, 2)
		cnt = contours[0]


		print "cnt",cnt
		M = cv2.moments(cnt)
		print M
		cx = int(M['m10']/M['m00'])
		cy = int(M['m01']/M['m00'])

		try:	
			f_roi= clone[cy-side:cy+side, cx-side:cx+side]
			# print M
			cv2.imshow("F_ROI", f_roi)
			# cv2.saveimage('F_ROI',f_roi)
			cv2.imwrite('F_ROI.png',f_roi)
			cv2.waitKey(0)
		except:
			print "cropping area is too small"
			pass
			image_croping(image)

	cv2.destroyAllWindows()

if __name__=="__main__":
	image = cv2.imread('testing.png',-1)
	image_cropping(image)