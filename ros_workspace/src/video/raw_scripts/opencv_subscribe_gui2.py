import numpy as np
import cv2
import roslib
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class object_filter(object):
	def __init__(self):
		self.time=0
		rospy.init_node("listener", anonymous = True)
		self.sub = rospy.Subscriber("newtopic", Image, self.callback)
		self.bridge = CvBridge()
		self.frame = np.zeros((640, 480), dtype = np.uint8)
	def callback(self, data):
		self.frame = self.bridge.imgmsg_to_cv2(data, "rgb8")
	def mouse_response(self):
		# self.clone=self.frame
		self.refPt = []
		cv2.namedWindow("Target")
		cv2.setMouseCallback("Target", self.click_and_crop)

		while True:
			if len(self.refPt) == 2:
				cv2.rectangle(self.frame, self.refPt[0], self.refPt[1], (0, 255, 0), 2)
			cv2.imshow("Target", self.frame)
			key = cv2.waitKey(1) & 0xFF

			if key == ord("r"):
				self.refPt=[]

			if key == ord("d"):
				if len(self.refPt) == 2:
					self.time+=1
					roi = self.frame[self.refPt[0][1]:self.refPt[1][1], self.refPt[0][0]:self.refPt[1][0]]
					cv2.imwrite('roi.png',roi)

			elif key == ord("q"):
				   cv2.destroyAllWindows()
				   break

		

	def click_and_crop(self, event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			self.refPt = [(x, y)]
		elif event == cv2.EVENT_LBUTTONUP:
			self.refPt.append((x, y))

object_filter=object_filter()
object_filter.mouse_response()