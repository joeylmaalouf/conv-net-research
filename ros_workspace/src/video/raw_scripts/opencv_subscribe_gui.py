import numpy as np
import cv2
import roslib
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import threading
import time
import sys
sys.path.insert(0, './pyxhook')
import pyxhook



class object_filter(object):
	def __init__(self):
		self.saving=False
		self.category=raw_input("what is the category:")
		self.saving_time=1
		rospy.init_node("listener")
		t = threading.Thread(name='rosbag_rate_listener', target=self.rosbag_rate_listener)
		t.start()
		self.time=0
		self.sub = rospy.Subscriber("newtopic", Image, self.callback)
		self.bridge = CvBridge()
		self.frame = np.zeros((640, 480), dtype = np.uint8)
		self.mouse_response()
		
	def callback(self, data):
		self.frame = self.bridge.imgmsg_to_cv2(data, "rgb8")
	def mouse_response(self):
		# self.clone=self.frame
		self.refPt = []
		cv2.namedWindow("Target")
		cv2.setMouseCallback("Target", self.click_and_crop)

		while True:
			self.time+=1
			if len(self.refPt) == 2:
				cv2.rectangle(self.frame, self.refPt[0], self.refPt[1], (0, 255, 0), 2)
			cv2.imshow("Target", self.frame)
			key = cv2.waitKey(1) & 0xFF

			if key == ord("r"):
				self.refPt=[]
			if self.saving:
				if len(self.refPt) == 2:
						roi = self.frame[self.refPt[0][1]:self.refPt[1][1], self.refPt[0][0]:self.refPt[1][0]]
						cv2.imwrite('full{0}({1}) .png'.format(self.category, self.time),self.frame)
						cv2.imwrite('{0}({1}) .png'.format(self.category,self.time),roi)
						file = open("{0}({1}).txt".format(self.category, self.time), "w")
						file.write(str(self.refPt))
						file.close()
						time.sleep(1)
			elif key == ord("q"):
				   cv2.destroyAllWindows()
				   break

		

	def click_and_crop(self, event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			self.refPt = [(x, y)]
		elif event == cv2.EVENT_LBUTTONUP:
			self.refPt.append((x, y))
	def rosbag_rate_listener(self):
		# self.Rusbag_rate=Rusbag_rate()
		self.running = True
		t1 = threading.Thread(name='rate_controlling', target=self.run)
		t1.start()
		self.msg=2
			
		pub = rospy.Publisher('rosbag_rate', Float64, queue_size=10)
		# rospy.init_node('rosbag_rate')
		rate = rospy.Rate(10) # 10hz
		while not rospy.is_shutdown():
			if not self.running:
				break
			pub.publish(self.msg)
			rate.sleep()
	def keyDownEvent(self, event):
		# print "Ascii: " + str(event.Ascii) + " Scan Code: " + str(event.ScanCode) + " Key Val: " + str(event.Key)
		if event.Key == "w": #If the scan code matches left control, signal that the ctrl button is pressed	
			self.msg+=1
			time.sleep(.1)
		if event.Key == "e": #If the scan code matches left shift, signal that the shift button is pressed
			if self.msg<=0:
				self.msg=0
			else:
				self.msg-=1
			time.sleep(.1)
		if event.Ascii == 32: #If the ascii value matches spacebar, terminate the while loop		
			self.running =  False
		elif event.Key == "s": #If the ascii value matches '4', and both ctrl and shift are pressed, run screenshot.py
			self.saving_time+=1
			if self.saving_time%2==0:
				self.saving=True
				print("saving")
				time.sleep(.1)
			else:
				self.saving=False
				self.msg=0
				print "cancel saving"
				time.sleep(.1)

	def keyUpEvent(self, event):
		pass

	def run(self):
		hookman = pyxhook.HookManager()
		hookman.KeyDown = self.keyDownEvent #Bind keydown and keyup events
		hookman.KeyUp = self.keyUpEvent
		hookman.HookKeyboard()  

		hookman.start() #Start event listener
		while self.running: #Stall
			time.sleep(.1)
		hookman.cancel() #Close listener







if __name__ == "__main__":
	object_filter=object_filter()
	