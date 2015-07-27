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



class video_processing(object):
	
	def __init__(self):
		print "Initializing"
		self.step=10
		self.saving=False
		self.category=raw_input("what is the category: ")
		self.saving_time=1
		self.new_frame = False
		self.frame_count=0
		self.frame = np.zeros((640, 480), dtype = np.uint8)
		self.refPt = []

		rospy.init_node("listener")
		t = threading.Thread(name='rosbag_rate_pub', target=self.rosbag_rate_pub)
		t.start()

		self.sub = rospy.Subscriber("newtopic", Image, self.image_callback)
		self.bridge = CvBridge()
		self.image_processing()

	def image_callback(self, data):
		
		self.frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
		self.new_frame = True
		self.frame_count+=1
	def image_processing(self):
		
		cv2.namedWindow("Image")
		cv2.setMouseCallback("Image", self.click_and_crop)
		last_save = rospy.Time.now()

		while True:

			if len(self.refPt) == 2:
				cv2.rectangle(self.frame, self.refPt[0], self.refPt[1], (0, 255, 0), 2)  

			cv2.imshow("Image", self.frame)
			key = cv2.waitKey(1) & 0xFF

			if key == ord("r"):
				self.saving=False
				self.refPt=[]
			if self.saving and self.new_frame:
				self.new_frame = False
				if len(self.refPt) == 2:
						last_save = rospy.Time.now()
						roi = self.frame[self.refPt[0][1]:self.refPt[1][1], self.refPt[0][0]:self.refPt[1][0]]
						cv2.imwrite('frames/{0}-{1}-full.png'.format(self.category, self.frame_count),self.frame)
						cv2.imwrite('frames/{0}-{1}-cropped.png'.format(self.category,self.frame_count),roi)
						image_file = open("frames/{0}-{1}-location.txt".format(self.category, self.frame_count), "w")
						image_file.write(str(self.refPt))
						image_file.close()
			elif key == ord("q"):
				   cv2.destroyAllWindows()
				   break
		

	def click_and_crop(self, event, x, y, flags, param):
		
		if event == cv2.EVENT_LBUTTONDOWN:
			self.refPt = [(x, y)]
		elif event == cv2.EVENT_LBUTTONUP:
			self.refPt.append((x, y))

	def rosbag_rate_pub(self):

		self.msg=1		
		self.running = True
		t1 = threading.Thread(name='rate_controlling', target=self.run)
		t1.start()

		pub = rospy.Publisher('rosbag_rate', Float64, queue_size=10)
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
			# time.sleep(1)
		if event.Key == "e": #If the scan code matches left shift, signal that the shift button is pressed
			if self.msg<=1:
				self.msg=1
			else:
				self.msg=self.msg-1
			# time.sleep(1)
		if event.Ascii == 32: #If the ascii value matches spacebar, terminate the while loop		
			self.running =  False
		if event.Key == "s": #If the ascii value matches '4', and both ctrl and shift are pressed, run screenshot.py
			self.saving_time+=1
			if self.saving_time%2==0:
				self.saving=True
				print("saving")
				# time.sleep(1)
			else:
				self.saving=False
				self.msg=1
				print "cancel saving"
				# time.sleep(1)	
		if len(self.refPt) == 2:	
			if event.Key == "Up":																																		# if event.Key == "Up":
				self.refPt=[(self.refPt[0][0],self.refPt[0][1]-self.step),(self.refPt[1][0],self.refPt[1][1]-self.step)]
			if event.Key == "Down":
				self.refPt=[(self.refPt[0][0],self.refPt[0][1]+self.step),(self.refPt[1][0],self.refPt[1][1]+self.step)]
			if event.Key == "Right":
				self.refPt=[(self.refPt[0][0]+self.step,self.refPt[0][1]),(self.refPt[1][0]+self.step, self.refPt[1][1])]
			if event.Key == "Left":
				self.refPt=[(self.refPt[0][0]-self.step,self.refPt[0][1]),(self.refPt[1][0]-self.step,self.refPt[1][1])]
			
	
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
	object_filter=video_processing()
	