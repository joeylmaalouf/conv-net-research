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
from os.path import exists
import subprocess


class video_processing(object):
	
	def __init__(self):
		print "Initializing"
		self.index = 0 
		self.colors = [(0,0,255),(255,0,0),(0,255,0)]
		self.flag=["out","half",""]
		self.step = 10
		self.saving = False
		self.category = raw_input("what is the category: ")
		self.saving_time = 1
		self.new_frame = False
		self.frame_count = 0
		self.data_frame = np.zeros((640, 480), dtype = np.uint8)
		self.display_frame = np.copy(self.data_frame)
		self.refPt = []
		# self.time = 0
		rospy.init_node("listener")
		t = threading.Thread(name='rosbag_rate_pub', target=self.rosbag_rate_pub)
		t.start()

		self.sub = rospy.Subscriber("newtopic", Image, self.image_callback)
		self.bridge = CvBridge()
		self.image_processing()

	def image_callback(self, data):
		self.data_frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
		self.new_frame = True
		self.frame_count += 1

	
	def image_processing(self):
		
		cv2.namedWindow("Image")
		cv2.setMouseCallback("Image", self.click_and_crop)
		last_save = rospy.Time.now()

		while True:
			self.display_frame = np.copy(self.data_frame)
			if self.saving:
				img = cv2.circle(self.display_frame,(50,50), 20, (0,0,255), -1)
			if len(self.refPt) >= 2:
				self.dx=(self.refPt[1][0]-self.refPt[0][0])/2
				self.dy=(self.refPt[1][1]-self.refPt[0][1])/2
				self.refPt=[(self.mx-self.dx,self.my-self.dy),(self.mx+self.dx,self.my+self.dy)]

				self.check_dimension()
				cv2.rectangle(self.display_frame, self.refPt[0], self.refPt[1], self.colors[self.index], 2)  
			
					  
			cv2.imshow("Image", self.display_frame)
			key = cv2.waitKey(1) & 0xFF

			if key == ord("r"):
				self.saving=False
				self.refPt=[]
				self.msg=1


			if self.saving and self.new_frame:
				self.new_frame = False
				if len(self.refPt) == 2:

						last_save = rospy.Time.now()


						# if self.refPt[0][0]>640:
						# 	self.refPt=[(640,self.refPt[0][1]),(640,self.refPt[1][1])]
					
						# 	self.refPt=[(0,self.refPt[0][1]),(0,self.refPt[1][1])]
						# if self.refPt[1][0]>640:
						# 	self.refPt=[(self.refPt[0][0],self.refPt[0][1]),(640,self.refPt[1][1])]


						# # if self.refPt[0][1]<0:
						# # 	self.refPt=[(self.refPt[0][0],0),(self.refPt[1][0],self.refPt[1][1])]
						# # if self.refPt[0][1]>480:
						# # 	self.refPt=[(self.refPt[0][0],480),(self.refPt[1][0],480)]
						# # if self.refPt[1][1]<0:
						# # 	self.refPt=[(self.refPt[0][0],0),(self.refPt[1][0],0)]
						# # if self.refPt[1][1]>480:
						# # 	self.refPt=[(self.refPt[0][0],self.refPt[0][1]),(self.refPt[1][0],480)]
						print self.refPt

						roi = self.data_frame[self.refPt[0][1]:self.refPt[1][1], self.refPt[0][0]:self.refPt[1][0]]
						if not exists("data_set/{0}".format(self.category)):
							print(self.category)
							subprocess.call(["mkdir", "data_set/{0}".format(self.category)])
	
						# if self.out:
						# 	self.check_dimension()
						# 	cv2.imwrite('data_set/{0}/out_{0}-{1}-full.png'.format(self.category, self.frame_count), self.data_frame)
						# 	cv2.imwrite('data_set/{0}/out_{0}-{1}-cropped.png'.format(self.category,self.frame_count), roi)
						# 	image_file = open("data_set/{0}/out_{0}-{1}-location.txt".format(self.category, self.frame_count), "w")
						# 	image_file.write(str(self.refPt))
						# 	image_file.close()

						# else:
						if self.flag[self.index]!="half":
							self.check_dimension()
							cv2.imwrite('data_set/{0}/{2}_{0}-{1}-full.png'.format(self.category, self.frame_count, self.flag[self.index]), self.data_frame)
							# cv2.imwrite('data_set/{0}/{2}_{0}-{1}-cropped.png'.format(self.category,self.frame_count, self.flag[self.index]), roi)
							# image_file = open("data_set/{0}/{2}_{0}-{1}-location.txt".format(self.category, self.frame_count, self.flag[self.index]), "w")
							# image_file.write(str(self.refPt))
							# image_file.close()
							

			if key == ord("w"): #If the scan code matches left control, signal that the ctrl button is pressed	
				self.msg+=1
				# time.sleep(1)
			if key == ord("e"):#If the scan code matches left shift, signal that the shift button is pressed
				if self.msg<=1:
					self.msg=1
				else:
					self.msg=self.msg-1
				# time.sleep(1)
			if key == 32: #If the ascii value matches spacebar, terminate the while loop		
				self.running =  False
			if key == ord("s"): #If the ascii value matches '4', and both ctrl and shift are pressed, run screenshot.py
				self.saving_time+=1
				if self.saving_time%2==0:
					self.saving=True
					print("saving")
					# time.sleep(1)
				else:
					self.saving=False
					print "cancel saving"
			if len(self.refPt) >= 2:
				if key ==82:																																		# if event.Key == "Up":
					self.refPt=[(self.refPt[0][0],self.refPt[0][1]-self.step),(self.refPt[1][0],self.refPt[1][1]+self.step)]
				if key ==84:
					self.refPt=[(self.refPt[0][0],self.refPt[0][1]+self.step),(self.refPt[1][0],self.refPt[1][1]-self.step)]
				if key == 83:
					self.refPt=[(self.refPt[0][0]-self.step,self.refPt[0][1]),(self.refPt[1][0]+self.step, self.refPt[1][1])]
				if key ==81:
					self.refPt=[(self.refPt[0][0]+self.step,self.refPt[0][1]),(self.refPt[1][0]-self.step,self.refPt[1][1])]
				self.check_dimension()

			if key == ord("z"):
				self.index = 0
			if key == ord("x"):
				self.index = 1
			if key == ord("c"):
				self.index = 2
			
				# 	self.out=False
				# elif self.out_time%3==0:
				# 	self.out=True
				# 	self.half=False
				# 	print("object out of frame")
				# else:
				# 	self.out=False
				# 	print("object inside of frame")

			# if key == ord("z"):
			# 	self.half_time+=1
			# 	if self.half_time%2==0:
			# 		self.half=True
			# 		print("object half of frame")
			# 	else:
			# 		# self.out=True
			# 		self.half=False
			# 		self.out= not self.out
			# 		print("object not half of frame")

			if key == ord("q"):
				cv2.destroyAllWindows()
			  	rospy.signal_shutdown("Q-key pressed; exiting program.")
				sys.exit(0)



	def check_dimension(self):

		if self.refPt[0][0]<0 or self.refPt[1][0]<0:
			self.refPt=[(0,self.refPt[0][1]),(2*(self.dx),self.refPt[1][1])]
		if self.refPt[0][0]>640 or self.refPt[1][0]>640:
			self.refPt=[(640-2*(self.dx),self.refPt[0][1]),(640,self.refPt[1][1])]
		
		if self.refPt[0][1]<0 or self.refPt[1][1]<0:
			self.refPt=[(self.refPt[0][0],0),(self.refPt[1][0],2*(self.dy))]
		if self.refPt[0][1]>480 or self.refPt[1][1]>480:
			self.refPt=[(self.refPt[0][0],480-2*(self.dy)),(self.refPt[1][0],480)]


	def click_and_crop(self, event, x, y, flags, param):
		
		if event == cv2.EVENT_LBUTTONDOWN:
			self.refPt = [(x, y)]
		if event == cv2.EVENT_LBUTTONUP:
			self.refPt.append((x, y))
		elif event == cv2.EVENT_MOUSEMOVE:
			self.mouse_cursur=[x,y]
			self.mx=x
			self.my=y

	def rosbag_rate_pub(self):
		self.msg = 1
		self.running = True
		# t1 = threading.Thread(name='rate_controlling', target=self.run)
		# t1.start()

		pub = rospy.Publisher('rosbag_rate', Float64, queue_size=10)
		rate = rospy.Rate(15) # 10hz
		while not rospy.is_shutdown():
			if self.running:
				pub.publish(self.msg)
				rate.sleep()

if __name__ == "__main__":
	object_filter=video_processing()
	