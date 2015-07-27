import numpy as np
import cv2
import roslib
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import threading
import time
import convnet_chair


class video_processing(object):
	
	def __init__(self):
		print "Initializing"

		self.cnn = convnet_chair.ConvolutionalNeuralNetwork()
		self.cnn.load_all_weights()
		self.cnn.create_model_functions()

		self.prediction = 0
		self.new_frame = 0
		self.data_frame = np.zeros((640, 480), dtype = np.uint8)
		self.display_frame = np.copy(self.data_frame)
		rospy.init_node("listener")
		t = threading.Thread(name='rosbag_rate_pub', target=self.rosbag_rate_pub)
		t.start()

		self.sub = rospy.Subscriber("newtopic", Image, self.image_callback)
		self.bridge = CvBridge()
		self.image_processing()

	def image_callback(self, data):
		self.data_frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
		self.new_frame +=1
	
	def image_processing(self):		
		cv2.namedWindow("Image")

		while True:
			self.display_frame = np.copy(self.data_frame)
	
			# cv2.imshow("Image", self.display_frame)
	
			if self.new_frame > 5:
				self.run()
				if self.new_frame > 10:
					self.new_frame = 0
			if self.prediction == 1:
				cv2.circle(self.display_frame,(40,40), 20, (0,255,0), -1)
				cv2.imshow("Image", self.display_frame)
			elif self.prediction == 2:
				cv2.circle(self.display_frame,(40,40), 20, (255,0,0), -1)
				cv2.imshow("Image", self.display_frame)
			else:
				cv2.circle(self.display_frame,(40,40), 20, (255,0,0), -1)
				cv2.imshow("Image", self.display_frame)

			key = cv2.waitKey(1) & 0xFF

			if key == ord("q"):
				cv2.destroyAllWindows()
				rospy.signal_shutdown("Q-key pressed; exiting program.")
				sys.exit(0)
			# if key == ord("a"):
			# 	self.run()
			if key == ord("w"): #If the scan code matches left control, signal that the ctrl button is pressed	
				self.msg+=1
				# time.sleep(1)
			if key == ord("e"):#If the scan code matches left shift, signal that the shift button is pressed
				if self.msg<=1:
					self.msg=1
				else:
					self.msg=self.msg-1



	def rosbag_rate_pub(self):
		self.msg = 1
		self.running = True

		pub = rospy.Publisher('rosbag_rate', Float64, queue_size=10)
		rate = rospy.Rate(15) # 10hz
		while not rospy.is_shutdown():
			if self.running:
				pub.publish(self.msg)
				rate.sleep()
	def run(self):
		self.rescale()
		self.rgb_to_gray()
		self.predict()


	def rescale(self):
		size=self.data_frame.shape
		self.img=self.data_frame
		m=10
		size=(size[1]/m,size[0]/m)
		self.img=cv2.resize(self.img,size)
		self.img = np.asarray(self.img)

	def rgb_to_gray(self):
		self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
		self.img = np.asarray(self.img)

	def predict(self):
		frame = self.img
		frame = frame.reshape(-1, 1, 48, 64)
		self.prediction = self.cnn.predict(frame)[0]
		self.prediction = np

		print("Prediction: {0}".format(["No chair", "chair","desk"][self.prediction]))
		


if __name__ == "__main__":
	object_filter=video_processing()
	