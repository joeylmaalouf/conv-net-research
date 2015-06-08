import Tkinter as tk
from PIL import Image as PImage
from PIL import ImageTk
import numpy as np
import roslib
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import convnet


class VideoWindow(object):
	def __init__(self, window_root):
		super(VideoWindow, self).__init__()

		self.cnn = convnet.ConvolutionalNeuralNetwork()
		self.cnn.load_all_weights()
		self.cnn.create_model_functions()

		self.root = window_root
		self.root.bind("<Escape>", lambda e: self.root.quit())
		self.label = tk.Label(self.root, image = None)
		self.label.pack()
		self.button = tk.Button(self.root, text = "Send image to convnet as 28x28 inverted digit picture.", command = self.predict_class)
		self.button.pack()

		rospy.init_node("listener", anonymous = True)
		self.sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
		self.bridge = CvBridge()
		self.frame = np.zeros((480, 480), dtype = np.uint8)

	def callback(self, data):
		frame = self.bridge.imgmsg_to_cv2(data, "rgb8")

		w = frame.shape[1]
		h = frame.shape[0]
		if w > h:
			offset = (w-h)/2
			frame = frame[:, offset:w-offset]
		elif w < h:
			offset = (h-w)/2
			frame = frame[offset:h-offset, :]

		self.frame = frame

	def update_frames(self):
		frame = self.draw_rect(self.frame)
		imgtk = ImageTk.PhotoImage(image = PImage.fromarray(frame))
		self.label.configure(image = imgtk)
		self.label.image = imgtk
		self.root.after(100, self.update_frames)

	def draw_rect(self, frame):
		img = np.copy(frame)
		x1, x2, y1, y2 = self.get_bounds(frame)
		cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255))
		return img

	def get_bounds(self, frame):
		w = frame.shape[0]
		h = frame.shape[1]
		x1 = w/4
		x2 = 3*w/4
		y1 = h/4
		y2 = 3*h/4
		return x1, x2, y1, y2

	def manipulate(self, frame):
		x1, x2, y1, y2 = self.get_bounds(frame)
		frame = frame[y1:y2, x1:x2]
		frame = cv2.resize(frame, (28, 28))
		cv2.bitwise_not(frame, frame) # MNIST is white on black, our examples are black on white
		return frame

	def run(self):
		self.update_frames()
		self.root.mainloop()

	def predict_class(self):
		frame = self.manipulate(self.frame)
		frame = frame.reshape(-1, 1, 28, 28)
		prediction = self.cnn.predict(frame)[0]
		print("Predicted digit: {0}".format(prediction))
		return prediction


if __name__ == "__main__":
	vwin = VideoWindow(tk.Tk())
	vwin.run()
