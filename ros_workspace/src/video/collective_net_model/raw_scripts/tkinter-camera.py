import Tkinter as tk
from PIL import Image as PImage
from PIL import ImageTk
import numpy as np
import roslib
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import click_and_crop

class CameraWindow(object):
	def __init__(self, window_root):
		self.root = window_root
		self.root.bind("<Escape>", lambda e: self.root.quit())
		self.label = tk.Label(self.root, image = None)
		self.label.pack()
		self.button = tk.Button(self.root, text = "Use current frame", command = self.use_frame)
		self.button.pack()
		rospy.init_node("listener", anonymous = True)
		self.sub = rospy.Subscriber("newtopic", Image, self.callback)
		self.bridge = CvBridge()
		self.frame = np.zeros((640, 480), dtype = np.uint8)

	def callback(self, data):
		self.frame = self.bridge.imgmsg_to_cv2(data, "rgb8")

	def update_frames(self):
		imgtk = ImageTk.PhotoImage(image = PImage.fromarray(self.frame))
		self.label.configure(image = imgtk)
		self.label.image = imgtk
		self.root.after(100, self.update_frames)

	def run(self):
		self.update_frames()
		self.root.mainloop()
		self.sub.unregister()

	def use_frame(self):
		image_edit=click_and_crop.image_edit()
		image_edit.image_cropping(self.frame)


if __name__ == "__main__":
	cwin = CameraWindow(tk.Tk())
	cwin.run()
