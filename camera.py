import Tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import cv2


class VideoWindow(object):
	def __init__(self, window_root, video_capture):
		super(VideoWindow, self).__init__()
		self.root = window_root
		self.cap = video_capture
		root.bind("<Escape>", lambda e: root.quit())
		self.label = tk.Label(self.root, image = None)
		self.label.pack()
		self.button = tk.Button(self.root, text = "Send image to convnet as 28x28 digit picture.", command = self.predict_class)
		self.button.pack()
		# cam_width, cam_height = int(cap.get(3)), int(cap.get(4))
		# 640x480 on our school laptops

	def getFrame(self):
		ret, frame = self.cap.read()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		return frame

	def updateFrame(self):
		imgtk = ImageTk.PhotoImage(image = Image.fromarray(self.getFrame()))
		self.label.configure(image = imgtk)
		self.label.image = imgtk
		self.root.after(100, self.updateFrame)

	def run(self):
		self.updateFrame()
		self.root.mainloop()
		cap.release()

	def predict_class(self):
		frame = cv2.resize(self.getFrame(), (28, 28))
		# connect to CNN here

if __name__ == "__main__":
	root = tk.Tk()
	cap = cv2.VideoCapture(0)
	win = VideoWindow(root, cap)
	win.run()
