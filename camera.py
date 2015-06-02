import Tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import cv2


class Window(object):
	def __init__(self, window_root, video_capture):
		super(Window, self).__init__()
		self.root = window_root
		self.cap = video_capture
		root.bind("<Escape>", lambda e: root.quit())
		self.label = tk.Label(self.root, image = None)
		self.label.pack()
		self.button = tk.Button(self.root, text = "Send image to convnet", command = self.predict_class)
		self.button.pack()
		# cam_width, cam_height = int(cap.get(3)), int(cap.get(4))
		# 640x480 on our school laptops

	def updateFrame(self):
		ret, frame = self.cap.read()

		b,g,r = cv2.split(frame)
		frame = cv2.merge((r,g,b))

		imgtk = ImageTk.PhotoImage(image = Image.fromarray(frame))
		self.label.configure(image = imgtk)
		self.label.image = imgtk
		self.root.after(100, self.updateFrame)

	def run(self):
		self.updateFrame()
		self.root.mainloop()
		cap.release()

	def predict_class(self):
		# connect to CNN here
		pass

if __name__ == "__main__":
	root = tk.Tk()
	cap = cv2.VideoCapture(0)
	win = Window(root, cap)
	win.run()
