import Tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import cv2
import convnet


class VideoWindow(object):
	def __init__(self, window_root, video_capture):
		super(VideoWindow, self).__init__()

		self.cnn = convnet.ConvolutionalNeuralNetwork()
		self.cnn.load_all_weights()
		self.cnn.create_model_functions()

		self.root = window_root
		self.cap = video_capture
		self.root.bind("<Escape>", lambda e: self.root.quit())
		self.label = tk.Label(self.root, image = None)
		self.label.pack()
		self.button = tk.Button(self.root, text = "Send image to convnet as 28x28 digit picture.", command = self.predict_class)
		self.button.pack()

	def getFrame(self):
		ret, frame = self.cap.read()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		w = self.cap.get(3)
		h = self.cap.get(4)
		if w > h:
			offset = (w-h)/2
			frame = frame[:, offset:w-offset]
		elif w < h:
			offset = (h-w)/2
			frame = frame[offset:h-offset, :]

		return frame

	def updateFrame(self):
		imgtk = ImageTk.PhotoImage(image = Image.fromarray(self.getFrame()))
		self.label.configure(image = imgtk)
		self.label.image = imgtk
		self.root.after(100, self.updateFrame)

	def run(self):
		self.updateFrame()
		self.root.mainloop()
		self.cap.release()

	def predict_class(self):
		frame = cv2.resize(self.getFrame(), (28, 28)).reshape(-1, 1, 28, 28)
		cv2.bitwise_not(frame, frame) # MNIST is white on black, our examples are black on white
		prediction = self.cnn.predict(frame)[0]
		print("Predicted digit: {0}".format(prediction))
		return prediction


if __name__ == "__main__":
	vwin = VideoWindow(tk.Tk(), cv2.VideoCapture(0))
	vwin.run()
