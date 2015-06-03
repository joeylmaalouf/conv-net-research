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
		self.label1 = tk.Label(self.root, image = None)
		self.label1.pack()
		self.label2 = tk.Label(self.root, image = None)
		self.label2.pack()
		self.button = tk.Button(self.root, text = "Send image to convnet as 28x28 inverted digit picture.", command = self.predict_class)
		self.button.pack()

	def get_frame(self):
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

	def update_frames(self):
		frame = self.draw_rect(self.get_frame())
		imgtk = ImageTk.PhotoImage(image = Image.fromarray(frame))
		self.label1.configure(image = imgtk)
		self.label1.image = imgtk
		frame = self.manipulate(frame)
		imgtk = ImageTk.PhotoImage(image = Image.fromarray(frame))
		self.label2.configure(image = imgtk)
		self.label2.image = imgtk
		self.root.after(100, self.update_frames)

	def manipulate(self, frame):
		x1, x2, y1, y2 = self.get_bounds(frame)
		frame = frame[y1:y2, x1:x2]
		frame = cv2.resize(frame, (28, 28))
		cv2.bitwise_not(frame, frame) # MNIST is white on black, our examples are black on white
		return frame

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

	def run(self):
		self.update_frames()
		self.root.mainloop()
		self.cap.release()

	def predict_class(self):
		frame = self.manipulate(self.get_frame())
		frame = frame.reshape(-1, 1, 28, 28)
		prediction = self.cnn.predict(frame)[0]
		print("Predicted digit: {0}".format(prediction))
		return prediction


if __name__ == "__main__":
	cv2.namedWindow("beforeclassification")
	vwin = VideoWindow(tk.Tk(), cv2.VideoCapture(0))
	vwin.run()
