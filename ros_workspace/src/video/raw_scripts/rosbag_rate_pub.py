import roslib
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
import cv2
import threading
import time
import sys
sys.path.insert(0, './pyxhook')
import pyxhook
	
class Rusbag_rate(object):
	def __init__(self):
		self.running = True
		self.time=1
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
		print "Ascii: " + str(event.Ascii) + " Scan Code: " + str(event.ScanCode) + " Key Val: " + str(event.Key)
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
		if event.Key == "s": #If the ascii value matches '4', and both ctrl and shift are pressed, run screenshot.py
			self.time+=1
			if self.time%2==0:
				print("saving")
				time.sleep(.1)
			else:
				self.msg=0
				print "cancel saving"
				time.sleep(.1)
		if event.Key == "Up":
			print "Up"
		if event.Key == "Down":
			print "Down"
		if event.Key == "Right":
			print "Right"
		if event.Key == "Left":
			print "Left"

	def keyUpEvent(self, event):
		pass
	# 	if event.ScanCode == 37: #If the scan code matches left control, signal that the ctrl button is not pressed
	# 		global ctrl
	# 		ctrl = False

	# 	if event.ScanCode == 50: #If the scan code matches left shift, signal that the shift button is not pressed
	# 		global shift
	# 		shift = False	

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
	Rusbag_rate=Rusbag_rate()
	