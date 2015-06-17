import time
import sys
sys.path.insert(0, './pyxhook')
import pyxhook

def keyDownEvent(event):
	print "Ascii: " + str(event.Ascii) + " Scan Code: " + str(event.ScanCode) + " Key Val: " + str(event.Key)
	if event.ScanCode == 37: #If the scan code matches left control, signal that the ctrl button is pressed	

	if event.ScanCode == 50: #If the scan code matches left shift, signal that the shift button is pressed
		global shift
		shift = True

	if event.Ascii == 32: #If the ascii value matches spacebar, terminate the while loop		
		global running
		running =  False
	elif event.Ascii == 52: #If the ascii value matches '4', and both ctrl and shift are pressed, run screenshot.py
		if ctrl and shift:
			print("Running Workflow #1")
			workflow1()

def keyUpEvent(event):
	if event.ScanCode == 37: #If the scan code matches left control, signal that the ctrl button is not pressed
		global ctrl
		ctrl = False

	if event.ScanCode == 50: #If the scan code matches left shift, signal that the shift button is not pressed
		global shift
		shift = False	

def workflow1(): #Workflow #1
	pass
	# subprocess.call(["python2", "./capture.py"]) #Spawn a new process that takes a screenshot
	# edit("SCREENSHOT")
	# anonymous_Upload("SCREENSHOT")

if __name__ == "__main__":
	hookman = pyxhook.HookManager()
	hookman.KeyDown = keyDownEvent #Bind keydown and keyup events
	hookman.KeyUp = keyUpEvent
	hookman.HookKeyboard()  

	hookman.start() #Start event listener
	running = True
	while running: #Stall
		time.sleep(.1)
	hookman.cancel() #Close listener