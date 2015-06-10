#!/usr/bin/env python

import roslib
# ; roslib.load_manifest('rbx1_vision')
import rospy
import sys
import cv2
import cv2.cv as cv
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import click_and_crop
# from click_and_crop import image_cropping
import pygtk
import gtk

class cvBridgeDemo():
    def __init__(self):

        # create the main window, and attach delete_event signal to terminating
        # the application
        self.window = gtk.Window(gtk.WINDOW_TOPLEVEL)
        self.window.connect("delete_event", self.close_application)
        self.window.set_border_width(10)

        # a horizontal box to hold the buttons
        self.vbox = gtk.VBox()
        self.window.add(self.vbox)
       
        self.button = gtk.Button("screenshot")
        self.vbox.pack_start(self.button)
        self.button.connect("clicked", self.button_clicked)

        self.image=gtk.Image()
        # self.button1 = gtk.Button()
        # self.button1.add(self.image)
        # self.button1.show()
        self.vbox.pack_start(self.image)
        self.window.show_all()

        self.node_name = "cv_bridge_demo"
        
        rospy.init_node(self.node_name)
        
        # What we do during shutdown
        rospy.on_shutdown(self.cleanup)
        
        # Create the OpenCV display window for the RGB image
        # self.cv_window_name = self.node_name
        # cv.NamedWindow(self.cv_window_name, cv.CV_WINDOW_NORMAL)
        # cv.MoveWindow(self.cv_window_name, 25, 75)
        
        # Create the cv_bridge object
        self.bridge = CvBridge()
        
        # Subscribe to the camera image and depth topics and set
        # the appropriate callbacks
        self.image_sub = rospy.Subscriber("/camera/rgb/image_color", Image, self.image_callback)

        rospy.loginfo("Waiting for image topics...")    

    def image_callback(self, ros_image):
        # Use cv_bridge() to convert the ROS image to OpenCV format
        try:
            frame = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        except CvBridgeError, e:
            print e
        


        self.pixbuf=gtk.gdk.pixbuf_new_from_array(frame, gtk.gdk.COLORSPACE_RGB,8)
        # self.image=gtk.Image()
        self.image.set_from_pixbuf(self.pixbuf)

        # Convert the image to a Numpy array since most cv2 functions
        # require Numpy arrays.
        # frame = np.array(frame, dtype=np.uint8)
        

        # image_croping(frame)
        # # Process the frame using the process_image() function
        # display_image = self.process_image(frame)
                       
        # Display the image.
        # cv2.imshow(self.node_name, display_image)
        # cv2.imshow(self.node_name, frame)
        
        # Process any keyboard commands
        # self.keystroke = cv2.waitKey(5)
        # if 32 <= self.keystroke and self.keystroke < 128:
        #     cc = chr(self.keystroke).lower()
        #     if cc == 'q':
        #         # The user has press the q key, so exit
        #         rospy.signal_shutdown("User hit q key to quit.")
                
   
    def cleanup(self):
        print "Shutting down vision node."
        # cv2.destroyAllWindows()   

    # when invoked (via signal delete_event), terminates the application.
    def close_application(self, widget, event, data=None):
        gtk.main_quit()
        return False

    # is invoked when the button is clicked.  It just prints a message.
    def button_clicked(self, widget, data=None):
        print("button callback reached")
        click_and_crop.image_cropping(self.frame)
        # print "button %s clicked" % data
    

def main(args):       
    try:
        cvBridgeDemo()
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            cv2.waitKey(5)
            r.sleep()
    except KeyboardInterrupt:
        print "Shutting down vision node."
        # cv.DestroyAllWindows()
    gtk.main()
    return 0

if __name__ == '__main__':
    main(sys.argv)
    