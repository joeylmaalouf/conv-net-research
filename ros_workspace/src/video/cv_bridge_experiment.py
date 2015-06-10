#!/usr/bin/env python
import roslib
import rospy
import sys
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import click_and_crop
import pygtk
import gtk

class cvBridgeDemo():
    def __init__(self):
        self.window = gtk.Window(gtk.WINDOW_TOPLEVEL)
        self.window.connect("delete_event", gtk.main_quit)
        self.window.set_border_width(10)

        self.vbox = gtk.VBox()
        self.window.add(self.vbox)
       
        self.button = gtk.Button("screenshot")
        self.vbox.pack_start(self.button)
        self.button.connect("clicked", self.button_clicked)

        self.image=gtk.Image()
        self.vbox.pack_start(self.image)
        self.window.show_all()

        self.node_name = "cv_bridge_demo"
        rospy.init_node(self.node_name)
        rospy.on_shutdown(self.cleanup)
                
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_color", Image, self.image_callback)

        rospy.loginfo("Waiting for image topics...")    

    def image_callback(self, ros_image):
        try:
            frame = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        except CvBridgeError, e:
            print e
    
        self.pixbuf=gtk.gdk.pixbuf_new_from_array(frame, gtk.gdk.COLORSPACE_RGB,8)
        self.image.set_from_pixbuf(self.pixbuf)
   
    def cleanup(self):
        print "Shutting down vision node."

    def button_clicked(self, widget, data=None):
        print("button callback reached")
        click_and_crop.image_cropping(self.frame)
    

def main(args):
    cvBridgeDemo()
    gtk.main()

if __name__ == '__main__':
    main(sys.argv)
    