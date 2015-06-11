#!/usr/bin/env python

import roslib
import rospy
import sys
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import click_and_crop
import pygtk
import gtk
import gobject

class cvBridgeDemo():
    def __init__(self):
        self.window = gtk.Window(gtk.WINDOW_TOPLEVEL)
        self.window.connect("delete_event", self.cleanup)
        self.window.set_border_width(10)
        self.vbox = gtk.VBox()
        self.window.add(self.vbox)
        self.button = gtk.Button("screenshot")
        self.vbox.pack_start(self.button)
        
        self.button.connect("button_press_event", self.button_clicked)

        print "the initialization after button clicked"
        self.image=gtk.Image()
        self.vbox.pack_start(self.image)
        self.window.show_all()

        self.node_name = "cv_bridge_demo"
        rospy.init_node(self.node_name)
        rospy.on_shutdown(self.cleanup)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_color", Image, self.image_callback)
        rospy.loginfo("Waiting for image topics...")

        gobject.timeout_add(1, self.wait)   

    def image_callback(self, ros_image):
        try:
            frame = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        except CvBridgeError, e:
            print e        
        self.pixbuf=gtk.gdk.pixbuf_new_from_array(frame, gtk.gdk.COLORSPACE_RGB,8)
        self.image.set_from_pixbuf(self.pixbuf)
                

    def cleanup(self, widget=None, event=None, data=None):
        rospy.signal_shutdown("Exiting program.")
        gtk.main_quit()

    def button_clicked(self, widget, event):
        print("button callback reached")
        # click_and_crop.image_cropping(self.frame)

    def wait(self):
        # print("waiting")
        cv2.waitKey(10)
        return True

def main(args):       
    cvBridgeDemo()
    gtk.main()
    return 0

if __name__ == '__main__':
    main(sys.argv)
    