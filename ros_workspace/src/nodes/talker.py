#!/usr/bin/env python
import rospy
from std_msgs.msg import String


def talker():
	pub = rospy.Publisher("chatter", String, queue_size=10)
	rospy.init_node("talker", anonymous=True)
	rate = rospy.Rate(10) # in hz
	while not rospy.is_shutdown():
		hello_str = "Hello, World! @ {0}".format(rospy.get_time())
		# rospy.loginfo(hello_str)
		pub.publish(hello_str)
		rate.sleep()


if __name__ == "__main__":
	try:
		talker()
	except rospy.ROSInterruptException:
		pass
