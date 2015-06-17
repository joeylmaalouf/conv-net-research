#!/bin/bash

export ROS_MASTER_URI="http://127.0.0.1:11311"
export ROS_HOSTNAME="127.0.0.1"

roslaunch bridge_opencv_pkg rosbag_play_prep.launch
