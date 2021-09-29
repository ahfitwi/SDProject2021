#----------------------------------------------------------
""" 
Created on Monday April 12 15:19:02 2021

@author: alem fitwi
"""
#----------------------------------------------------------
#----------------------------------------------------------
# Relative path to Object Detection Model directory
#----------------------------------------------------------
MODEL_PATH = "ODM_mdl"

#----------------------------------------------------------
# Relative path to configuration file
#---------------------------------------------------------- 
CONFIG_PATH = "sdp_config_mdl"

#----------------------------------------------------------
# Relative path to Input folder
#----------------------------------------------------------
INPUT_PATH = "input_video_stream/"
INPUT_VID  = "PETS09_S2L1.mp4" #Just a Sample 

#----------------------------------------------------------
# Relative path to output folder
#----------------------------------------------------------
OUTPUT_PATH = "output_video_stream/"
#----------------------------------------------------------
# Video Source
#----------------------------------------------------------
VID_SRC= "disk" # choose one: "camera or disk" 

#----------------------------------------------------------
# Display Video?
#----------------------------------------------------------
VID_DISPLAY= "yes" # "no" 

#----------------------------------------------------------
# Write Video?
#----------------------------------------------------------
VID_WRITE = "yes" # "no" 
#----------------------------------------------------------
# Filter Weak Detection probability and NMS Threshold
#----------------------------------------------------------
MIN_CONF = 0.3
NMS_THRESH = 0.3 # Nonmaximum supression threshold.

#----------------------------------------------------------
# CPU or GPU  (CPU --> False or GPU --> True)
#----------------------------------------------------------
USE_GPU = False

#----------------------------------------------------------
# Avg HUMAN_BREADTH & Minimum Safe Distance B/n Two People
#----------------------------------------------------------
HUMAN_BREADTH = 56 # cm, Tricep-to-tricep avg human width
MIN_DISTANCE = 200 #cm
#----------------------------------------------------------
# --------------------------END---------------------------
#----------------------------------------------------------