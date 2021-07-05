#-----------------------------------------------------------------------

""" 
Created on Tuesday May 25 12:11:14 2021
@author: alem fitwi
"""
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

# Project_name: Social Distancing (Min_distance and Crowd Density)

#-----------------------------------------------------------------------
1. Introudction

#----------------------------------------------------------------------------------------------------
It comprises two modules and the main program, namely sdp_config_mdl, sdp_process_mdl & SDD_main.py. 
Besides, there are three additional sub-folders, namely input_video_stream where you can place input 
videos whenever reading from a disk, output_video_stream (stores processed video streams. If working 
on local or cloud NVR, this might not be necessary), and ODM_mdl which stores the revelent 
files necessary for human object detection.

#----------------------------------------------------------------------------------------------------
For these all modules and programs to work properly, the latest version of the following python 
libraries must be first installed on the machine these programs will be run (or requirements.txt 
for a venv).
 
	import cv2: version = 4.1.2
	import imutils: version = 0.5.3
	import argparse: version =1.1
	import numpy as np: version = 1.20.3
	from scipy.spatial import distance as dist: scipy version = 1.3.1

#----------------------------------------------------------------------------------------------------
Please also make sure that the following builtin-python libraries are imported to the code!
	import operator
	import glob
	import os
	import sys
        import json
	import time
	from sys import path
	from datetime import datetime as dtt

#_____________________________________________________________________________________________________
# 2. Social Distancing Project Configuration Module (sdp_config_mdl)
#_____________________________________________________________________________________________________
It comprises the following:
      __pycache__ # folder containing the bytecode of sdconfig.py compiled and ready to be executed
      sdconfig.py	
#----------------------------------------------------------------------------------------------------
The constants defined in this configuration file are briefly described as ensues:
#----------------------------------------------------------------------------------------------------
2.1 Relative path to human detector model directory: MODEL_PATH = "ODM_mdl"

#----------------------------------------------------------------------------------------------------
2.2 Relative path to configuration file: CONFIG_PATH = "sdp_config_mdl"

#----------------------------------------------------------------------------------------------------
2.3 Relative path to Input folder: INPUT_PATH = "input_video_stream/"

#----------------------------------------------------------------------------------------------------
2.4 Name of input video (useful when reading from disk): INPUT_VID  = "example.mp4"

#----------------------------------------------------------------------------------------------------
2.5 Relative path to output folder: OUTPUT_PATH = "output_video_stream/"

#----------------------------------------------------------------------------------------------------
2.6 Video Source: VID_SRC = "disk" # "camera" (set it as "disk" or camera)

#----------------------------------------------------------------------------------------------------
2.7  Display Video?: VID_DISPLAY= "yes" # "no" (set it as "yes" or "no")

#----------------------------------------------------------------------------------------------------
2.8 Write Video?: VID_WRITE = "yes" # "no" (set it as "yes" or "no")

#----------------------------------------------------------------------------------------------------
2.9  Filter Weak Detection probability and NMS Threshold: set
	MIN_CONF = 0.3
	NMS_THRESH = 0.3

#----------------------------------------------------------------------------------------------------
2.10 CPU or GPU  (CPU --> False or GPU --> True): set USE_GPU = False if you have no GPU else True!

#----------------------------------------------------------------------------------------------------
2.11 Avg HUMAN_BREADTH & Minimum Safe Distance B/n Two People: It is next to impossible to track the
     distance between two dynamic objects using a single camera; as a result, we introduced a scheme
     based on the average HUMAN_BREADTH = 56cm. It is employed to compute the metric per pixel (MPP).
     The minimum recommended social distance between two people is 200 cm.

	HUMAN_BREADTH = 56 #cm Tricep-to-tricep avg human width
	MIN_DISTANCE = 200 #cm

#_____________________________________________________________________________________________________
# 3. Social Distance and Area Estimation Processing Module (sdp_process_mdl)
#_____________________________________________________________________________________________________
It comprises the following:
	__pycache__, # folder containing the bytecode of hdm.py compiled and ready to be executed
	hdm.py, 
	alg1.png,
	min_distance.py, #  Code for determining minimum social distance b/n two people	
	alg2.png, &
	area_est.py, #  Code for estimating rectangular area occupied by people and desnity	

#----------------------------------------------------------------------------------------------------	
3.1 Human Detection Model (hdm.py): A slightly modified YOLOv3 model is employed for the detection of 
people. It is modified such that it detects only humans unlike the original YOLOv3, which can detect 
more 20 d/t types of objects. The "ODM_mdl" subfolder in the project comprises:

	coco.names #Names of objects
	yolov3.cfg #Model
	yolov3.weights	# Weights

#----------------------------------------------------------------------------------------------------
3.2 Determination of the minimum social distance b/n two dynamic people (min_distance.py): For 
it is difficult to measure the distance between two moving people using a single camera, an algorithm 
(portrayed in alg1.png) that estimates the gap based an average HUMAN_BREADTH was indtroduced. There
should exist at least two people at any time for the minimum social distance calculation to work. If 
not, "No sufficient # of people" will be displayed on the frames!

#----------------------------------------------------------------------------------------------------
3.3 Area Estimation/approxiation program (area_est.py): For it is difficult to compute an area 
using a single camera, an algorithm (alg2.png) that estimates the gap based an average HUMAN_BREADTH 
was indtroduced. 

#_____________________________________________________________________________________________________
# 4. Social-Distancing-Detection Main-Program (SDD_main.py)
#_____________________________________________________________________________________________________
All builting libraries and user-defined modules are imported here and then all required processes are
performed. The processes listed in what ensues are performed:

		Preliminary Video Preprocessing (following a read from disk or a feed from camera)
		Detecting people or humans using YOLOv3 (coco.names, model, and weights are provided)
		Computing Minimum distance between people based average HUMAN_BREADTH
		Estimating Area crowded by people using an aproximating algorithm.
		Reporting # of violations and Crowd Density
	
NB: The first thing you should do in this program before trying to run it is setting the common
    path to the project folder (SocialDistancingProject). Set it as follows depending on where you
    place it (see the program):
            COM_PATH = "/home/alem/0_SDP/SocialDistancingProject/"	

#_____________________________________________________________________________________________________
# 5. input_video_stream subfolder
#_____________________________________________________________________________________________________
Whenever you want to process a video from a disk, you can place the video inside this folder. Yo can 
also place it wherever you want as long as the path setting is updated accordingly. 

#_____________________________________________________________________________________________________
# 6. output_video_stream subfolder
#_____________________________________________________________________________________________________
A place where processed video streams or frames are stored. You can save the videos whereever you like.
However, never forget to make the necessary path setting changes! 

	Output video streams
	Report.json

NB: Please create a json file named Report.json first before you deploy the application. Then, all 
    reports of every frame will be appended to it! The initial Report.json should contain the 
    following values:

	{
    		"frame": [
        			{"Violations": "Integer"},
        			{"Density": "Float"},
                                {"Timestamp": "Float"}
                         ]
        }

#_____________________________________________________________________________________________________
# 				END
#_____________________________________________________________________________________________________
