#----------------------------------------------------------
""" 
Created on Monday April 12 14:13:56 2021

@author: alem fitwi
"""
#----------------------------------------------------------

# Import Necessary Libraries and Packages
#----------------------------------------------------------
import cv2
import glob
import json
import time
import imutils
import os, sys
import argparse
import numpy as np
from sys import path
from datetime import datetime as dtt
#----------------------------------------------------------

# Path Settings and Modules Importing --> Linux OS
#----------------------------------------------------------
# Set common path to project location and paths to modules
COM_PATH = "/home/alem/0_SDP/SocialDistancingProject/"
path.append(os.path.join(COM_PATH,'sdp_config_mdl/')) 
path.append(os.path.join(COM_PATH,'sdp_process_mdl/')) 
#----------------------------------------------------------

# Import User-defined Modules
from sdp_config_mdl.sdconfig import INPUT_PATH, VID_SRC
from sdp_config_mdl.sdconfig import OUTPUT_PATH, MIN_CONF
from sdp_config_mdl.sdconfig import NMS_THRESH, MIN_DISTANCE
from sdp_config_mdl.sdconfig import HUMAN_BREADTH, INPUT_VID
from sdp_config_mdl.sdconfig import VID_DISPLAY,MODEL_PATH
from sdp_config_mdl.sdconfig import CONFIG_PATH, USE_GPU
from sdp_config_mdl.sdconfig import VID_WRITE
from sdp_process_mdl.hdm import detect_humans
from sdp_process_mdl.min_distance import check_min_distance
from sdp_process_mdl.area_est import estimate_area
#----------------------------------------------------------

# Main
if __name__ == '__main__':
    dtdate = dtt.now().strftime("%Y_%B_%d_%H_%M_%S")
    labelsPath = os.path.sep.join(
	                  [MODEL_PATH, "coco.names"])
    labelsPath = os.path.join(COM_PATH, labelsPath)
    #------------------------------------------------------

    weightsPath = os.path.sep.join(
	             [MODEL_PATH, "yolov3.weights"])
    weightsPath = os.path.join(COM_PATH, weightsPath)
    #------------------------------------------------------

    configPath = os.path.sep.join(
	             [MODEL_PATH, "yolov3.cfg"])
    configPath = os.path.join(COM_PATH, configPath)
    #-----------------------------------------------------

    output_path = os.path.join(COM_PATH, OUTPUT_PATH)
    #----------------------------------------------------

    # Load the COCO labels based on which YOLOv3 was trained
    #-------------------------------------------------------
    LABELS = open(labelsPath).read().strip().split("\n")

    #--------------------------------------------------------

    # Load YOLOv3 Model Trained on COCO dataset (80 classes)
    #--------------------------------------------------------
    print("[INFO] Loading YOLOv3 Human Detection Model from"+
                " Disk ...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    #-------------------------------------------------------
    # Choose CPU or GPU?
    #-------------------------------------------------------
    if USE_GPU:
        print("[INFO] Setting Backend & Target to CUDA ...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    #-------------------------------------------------------
    # Select only the required *output* layer names 
    #------------------------------------------------------- 
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in 
                            net.getUnconnectedOutLayers()]

     #-------------------------------------------------------

    # Report Violations and Density (saved in JSON format)
    #------------------------------------------------------- 
    def write_report_json(new_data, filename):
        with open(filename,'r+') as file:
            file_data = json.load(file)
            file_data.update(new_data)
            file.seek(0)
            json.dump(file_data, file, indent = 4)

    #-------------------------------------------------------

    # Initialize the video stream and pointer to O/P video 
    #------------------------------------------------------- 
    print("[INFO] Reading/Accessing video stream...")
    if VID_SRC == 'disk':
        vs_fn = os.path.sep.join([INPUT_PATH, INPUT_VID])
        vs_path = os.path.join(COM_PATH, vs_fn)
        vs = cv2.VideoCapture(vs_path)
    else:
        vs = cv2.VideoCapture(0)    
    
    rpt_dct, count, writer = {}, 1, None
    filenamejson = output_path + "Report.json"

    while True:	
        (grabbed, frame) = vs.read()        
        if not grabbed:
            break
        #--------------------------------------------------

        frame = imutils.resize(frame, width=720)
        results, widths = detect_humans(frame, net, ln, 
        NMS_THRESH, MIN_CONF, PIdx=LABELS.index("person"))         
        #--------------------------------------------------

        violations = set() #Initialize distance violation  
        x1, x2, x3 = 0, 0, 0      
        #--------------------------------------------------

        if len(results) >= 2: #At least 2 people
            centroids = np.array([r[2] for r in results])            
            # loop over the upper triangle of distance matrix
            violations =check_min_distance(centroids, widths, 
                          MIN_DISTANCE=200, HUMAN_BREADTH=56)
            coord1, coord2, est_area = estimate_area(centroids, 
                              widths, HUMAN_BREADTH=56)
            density = round(len(centroids)/est_area,2)
            cv2.rectangle(frame,coord1,coord2, (242, 89, 124), 2)
            #-----------------------------------------------------

            new_data = {'frame'+str(count):[
                        {'Violations':len(violations)},
                        {'Density':density},
                        {'Timestamp': time.time()}]}
            
            write_report_json(new_data, filenamejson)            
            #-----------------------------------------------------
	
            # Loop over the results              
            for (i, (prob, bbox, centroid)) in enumerate(results):	
                (startX, startY, endX, endY) = bbox
                (cX, cY) = centroid
                color = (0, 255, 0)
                if i in violations:
                    color = (0, 0, 255)
                # Draw a bounding box around the person 
                cv2.rectangle(frame, (startX, startY), 
                    (endX, endY), (255, 89, 234), 2)
                color1 = (255, 0, 255)		    
                cv2.circle(frame, (cX, cY), 5, color, -1)   
                cv2.putText(frame, str(), (15, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)            
            #-----------------------------------------------------

            # Draw the total # of social distancing violations 
            nv = "# of MSD Violations: {}".format(len(violations))
            pd = "Density: {}:{} = {}".format(
                        len(centroids), est_area, density)
            cv2.putText(frame, nv, (15, frame.shape[0] - 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
            cv2.putText(frame, pd, (15, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (234, 16, 255), 2)
        else:
            no = "No sufficient # of people!"
            cv2.putText(frame, no, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.85, (0, 0, 255), 2)
        count += 1
        #-----------------------------------------------------

        # Option to display frames
        if VID_DISPLAY == 'yes':
            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            # Press 'q' to stop the stream
            if key == ord("q") or grabbed == False:
                break
        #-----------------------------------------------------

        if VID_WRITE == "yes" and writer is None:            
            vname = "video"+dtdate+".mp4"
            fname = os.path.sep.join([OUTPUT_PATH, vname])
            write_path = os.path.join(COM_PATH, fname)
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(write_path, fourcc, 25,
                (frame.shape[1], frame.shape[0]), True)

        # Write the frame to the output video file
        if writer is not None:
            writer.write(frame)      

#----------------------------------------------------------
# 		     ~END~
#---------------------------------------------------------- 
