#----------------------------------------------------------
""" 
Created on Monday April 12 09:39:34 2021

@author: alem fitwi
"""
#----------------------------------------------------------
#----------------------------------------------------------
# Import the packages necessary for this module
#----------------------------------------------------------
import numpy as np
import cv2

#----------------------------------------------------------

# Human Object Detection Model (ODM) Function
#----------------------------------------------------------
def detect_humans(frame,net,ln,NMS_THRESH,MIN_CONF,PIdx=0):
	"""Detects only humans"""
	(H, W) = frame.shape[:2]
	results = []
	boxes, centroids, confidences = [], [], []
	blob = cv2.dnn.blobFromImage(frame, 
	      1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)
	# loop over each of the layer outputs
	for output in layerOutputs:		
		for detection in output:			
			scores = detection[5:]
			CID = np.argmax(scores)
			confidence = scores[CID]
			if CID == PIdx and confidence > MIN_CONF:				
				box = detection[0:4] *\
					np.array([W, H, W, H])
				(centerX, centerY, width, height) =\
					             box.astype("int")

				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
			
				boxes.append([x, y, int(width), 
				                         int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))

	# Apply non-maxima suppression to suppress weak, 
	#  overlapping bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, 
	                             MIN_CONF, NMS_THRESH)

	# ensure at least one detection exists 
	widths = []
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])		
			widths.append(w)	
			r = (confidences[i], (x, y, x + w, y + h), 
			                            centroids[i])
			results.append(r)
			#s.append(MIN_DISTANCE)#*h/H
	# return the list of results
	return results, widths

#----------------------------------------------------------
#                            END
#---------------------------------------------------------- 
