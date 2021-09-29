#----------------------------------------------------------
""" 
Created on Friday April 16 10:37:06 2021

@author: alem fitwi
"""
#----------------------------------------------------------
# Libraries
#---------------------------------------------------------- 
import numpy as np
from scipy.spatial import distance as dist
#----------------------------------------------------------

# A function to Determine the minimum social distance
#---------------------------------------------------------- 
def check_min_distance(centroids, widths, MIN_DISTANCE=200,
                                         HUMAN_BREADTH=56):
    violations = set()
    width_pixel = np.empty(shape=(len(centroids),
                      len(centroids)),dtype='float')
    for i in range(len(widths)):
        for j in range(len(widths)):
            width_pixel[i][j]=(
                  widths[i]+widths[j])/2	           
    distance_pixel = dist.cdist(centroids,centroids, 
                                metric="euclidean")     
    for i in range(0, distance_pixel.shape[0]):
        for j in range(i + 1, distance_pixel.shape[1]):
            PPM = HUMAN_BREADTH/width_pixel[i,j]
            if distance_pixel[i, j]*PPM<MIN_DISTANCE:             
                violations.add(i)
                violations.add(j)
    return violations

#----------------------------------------------------------
#                            END
#---------------------------------------------------------- 
