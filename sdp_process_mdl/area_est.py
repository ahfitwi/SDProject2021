#----------------------------------------------------------
""" 
Created on Monday April 26 18:21:23 2021

@author: alem fitwi
"""
#----------------------------------------------------------
import operator
import numpy as np

def estimate_area(centroids, widths, HUMAN_BREADTH=56): 
    """Estimates the crowded area"""
    
    censtr = [str(x) for x in centroids]    
    cwdct = {}
    for tup in list(zip(censtr,widths)):
        cwdct[tup[0]] = tup[1]

    xmin  = min(centroids, key=operator.itemgetter(0))
    ymin = min(centroids, key=operator.itemgetter(1))
    xmax  = max(centroids, key=operator.itemgetter(0))
    ymax = max(centroids, key=operator.itemgetter(1))
    
    coord1 = (xmin[0]-int(0.5*cwdct[str(xmin)]),
              xmin[1] -int(0.5*cwdct[str(xmin)]) 
              if xmin[1]<ymin[1] 
              else ymin[1]-int(0.5*cwdct[str(xmin)]))
    coord2 = (xmax[0]+int(0.5*cwdct[str(xmax)]),xmax[1]+ 
              int(0.5*cwdct[str(xmax)]) if xmax[1]>ymax[1] 
              else ymax[1]+int(0.5*cwdct[str(xmax)]))
    
    avgw = np.mean(widths)
    a_pixel = coord2[0]-coord1[0]
    a_cm = a_pixel*HUMAN_BREADTH/avgw
    b_pixel = coord2[1]-coord1[1]
    b_cm = b_pixel*HUMAN_BREADTH/avgw
    est_area = round(a_cm/100 * b_cm/100,2)
    
    return coord1, coord2, est_area

#----------------------------------------------------------
#                            END
#---------------------------------------------------------- 
