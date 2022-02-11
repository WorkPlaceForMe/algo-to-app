#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 15:50:26 2021

@author: fibish
"""

import numpy as np
  
# Malisiewicz et al.
#def non_max_suppression_fast(classes,scores,boxes, overlapThresh):
def check_overlap(dets, overlapThresh):
    if not dets:
        return False
    else:
        boxes = np.array([d['xyxy'] for d in dets], dtype=np.float32)

        # initialize the list of picked indexes 
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        #idxs = np.argsort(scores)

        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # compute the width and height of the bounding box (overlap)
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            # compute the ratio of overlap
            areaMin = np.minimum(area[i], area[idxs[:last]])
            overlap = (w * h) / areaMin
            if overlap > overlapThresh:
                return True
            else:
                return False