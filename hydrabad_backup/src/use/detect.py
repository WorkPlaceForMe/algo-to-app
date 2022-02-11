#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 15:44:12 2021

@author: fibish
"""

from use.point_in_poly import point_in_poly
  
def format0(dets, yoloClass, thres):
    ''' format yolo result into sherlock format '''
    dets2 = []
    for cls, score, (x1,y1,x2,y2) in dets:
        if yoloClass is not None and cls not in yoloClass:
            continue
        if score < thres:
            continue
        #if point_in_poly(x1,y1,[[[2, 118], [4, 325], [191, 248], [168, 66]]]):
        #    continue
        det = {}
        x = int((x1+x2)/2)
        y = int((y1+y2)/2)
        w = int(x2-x1)
        h = int(y2-y1)

        #if x < 300 and y < 260:
        #    continue
        #if w > 400:
        #    continue

        det['cls'] = cls
        det['conf'] = score
        det['xyxy'] = [int(x1),int(y1),int(x2),int(y2)]
        det['xywh'] = [x,y,w,h]
        dets2.append(det)
    return dets2

def formatPlate(dets, yoloClass, thres, screenXY):
    dets2 = []
    for det in dets:                                                                                                          
        cls, score, (x,y,w,h) = det                                                                                           
        if score < thres:                                                                                                     
            continue                                                                                                          

        x = x/416*screenXY[0]                                                                                                 
        w = w/416*screenXY[0] * 1.2
        y = y/416*screenXY[1]              
        
        h = h/416*screenXY[1] * 1.2
        x1 = int(x - w/2)                                                                                                     
        y1 = int(y - h/2)                                                                                                     
        x2 = int(x1 + w)                                                                                                      
        y2 = int(y1 + h)

        det = {}
        det['cls'] = cls
        det['conf'] = score
        det['xyxy'] = [int(x1),int(y1),int(x2),int(y2)]
        det['xywh'] = [x,y,w,h]
        dets2.append(det)
    return dets2

def formatMask(dets, yoloClass, thres, screenXY):
    dets2 = []
    for det in dets:                                                                                                          
        cls, score, (x,y,w,h) = det                                                                                           
        if score < thres:                                                                                                     
            continue                                                                                                          

        x = x/608*screenXY[0]                                                                                                 
        w = w/608*screenXY[0]                                                                                                 
        y = y/608*screenXY[1]                                                                                                 
        h = h/608*screenXY[1]                                                                                                 
        x1 = int(x - w/2)                                                                                                     
        y1 = int(y - h/2)                                                                                                     
        x2 = int(x1 + w)                                                                                                      
        y2 = int(y1 + h)
        
        det = {}
        det['cls'] = cls
        det['conf'] = score
        det['xyxy'] = [int(x1),int(y1),int(x2),int(y2)]
        det['xywh'] = [x,y,w,h]
        dets2.append(det)
    return dets2