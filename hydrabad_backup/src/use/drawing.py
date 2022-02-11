#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 15:46:37 2021

@author: fibish
"""

import os
import cv2
from pathlib import Path

def saveImg(path, img):
    dir0 = os.path.dirname(path)
    Path(dir0).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, img)

def draw_roi(frame, pts, color):
    for i in range(len(pts)-1):
        cv2.line(frame, tuple(pts[i]), tuple(pts[i+1]), color,2)
    cv2.line(frame, tuple(pts[-1]), tuple(pts[0]), color,2)

def draw_rois(frame, rois, color):
    for roi in rois:
        draw_roi(frame, roi, color)

def get_distance(p, q):
    return ((p[0]-q[0])**2 + (p[1]-q[1])**2)**.5

def putTexts(img, texts, x1, y1, size=1, thick=1, color=(255,255,255)):
    for text1 in texts[::-1]:
        text1 = str(text1)
        (text_x, text_y) = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, fontScale=size, thickness=thick)[0]
        x1, y1 = int(x1), int(y1)
        cv2.rectangle(img, (x1,y1-text_y), (x1+text_x, y1), (0, 0, 0), -1)
        cv2.putText(img, text1, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)
        y1 -= text_y

def putTexts2(img, texts, x1, y1, size=1.5, thick=1, color=(0,0,0)):
    for text1 in texts[::-1]:
        text1 = str(text1)
        (text_x, text_y) = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, fontScale=size, thickness=thick)[0]
        x1, y1 = int(x1), int(y1)
        cv2.rectangle(img, (x1,y1-text_y), (x1+text_x, y1), (255, 255, 255), -1)
        cv2.putText(img, text1, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)
        y1 -= text_y * 1.2
