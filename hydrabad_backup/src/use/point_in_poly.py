#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 15:52:12 2021

@author: fibish
"""

"""
Determine if a point is inside a given polygon or not.

The algorithm is called the "Ray Casting Method".
Source: http://geospatialpython.com/2011/01/point-in-polygon.html
"""

def point_in_poly_single(x, y, poly):
    """                                                                                                                                                                  
    Determine if a point is inside a given polygon or not.                                                                                                               
                                                                                                                                                                         
    Polygon is a list of (x,y) pairs.                                                                                                                                    
    This function returns True or False.                                                                                                                                 
    """
    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def point_in_poly(x, y, polylist):
    """
    Determine if a point is inside a given polygon or not.

    Polygon is a list of (x,y) pairs.
    This function returns True or False.
    """
    result = False
    for poly in polylist:
        n = len(poly)
        inside = False

        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xints:
                            inside = not inside
            p1x, p1y = p2x, p2y
        if(inside is True):
            return True

    return result