#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 15:34:01 2021

@author: fibish
"""

# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import time

# from datetime import datetime
# import mysql.connector

# mydb = mysql.connector.connect(host="localhost",
# user="graymatics",
# passwd="graymatics",
# database="inteldemo"
# )

# mycursor = mydb.cursor()

# table = 'peoplecount'
# columns = 'timestamp, id'

class Track:
    def __init__(self, det, id_):
        self.attr = det
        self.id = id_
        self.dict = {}
        self.tag = set([])
        self.miss = 0

class Tracker():
    def __init__(self, maxDisappeared=50):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        '''
        A counter used to assign unique IDs to each object.
        In the case that an object leaves the frame and does not come back for maxDisappeared
        frames, a new (next) object ID would be assigned.
        '''
        self.nextObjectID = 0
        '''
        A dictionary that utilizes the object ID as the key and the centroid
        (x, y)-coordinates as the value
        '''
        self.objects = OrderedDict()

        '''
        Maintains number of consecutive frames (value) a particular object ID (key) has been marked as "lost"
        '''
        self.disappeared = OrderedDict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

    #def register(self, centroid, dimensions):
    def register(self, det):
        # when registering an object we use the next available object
        # ID to store the centroid
        # print(self.objects[self.nextObjectID])
        self.disappeared[self.nextObjectID] = 0
        # current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # current_datetime_for_image = current_datetime.split(" ")[0]+'T'+current_datetime.split(" ")[1]+'Z'
        # sql = "INSERT INTO {} ({}) VALUES (%s,%s)".format(table, columns)
        # mycursor.execute(sql, (current_datetime_for_image, nextObjectID) )
        # mydb.commit()

        self.id_base = str(int(time.time()))
        self.id_show = '{}_{}'.format(self.id_base, self.nextObjectID)
        self.objects[self.nextObjectID] = Track(det, self.id_show)
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    '''
    The function accepts a list of bounding box rectangles.
    Format of the rects parameter is assumed to be
    a tuple with this structure: (startX, startY, endX, endY)
    '''

    def update(self, rects):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            delete_this = []
            for objectID in self.disappeared.keys():
                self.disappeared[objectID] += 1
                self.objects[objectID].miss = self.disappeared[objectID]

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    delete_this.append(objectID)

            for target_id in delete_this:
                self.deregister(target_id)
            # return early as there are no centroids or tracking info
            # to update
            # return self.objects
            return list(self.objects.values())

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        inputDim = np.zeros((len(rects), 4), dtype="int")
        # loop over the bounding box rectangles
        #for (i, (startX, startY, endX, endY)) in enumerate(rects):
        for i, det in enumerate(rects):
            #startX, startY, endX, endY = det['xyxy']
            ## use the bounding box coordinates to derive the centroid
            #cX = int((int(startX) + int(endX)) / 2.0)
            ## cY = int((startY + endY) / 2.0)
            #cY = int(endY)
            #inputCentroids[i] = (cX, cY)
            #inputDim[i] = (startX, startY, endX, endY)
            inputCentroids[i] = det['xywh'][:2]
            inputDim[i] = det['xyxy']

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(rects[i])

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            objectCentroids = [item.attr['xywh'][:2] for item in objectCentroids]
            # print(objectCentroids)

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            '''
            output NumPy array shape of our distance map D  will be
            (# of object centroids, # of input centroids)
            '''
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()
            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                #self.objects[objectID] = [inputCentroids[col], inputDim[col]]
                #self.objects[objectID][0] = [inputCentroids[col]]
                #self.objects[objectID][1] = [inputDim[col]]
                self.objects[objectID].attr = rects[col]
                self.disappeared[objectID] = 0
                self.objects[objectID].miss = self.disappeared[objectID]

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    self.objects[objectID].miss = self.disappeared[objectID]

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(rects[col])

            # return the set of trackable objects
        # print("CENT_TRACK", list(self.objects.values()))
        # print("VALUES ONLY", self.objects.values())
        return list(self.objects.values())
