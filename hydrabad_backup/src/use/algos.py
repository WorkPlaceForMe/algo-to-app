#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 16:00:16 2021

@author: fibish
"""

import os
import cv2
import ast
import numpy as np
import zmq
import uuid
from collections import deque, Counter

import drawing
import name
import detect
from cent_tracking import Tracker
import ocr.server1
import ocr.server2
from tracking9 import Tracker as Tracker2

MYSQL_DB = os.environ.get('MYSQL_DB')
MYSQL_IP = os.environ.get('MYSQL_IP')

class CrowdCounter:
    def __init__(self, timer, outstream, mysql_helper, algos_dict):
        self.outstream = outstream
        mysql_fields = [['time','datetime'],
           ['number_of_ppl','int(11)'],
           ['camera_name','varchar(40)'],
           ['cam_id','varchar(40)'],
           ['id_branch','varchar(40)'],
           ['id_account','varchar(40)']]
        self.mysql_helper = mysql_helper
        self.mysql_helper.add_table('crowd_count', mysql_fields)
        self.timer = timer
        self.attr = algos_dict
        self.last_sent = self.timer.now_t

    def run(self, frame, personDets):
        peopleCount = 0
        for det in personDets:
            x1, y1, x2, y2 = det['xyxy']
            if (x2 -x1) > 100:
                continue
            peopleCount += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        drawing.putTexts2(frame, [f"Count: {peopleCount}"], 650, 120, size=1, thick=2, color=(0,0,0))
        self.outstream.write(frame)

        date = self.timer.now.strftime('%Y-%m-%d %H:%M:%S')
        
        mysql_values = [date, peopleCount, self.attr['camera_name'], self.attr['camera_id'], self.attr['id_branch'], self.attr['id_account']]
        if self.timer.now_t -self.last_sent > 15:
            self.mysql_helper.insert_fast('crowd_count', mysql_values)
            self.last_sent = self.timer.now_t

class VehicleType:
    def __init__(self, timer, outstream, mysql_helper, algos_dict):
        self.outstream = outstream
        mysql_fields = [
            ['track_id','varchar(45)'],
            ['time','datetime'],
            ['class','varchar(45)'],
            ['camera_name','varchar(45)'],
            ['cam_id','varchar(45)'],
            ['id_account','varchar(45)'],
            ['id_branch','varchar(45)']
            ]
        self.mysql_helper = mysql_helper
        self.mysql_helper.add_table('vehicle', mysql_fields)
        self.timer = timer
        self.attr = algos_dict

    def run(self, frame, vehicleTracks):
        date = self.timer.now.strftime('%Y-%m-%d %H:%M:%S')
        for track in vehicleTracks:
            x1, y1, x2, y2 = track.attr['xyxy']
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,0), 2)
            drawing.putTexts2(frame, [track.id, f'{track.attr["cls"]}'], x2, y2, size=1, thick=1, color=(0, 0, 0))

            if 'mysql_vehicle' not in track.tag:
                mysql_values = [track.id, date, track.attr['cls'], self.attr['camera_name'], self.attr['camera_id'], self.attr['id_account'], self.attr['id_branch']]
                track.tag.add('mysql_vehicle')
                self.mysql_helper.insert_fast('vehicle', mysql_values)

        self.outstream.write(frame)

class Loiter:
    def __init__(self, timer, outstream, mysql_helper, algos_dict):
        self.outstream = outstream
        self.timer = timer
        mysql_fields = [
            ['time','datetime'],
            ['dwell','int(11)'],
            ["track_id","varchar(40)"],
            ["camera_name","varchar(40)"],
            ["cam_id","varchar(40)"],
            ["id_branch","varchar(40)"],
            ["id_account","varchar(40)"],
            ['id',"varchar(45)"]
            ]
        self.mysql_helper = mysql_helper
        self.mysql_helper.add_table('loitering', mysql_fields)
        self.attr = algos_dict
        self.time_thres = 30

    def get_alert_level(self, track):
        duration = track.dict['loiter']
        level = None
        if duration > self.time_thres:
            isLoiter = True
            if 'sql_loiter0' not in track.tag:
                level = 0
                track.tag.add('mysql_loiter')
            elif duration > 2 * self.time_thres:
                if 'sql_loiter1' not in track.tag:
                    level = 1
                elif duration > 4 * self.time_thres and 'sql_loiter2' not in track.tag:
                    level = 2
        else:
            isLoiter = False
        return isLoiter, level

    def run(self, frame, personTracks):
        date = self.timer.now.strftime('%Y-%m-%d %H:%M:%S')
        for track in personTracks:
            x1, y1, x2, y2 = track.attr['xyxy']
            if 'loiter' not in track.dict:
                track.dict['loiter'] = 0
            track.dict['loiter'] += self.timer.dt
            isLoiter, level = self.get_alert_level(track)

            # draw
            if isLoiter:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

            # mysql
            if level is not None:
                uuid_ = str(uuid.uuid4())
                mysql_values = [date, track.dict['loiter'], track.id, self.attr['camera_name'],
                        self.attr['camera_id'], self.attr['id_branch'], self.attr['id_account'], uuid_]
                self.mysql_helper.insert_fast('loitering', mysql_values)
                mysql_values = (uuid_, 'loitering', date, date, 'NULL',
                        self.attr['id_account'], self.attr['id_branch'], level, 'NULL', 'NULL')
                self.mysql_helper.insert_fast('tickets', mysql_values)
        self.outstream.write(frame)

class FR:
    def __init__(self, timer, outstream, mysql_helper, algos_dict):
        context = zmq.Context()
        self.faceSocket = context.socket(zmq.REQ)
        self.faceSocket.connect("tcp://fr-server:5603")
        self.knownFace = name.KnownData('/home/face_data/', "tcp://fr-server:5605")
        self.face_thres = 0.1
        self.knownface_thres = 0.4
        self.outstream = outstream
        self.tracker = Tracker2((1280,720))
        self.average_len = 20
        mysql_fields = [
            ['id','varchar(45)'],
            ['time','datetime'],
            ['name','varchar(45)'],
            ['gender','varchar(20)'],
            ['age','varchar(20)'],
            ['emotion','varchar(20)'],
            ['cam_id','varchar(45)'],
            ['cam_name','varchar(45)'],
            ['id_account','varchar(45)'],
            ['id_branch','varchar(45)']
            ]
        self.mysql_helper = mysql_helper
        self.mysql_helper.add_table('faces', mysql_fields)
        self.timer = timer
        self.attr = algos_dict

    def format(self, dets):
        dets2 = []
        for score, xywh, xyxy, facepoint, age, gender, emotion, feature in dets:
            det = {}
            if score < 0.5:
                continue
            det['score'] = score
            det['xyxy'] = xyxy
            det['xywh'] = xywh
            det['facepoint'] = facepoint
            det['feature'] = feature

            x, y, w, h = xywh
            nx, ny = facepoint[2]
            r1 = abs(nx-x)/w
            r2 = abs(ny-y)/h
            det['tilt'] = '{:.2f},{:.2f},{}'.format(r1, r2, w)

            if r1<.3 and r2<.3 and w > 20:
                det['front'] = True
            else:
                det['front'] = False

            if det['front']:
                dets2.append(det)
        return dets2

    def get_average(self, track, name):
        if 'name' not in track.dict:
            track.dict['name'] = deque([], maxlen=self.average_len)
        if name != 'unknown':
            track.dict['name'].append(name)

        if len(track.dict['name']) >= self.average_len:
            return Counter(track.dict['name']).most_common(1)[0][0]
        else:
            return None

    def save_alert(self, track, date, avg_name):
        if 'mysql_face' not in track.tag:
            track.tag.add('mysql_face')
            mysql_values = (track.id, date, avg_name, "NULL", "NULL", "NULL",
                self.attr['camera_id'], self.attr['camera_name'], self.attr['id_account'], self.attr['id_branch'])
            self.mysql_helper.insert_fast('faces', mysql_values)

    def run(self, frame):
        self.faceSocket.send(frame)
        message = self.faceSocket.recv()
        faceDets = ast.literal_eval(message.decode())
        faceDets = self.format(faceDets)
        tracks = self.tracker.update(faceDets)
        date = self.timer.now.strftime('%Y-%m-%d %H:%M:%S')
        for track in tracks:
            x1,y1,x2,y2 = track.attr['xyxy']
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
            known_name = self.knownFace.getName(track.attr['feature'], thres=self.knownface_thres)
            avg_name = self.get_average(track, known_name)
            if avg_name is not None:
                drawing.putTexts(frame, [avg_name], x1, y1, size=1, thick=1, color=(255,255,255))
                self.save_alert(track, date, avg_name)
        self.outstream.write(frame)
        
class Clothing:
    def __init__(self, timer, outstream, mysql_helper, algos_dict):
        context = zmq.Context()
        self.pubSocket = context.socket(zmq.REQ)
        self.pubSocket.connect(f'tcp://test_clothing:5612')
        self.outstream = outstream
        self.castSize = (640, 480)
        mysql_fields = [
            ["track_id", "varchar(45)"],
            ["time","datetime"],
            ["cam_id","varchar(45)"],
            ["cam_name","varchar(45)"],
            ["id_account","varchar(45)"],
            ["id_branch", "varchar(45)"],
            ["sleeve_length", "varchar(45)"],
            ["top_colour", "varchar(45)"],
            ["bottom_length", "varchar(45)"],
            ["bottom_colour", "varchar(45)"],
            ["shoe_colour", "varchar(45)"]
            ]
        self.mysql_helper = mysql_helper
        self.mysql_helper.add_table('clothing', mysql_fields)
        self.timer = timer
        self.attr = algos_dict
        self.average_len = 20

    def get_average(self, track, labels):
        if 'clothes' not in track.dict:
            track.dict['clothes'] = [deque(['Unknown'], maxlen=self.average_len) for i in range(5)]

        isFull = False
        for i, label in enumerate(labels):
            if name != 'Unknown':
                track.dict['clothes'][i].append(label)
                if len(track.dict['clothes'][i]) >= self.average_len:
                    isFull = True
        if isFull:
            avg_labels = []
            for q in track.dict['clothes']:
                avg_labels.append(Counter(q).most_common(1)[0][0])
            return avg_labels
        else:
            return None
        
    def save_alert(self, track, date, labels):
        if 'mysql_clothes' not in track.tag:
            track.tag.add('mysql_clothes')
            mysql_values = (track.id, date, self.attr['camera_id'], self.attr['camera_name'],
                    self.attr['id_account'], self.attr['id_branch'], labels[0], labels[1], labels[2], labels[3], labels[4])
            self.mysql_helper.insert_fast('clothing', mysql_values)

    def run(self, frame, yoloTracks):
        date = self.timer.now.strftime('%Y-%m-%d %H:%M:%S')
        for track in yoloTracks:
            if track.miss > 0:
                continue
            x1,y1,x2,y2 = track.attr['xyxy']
            img = frame[y1:y2,x1:x2]
            img = np.ascontiguousarray(img)
            self.pubSocket.send_multipart([str(img.shape).encode(), img])
            message = self.pubSocket.recv()
            labels = ast.literal_eval(message.decode())

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,0), 2)
            avg_labels = self.get_average(track, labels)
            if avg_labels is not None:
                upperwear = f'{avg_labels[0]} {avg_labels[1]} top'.replace('Unknown ','').lower()
                lowerwear = f'{avg_labels[2]} {avg_labels[3]} btm'.replace('Unknown ','').lower()
                shoe = f'{avg_labels[4]} shoes'.replace('Unknown ','').lower()
                drawing.putTexts2(frame, [upperwear, lowerwear, shoe], x1, y1, size=.8, thick=1, color=(0,0,0))
                self.save_alert(track, date, avg_labels)
        self.outstream.write(frame)

class ANPR:
    def __init__(self, outstream):
        context = zmq.Context()
        self.plateSocket = context.socket(zmq.REQ)
        self.plateSocket.connect("tcp://anpr-server:5611")
        self.outstream = outstream
        self.castSize = (640, 480)
        self.confidence = 0.1
        self.screenXY = (1280, 720)
        self.yoloTracker = Tracker()
        self.average_count = 10
        
    def run(self, frame):
        frame0 = frame.copy()
        frame_top = frame.copy()
        self.plateSocket.send(frame)
        message = self.plateSocket.recv()
        plateDets = ast.literal_eval(message.decode())
        plateDets = detect.formatPlate(plateDets, None, self.confidence, self.screenXY)
        yoloTracks = self.yoloTracker.update(plateDets)

        for i, track in enumerate(yoloTracks):
            x1,y1,x2,y2 = track.attr['xyxy']
            x,y,w,h = track.attr['xywh']
            lp_number = None
            if track.miss == 0:
                pass
                x_det = x2 - x1
                y_det = y2 - y1
                plate_ratio = x_det / y_det
                # Second Row, if exists
                if plate_ratio < 2.5:
                    y1, y2 = int(y1 + h*.35), int(y2 + h*.05) #h*.525 #.48 .05
                    mask_y1 = int(y1 + h*.48)
                    mask_y2 = int(y1 + h*.1)
                    mask_left = int(x1 + w*.14)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                    frame0[mask_y1:y2, x1:x2] = 0
                    frame0[y1:mask_y2, x1:x2] = 0
                    frame0[y1:y2, x1:mask_left] = 0
                    
                    # if
                    second_row = frame0[y1:y2,x1:x2]
                    try:
                        img, plate = ocr.server1.main(second_row)
                    except:
                        break
                    try:
                        print(plate)
                    except:
                        pass
                    if 'plate' not in track.dict:
                        track.dict['plate'] = deque([''], maxlen=30)
                        track.dict['count_plate'] = 1
                        if len(plate) > 3 and len(plate) < 6:
                            track.dict['plate'].append(plate)
                    elif track.dict['count_plate'] < self.average_count:
                        if len(plate) > 3 and len(plate) < 6:
                            #plate = plate[:9]
                            #plate = plate.replace('Y','U')
                            track.dict['plate'].append(plate)
                            track.dict['count_plate'] += 1
                    #texts = [plate]
                    if track.dict['count_plate'] >= self.average_count:
                        true_plate = Counter(track.dict['plate']).most_common(1)[0][0]
                        drawing.putTexts(frame, [true_plate], x1, y1, size=1, thick=1, color=(255,255,255))
                        # track.dict['count_plate'] = average_count - 5
                # print(plate)
                #........
                # First Row
                x1,y1,x2,y2 = track.attr['xyxy']
                if plate_ratio < 2.5:
                    y1, y2 = int(y1 - h*.05), int(y1 + h*.47) #.53
                    mask_y1 = int(y1 + h*.48)
                    mask_y2 = int(y1 + h*.1)
                    mask_right = int(x2 - w*.15)
                    frame_top[mask_y1:y2, x1:x2] = 0
                    frame_top[y1:mask_y2, x1:x2] = 0
                    frame_top[y1:y2, mask_right:x2] = 0
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                try:
                    img2, plate2 = ocr.server2.main(frame_top[y1:y2,x1:x2])
                except:
                    break
                try:
                   print(plate2)
                except:
                   pass
                if 'plate2' not in track.dict:
                    track.dict['plate2'] = deque([''], maxlen=30)
                    track.dict['plate2'].append(plate2)
                    track.dict['count_plate2'] = 1
                elif track.dict['count_plate2'] < self.average_count:
                    if len(plate2) > 2:
                        track.dict['plate2'].append(plate2)
                        track.dict['count_plate2'] += 1
                #if len(plate2) >= 8:
                #    plate2 = plate2[:9]
                #    #plate = plate.replace('Y','U')
                if track.dict['count_plate2'] >= self.average_count:
                    true_plate2 = Counter(track.dict['plate2']).most_common(1)[0][0]
                    drawing.putTexts(frame, [true_plate2], x1, y1, size=1, thick=1, color=(255,255,255))
                try:
                    # print(counter)
                    if (len(true_plate2) > 2):
                        if plate_ratio < 2.5 and (len(true_plate) > 3):
                            lp_number = true_plate2 + true_plate
                        else:
                            lp_number = true_plate2
                        print("REAL NUMBER", lp_number)
                except:
                    pass

                if lp_number is not None and 'plate' not in track.tag:
                #    image_name = f'/home/resources/{id_account}/{id_branch}/plate/{camid}/{date1}_{track.id}.jpg'
                    track.tag.add('plate')
            else:
                pass

        self.outstream.write(frame)
