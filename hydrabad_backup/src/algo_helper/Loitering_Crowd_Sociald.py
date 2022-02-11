import os
import cv2
import zmq
import uuid
import time
import numpy as np
import ast

import use.drawing as drawing
import use.point_in_poly as point_in_poly
import use.detect as detect
import use.overlap as overlap
import use.tracking9 as tracking9
from algo_helper.old_files.sociald import SocialDistancing
from algo_helper.nms import filter_overlap
from algo_helper.old_files.mask import Mask


MYSQL_DB = os.environ.get('MYSQL_DB')
MYSQL_IP = os.environ.get('MYSQL_IP')

class Repeater:
    def __init__(self, repeat):
        self.repeat = repeat
        self.dets = []

    def update(self):
        dets = []
        for det in self.dets:
            det['count'] += 1
            if det['count'] <= self.repeat:
                dets.append(det)
        self.dets = dets

    def addNew(self, dets):
        for det in dets:
            det['count'] = 0
            self.dets.append(det)

    def run(self, dets):
        self.update()
        self.addNew(dets)
        dets_out = filter_overlap(self.dets, .5)
        return dets_out

class CrowdCounter:
    def __init__(self, timer, outstream, mysql_helper, algos_dict, elastic):
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
        self.es = elastic
        self.sociald = SocialDistancing(timer)
        self.mask = Mask()
        self.repeater = Repeater(5)

    def send_es(self, index, date, peopleCount):
        data = {}
        data['description'] = f'{peopleCount} people detected at {date} at {self.attr["camera_name"]}'
        #data['filename'] = imgName
        data['time'] = date
        data['cam_name'] = self.attr['camera_name']
        data['algo'] = self.attr['algo_name']
        data['algo_id'] = self.attr['algo_id']
        self.es.index(index=index, body=data)

    def filterDets(self, personDets):
        dets_out = []
        for det in personDets:
            x1, y1, x2, y2 = det['xyxy']
            x, y, w, h = det['xywh']

            if det['conf'] < (0.01 * self.attr['atributes'][0]['conf']):
                continue
            if (x2 -x1)/(y2-y1) > 0.8 or (y2-y1)>(0.5*720):
                continue

            if self.attr['rois'] is None or point_in_poly.point_in_poly(x, y, self.attr['rois']):
                dets_out.append(det)
        return dets_out

    def run(self, frame, personDets):
        personDets = self.filterDets(personDets)
        personDets = self.repeater.run(personDets)
        if (self.attr['rois'] is not None):
            drawing.draw_rois(frame, self.attr['rois'], (255,0,0))
        for det in personDets:
            x1, y1, x2, y2 = det['xyxy']
            x, y, w, h = det['xywh']
            if not point_in_poly.point_in_poly(x, y2, self.attr['rois']):
                continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        peopleCount = len(personDets)
        drawing.putTexts2(frame, [f"Count: {peopleCount}"], 650, 120, size=1, thick=2, color=(0,0,0))

        # social distancing
        frame = self.sociald.run(frame, personDets)

        # mask
        frame = self.mask.run(frame)

        self.outstream.write(frame)

        date = self.timer.now.strftime('%Y-%m-%d %H:%M:%S')
        mysql_values = [date, peopleCount, self.attr['camera_name'], self.attr['camera_id'], self.attr['id_branch'], self.attr['id_account']]
        if self.timer.now_t -self.last_sent > 15:
            self.mysql_helper.insert_fast('crowd_count', mysql_values)
            self.send_es('gmtc_searcher', date, peopleCount)
            self.last_sent = self.timer.now_t

class Loiter:
    def __init__(self, timer, outstream, mysql_helper, algos_dict, elastic):
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
        self.time_thres = 15
        self.es = elastic
        self.castSize = (640, 480)

    def send_es(self, index, date, duration, imgName):
        data = {}
        data['description'] = f'loitering for {duration:.0f}s from {date} at {self.attr["camera_name"]}'
        data['filename'] = imgName.replace('/resources/', '')
        data['time'] = date
        data['cam_name'] = self.attr['camera_name']
        data['algo'] = self.attr['algo_name']
        data['algo_id'] = self.attr['algo_id']
        self.es.index(index=index, body=data)

    def send_img(self, frame, id):
        date = self.timer.now.strftime('%Y-%-m-%-d_%H:%M:%S')
        imgName = f"/resources/{self.attr['id_account']}/{self.attr['id_branch']}/loitering/{self.attr['camera_id']}/{date}_{id}.jpg"
        imgPath = '/home' + imgName
        resizedFrame = cv2.resize(frame, self.castSize)
        drawing.saveImg(imgPath, resizedFrame)
        return imgName

    def get_alert_level(self, track):
        duration = track.dict['loiter']
        level = None
        if duration > self.time_thres:
            isLoiter = True
            if 'sql_loiter0' not in track.tag:
                level = 1
                track.tag.add('sql_loiter0')
            elif duration > 2 * self.time_thres:
                if 'sql_loiter1' not in track.tag:
                    level = 2
                    track.tag.add('sql_loiter1')
                elif duration > 4 * self.time_thres and 'sql_loiter2' not in track.tag:
                    level = 3
                    track.tag.add('sql_loiter2')
        else:
            isLoiter = False
        return isLoiter, level

    def run(self, frame, personTracks):
        date = self.timer.now.strftime('%Y-%m-%d %H:%M:%S')
        drawing.draw_rois(frame, self.attr['rois'], (255,0,0))
        for track in personTracks:
            x1, y1, x2, y2 = track.attr['xyxy']
            x, y, w, h = track.attr['xywh']
            if track.attr['conf'] < (0.01*self.attr['atributes'][0]['conf']):
                continue
            if 'loiter' not in track.dict:
                track.dict['loiter'] = 0
            if not point_in_poly.point_in_poly(x, y2, self.attr['rois']):
                continue
            track.dict['loiter'] += self.timer.dt
            isLoiter, level = self.get_alert_level(track)

            # draw
            if isLoiter:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                #drawing.putTexts(frame,[f'{track.id[-3:]}'], x1, y1, size=.9, thick=2, color=(0,0,255))
                drawing.putTexts(frame, ['Loitering Detected'], 30, 30, size=1, thick=2, color=(0,0,255))
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
               # drawing.putText(frame,[f'{track.id[-3:]}'], x1, y1, size=.9, thick=2, color=(0, 0, 0))
            # mysql
            if level is not None:
                uuid_ = str(uuid.uuid4())
                mysql_values = [date, track.dict['loiter'], track.id, self.attr['camera_name'], 
                        self.attr['camera_id'], self.attr['id_branch'], self.attr['id_account'], uuid_]
                self.mysql_helper.insert_fast('loitering', mysql_values)
                mysql_values = (uuid_, 'loitering', date, date, 'NULL', 
                        self.attr['id_account'], self.attr['id_branch'], level, 'NULL', 'NULL')
                self.mysql_helper.insert_fast('tickets', mysql_values)
                imgName = self.send_img(frame, track.id)
                self.send_es('gmtc_searcher', date, track.dict['loiter'], imgName)
        self.outstream.write(frame)

