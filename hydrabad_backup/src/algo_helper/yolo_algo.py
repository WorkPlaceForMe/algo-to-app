
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


MYSQL_DB = os.environ.get('MYSQL_DB')
MYSQL_IP = os.environ.get('MYSQL_IP')

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

    def send_es(self, index, date, peopleCount):
        data = {}
        data['description'] = f'{peopleCount} people detected at {date} at {self.attr["camera_name"]}'
        #data['filename'] = imgName
        data['time'] = date
        data['cam_name'] = self.attr['camera_name']
        data['algo'] = self.attr['algo_name']
        data['algo_id'] = self.attr['algo_id']
        self.es.index(index=index, body=data)
        
    def run(self, frame, personDets):
        peopleCount = 0
        if (self.attr['rois'] is not None):
            drawing.draw_rois(frame, self.attr['rois'], (255,0,0))
        for det in personDets:
            if det['conf'] < (0.01*self.attr['atributes'][0]['conf']):
                continue
            x1, y1, x2, y2 = det['xyxy']
            x, y, w, h = det['xywh']
            if (x2 -x1)/(y2-y1) > 0.8 or (y2-y1)>(0.5*720):
                continue
            if (self.attr['rois'] is not None):
                if (point_in_poly.point_in_poly(x, y, self.attr['rois'])):
                    peopleCount += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                peopleCount += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        drawing.putTexts2(frame, [f"Count: {peopleCount}"], 650, 120, size=1, thick=2, color=(0,1,0))
        self.outstream.write(frame)

        date = self.timer.now.strftime('%Y-%m-%d %H:%M:%S')
        mysql_values = [date, peopleCount, self.attr['camera_name'], self.attr['camera_id'], self.attr['id_branch'], self.attr['id_account']]
        if self.timer.now_t -self.last_sent > 15:
            self.mysql_helper.insert_fast('crowd_count', mysql_values)
            self.send_es('gmtc_searcher', date, peopleCount)
            self.last_sent = self.timer.now_t

class NoHelmet:
    def __init__(self):
        context = zmq.Context()
        self.helmSocket = context.socket(zmq.REQ)
        self.helmSocket.connect("tcp://helmet_server:5601")
        self.yoloTracker = tracking9.Tracker((1280,720))
        
    def run(self, frame, yoloDets, confidence):
        self.helmSocket.send(frame)
        message = self.helmSocket.recv()
        helmDets = ast.literal_eval(message.decode())
        helmDets = detect.format0(helmDets, None, confidence)
        yoloDets = yoloDets + helmDets
        motorbikerDets = []
        for det1 in yoloDets:
            for det2 in yoloDets:
                if det1['cls'] == 'person' and det2['cls'] == 'motorbike':
                    tempdets = [det1, det2]
                    if overlap.check_overlap(tempdets, 0.4):
                        if det1 not in motorbikerDets:
                            motorbikerDets.append(det1)
        frame = np.frombuffer(frame, dtype=np.uint8).reshape((720,1280,3))

        for det in motorbikerDets:
            if det['cls'] == 'person':
                det['cls'] = 'motorbiker'

        wearing_helmet = []
        not_wearing_helmet = []
        for det1 in motorbikerDets:
            check_helmet = False
            for det2 in helmDets:
                if det1['cls'] == 'motorbiker' and det2['cls'] == b'helmet':
                    tempdets = [det1, det2]
                    if overlap.check_overlap(tempdets, 0.01):
                        check_helmet = True
                        if tempdets[0] not in wearing_helmet:
                            wearing_helmet.append(tempdets[0])
                        if tempdets[1] not in wearing_helmet:
                            wearing_helmet.append(tempdets[1])
            if check_helmet == False:
                if det1 not in not_wearing_helmet:
                    not_wearing_helmet.append(det1)

        for det in not_wearing_helmet:
            if det['cls'] == 'motorbiker':
                det['cls'] = 'motorbiker_no_helmet'
                
        for det in wearing_helmet:
            if det['cls'] == 'motorbiker':
                det['cls'] = 'motorbiker_helmet'

        finalDets = wearing_helmet + not_wearing_helmet
        yoloTracks = self.yoloTracker.update(finalDets)
        return yoloTracks

class VehicleType:
    def __init__(self):
        pass

    def run(self, vehicleTracks, confidence):
        for track in vehicleTracks:
            if track.attr['conf'] < confidence:
                continue
            carTracks = []
            if track.attr["cls"] != "motorbike":
                carTracks.append(track)
        return carTracks

class VehicleAll:
    def __init__(self, timer, outstream, mysql_helper, algos_dict, elastic):
        self.outstream = outstream
        self.timer = timer
        self.attr = algos_dict
        self.carType = VehicleType()
        self.helmet = NoHelmet()

    def draw(self, frame, bikes, cars):
        for track in cars:
            x1, y1, x2, y2 = track.attr['xyxy']
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,0), 2)
            drawing.putTexts2(frame, [track.id, track.attr['cls']], x2, y2, size=1, thick=1, color=(0, 0, 0))

        for track in bikes:
            x1,y1,x2,y2 = track.attr['xyxy']
            if track.attr['cls'] == b'helmet':
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            elif track.attr['cls'] == 'motorbiker_helmet':
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            elif track.attr['cls'] == 'motorbiker_no_helmet':
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                drawing.putTexts(frame, ["No helmet"], x1, y1, size=1, thick=1, color=(255,255,255))

    def run(self, frame, yoloDets, vehicleTracks):
        bikes = self.helmet.run(frame, yoloDets, 0.01*self.attr['atributes'][0]['conf'])
        cars = self.carType.run(vehicleTracks, 0.01*self.attr['atributes'][0]['conf'])
        self.carbrand.run(frame, vehicleTracks)
        self.draw(frame, bikes, cars)
        self.outstream.write(frame)


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
        self.time_thres = 30
        self.es = elastic
        self.castSize = (640, 480)

    def send_es(self, index, date, duration, imgName):
        data = {}
        data['description'] = f'loitering for {duration:.0f}s from {date} at {self.attr["camera_name"]}'
        data['filename'] = imgName
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
                level = 0
                track.tag.add('sql_loiter0')
            elif duration > 2 * self.time_thres:
                if 'sql_loiter1' not in track.tag:
                    level = 1
                    track.tag.add('sql_loiter1')
                elif duration > 4 * self.time_thres and 'sql_loiter2' not in track.tag:
                    level = 2
                    track.tag.add('sql_loiter2')
        else:
            isLoiter = False
        return isLoiter, level

    def run(self, frame, personTracks):
        date = self.timer.now.strftime('%Y-%m-%d %H:%M:%S')
        for track in personTracks:
            x1, y1, x2, y2 = track.attr['xyxy']
            if track.attr['conf'] < (0.01*self.attr['atributes'][0]['conf']):
                continue
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
                imgName = self.send_img(frame, track.id)
                self.send_es('gmtc_searcher', date, track.dict['loiter'], imgName)
        self.outstream.write(frame)

