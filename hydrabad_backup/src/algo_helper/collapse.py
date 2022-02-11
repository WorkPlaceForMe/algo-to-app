# COLLAPSE ANALYTIC
import os
import zmq
import cv2
import uuid
import numpy as np
import use.drawing as drawing
from use.tracking9 import Tracker

class Collapse:
    def __init__(self, timer, outstream, mysql_helper, algos_dict, elastic):
        # self.weaponSocket = context.socket(zmq.REQ)
        # self.weaponSocket.connect("tcp://weapon_server:5611")
        self.yoloTracker = Tracker((1280,720))
        # self.confidence = 0.3
        self.screenXY = (1280, 720)
        self.castSize = (640, 480)
        self.outstream = outstream
        mysql_fields = [
                ['time','datetime'],
                ['camera_name','varchar(40)'],
                ['cam_id','varchar(40)'],
                ['id_branch','varchar(40)'],
                ['id_account','varchar(40)']
                ]

        self.mysql_helper = mysql_helper
        self.mysql_helper.add_table('Collapse', mysql_fields)
        self.timer = timer
        self.attr = algos_dict
        self.es = elastic
        self.count_frame = 0




    def send_es(self, index, date, imgName):
        data = {}
        data['description'] = f'Collapsing person detected on {date} at {self.attr["camera_name"]}'
        date['filename'] = imgName
        data['time'] = date
        data['cam_name'] = self.attr['camera_name']
        data['algo'] = self.attr['algo_name']
        data['algo_id'] = self.attr['algo_id']
        self.es.index(index=index, body=data)

    def send_img(self, frame):
        date = self.timer.now.strftime('%Y-%-m-%-d_%H:%M:%S')
        imgName = f"/resources/{self.attr['id_account']}/{self.attr['id_branch']}/violence/{self.attr['camera_id']}/{date}.jpg"
        imgPath = '/home' + imgName
        resizedFrame = cv2.resize(frame, self.castSize)
        drawing.saveImg(imgPath, resizedFrame)
        return imgName

    def send_mysql(self, date, confidence):
        uuid_ = str(uuid.uuid4())
        mysql_values = [date, None, confidence, self.attr['camera_name'], self.attr['camera_id'],
                self.attr['id_branch'], self.attr['id_account'], uuid_, None]
        self.mysql_helper.insert_fast('Collapse', mysql_values)

    def format(dets):
        dets2 = []
        for cls, score, (x,y,w,h) in dets:
            det = {}
            x1 = int(x-w/2)
            y1 = int(y-h/2)
            x2 = int(x+w/2)
            y2 = int(y+h/2)
            
            det['cls'] = cls
            det['conf'] = score
            det['xyxy'] = [int(x1),int(y1),int(x2),int(y2)]
            det['xywh'] = [x,y,w,h]
            dets2.append(det)
        return dets2
    
    def isFall(self, w,h):
        #print(float(w)/h)
        if float(w)/h >= 1.2 and (225<w<300 and h<100):
            return True
        else: 
            return False
    
    def run(self, frame, dets, stream=True):
        self.count_frame +=1
        #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        #print(self.count_frame)
        #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        if dets:
            z=1
            for det in dets:
                #print(track.attr)
                x1,y1,x2,y2 = det['xyxy']
                x,y,w,h = det['xywh']
                if self.isFall(w,h) and det['conf']>= 0.01:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                    drawing.putTexts2(frame, ['Person Collapsed!'], x1, y1, size=1, thick=1, color=(0, 0, 255))
                z+=1

        if stream:
            self.outstream.write(frame)
        

            
            
