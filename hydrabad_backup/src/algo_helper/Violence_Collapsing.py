import zmq
import ast
import numpy as np
import uuid
import cv2

import use.drawing as drawing
#from use.alert_trinity import AlertTrinity
#from algo_helper.clothing import Clothing
#from algo_helper.fr import FR #
from algo_helper.Loitering_Crowd_Sociald_HYD import CrowdCounter
from algo_helper.nms import filter_overlap
#from algo_helper.weapon import Weapon
from algo_helper.collapse import Collapse


# Merging Violence with Crowd,FR,Collapse,Weapon
class Merged:
    def __init__(self, timer, outstream, mysql_helper, algos_dict, elastic):
        context = zmq.Context()
        self.violenceSocket = context.socket(zmq.REQ)
        self.violenceSocket.connect("tcp://violence-server:6603")
        #self.violenceSocket.connect("tcp://localhost:6603")
        self.outstream = outstream
        self.scoreCount = 0
        self.vioScoreAvg = 0
        mysql_fields = [
            ['time','datetime'],
            ['clip_path','varchar(45)'],
            ["confidence","float"],
            ["camera_name","varchar(45)"],
            ["cam_id","varchar(45)"],
            ['id_branch',"varchar(45)"],
            ["id_account","varchar(45)"],
            ['id',"varchar(45)"],
            ['severity',"varchar(45)"]
            ]
        self.mysql_helper = mysql_helper
        self.mysql_helper.add_table('violence', mysql_fields)
        self.attr = algos_dict
        self.timer = timer
        self.es = elastic
        self.last_sent = self.timer.now_t
        self.castSize = (640, 480)
        self.pcount=0
        #self.alertTrinity = AlertTrinity()
        #self.clothing = Clothing(timer, outstream, self.mysql_helper, algos_dict, elastic)
        #self.fr = FR(timer, outstream, self.mysql_helper, algos_dict, elastic)
        self.collapse = Collapse(timer, outstream, self.mysql_helper, algos_dict, elastic)
        #self.weapon = Weapon(timer, outstream, self.mysql_helper, algos_dict, elastic)

    def send_mysql(self, date, confidence):
        uuid_ = str(uuid.uuid4())
        mysql_values = [date, None, confidence, self.attr['camera_name'], self.attr['camera_id'],
                self.attr['id_branch'], self.attr['id_account'], uuid_, 'high']
        self.mysql_helper.insert_fast('violence', mysql_values)

        mysql_values = (uuid_, 'Merged', date, date, 'NULL',
                self.attr['id_account'], self.attr['id_branch'], 0, 'NULL', 'NULL')
        self.mysql_helper.insert_fast('tickets', mysql_values)

    def send_es(self, index, date, imgName):
        data = {}
        data['description'] = f'Violence detected at {date} at {self.attr["camera_name"]}'
        data['filename'] = imgName.replace('/resources/', '')
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

    def send_alert(self, frame, confidence):
        #if self.timer.now_t - self.last_sent > 20:
        self.last_sent = self.timer.now_t
        date = self.timer.now.strftime('%Y-%m-%d %H:%M:%S')
        self.send_mysql(date, confidence)
        imgName = self.send_img(frame)
        self.send_es('gmtc_searcher', date, imgName)
        #image_name = f'/home/resources/{id_account}/{id_branch}/plate/{camid}/{date1}_{track.id}.jpg'
        #self.alertTrinity.trigger_alert(self.attr['algo_name'], 1)

    def getDistMin(self, tracks):
        tracks = [t for t in tracks]# if t.miss==0]
        if len(tracks) < 2:
            return 99999
        x,y,w,h = [],[],[],[]
        for track in tracks:
            x.append(track.attr['xywh'][0])
            y.append(track.attr['xywh'][1])
            w.append(track.attr['xywh'][2])
            h.append(track.attr['xywh'][3])

        x = np.array(x)[None,:]
        y0 = np.array(y)[None,:]
        dist = ((x-x.T)**2 + (y0-y0.T)**2)**.5

        w = np.array(w)
        w1 = np.broadcast_to(w[None,:], (len(w),len(w)))
        w2 = np.broadcast_to(w[:,None], (len(w),len(w)))
        w_min = np.amin(np.array([w1,w2]),axis=0)

        dist = dist/ w_min
        np.fill_diagonal(dist, 20000)
        distMin = np.amin(dist)
        return distMin

    def filter_for_crowd(self, yoloTracks):
        tracks2 = []
        for track in yoloTracks:
            x1,y1,x2,y2 = track.attr['xyxy']
            x,y,w,h = track.attr['xywh']
            #if w > 200:# or h <40:
             #   continue
            tracks2.append(track)
        return tracks2

    def crowd(self, frame, yoloTracks):
        dets = [t.attr for t in yoloTracks]
        dets = filter_overlap(dets, .5)
        for det in dets:
            x1,y1,x2,y2 = det['xyxy']
           # self.pcount += 1
           # cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,0), 2)
        self.pcount = len(yoloTracks)
        drawing.putTexts(frame, [f'People Count: {self.pcount}'], 30, 30, size=1, thick=2, color=(255,255,255))
        if self.pcount > 35:
            drawing.putTexts(frame, ['Over crowded'], 30, 90, size=1, thick=2, color=(0,0,255))
        #else:
            #drawing.putTexts(frame, [f'People Count: {self.pcount}'], 30, 30, size=1, thick=2, color=(255,255,255))

    def run(self, frame,yoloDets, yoloTracks):
        self.violenceSocket.send(frame)
        message = self.violenceSocket.recv()
        vioScore = ast.literal_eval(message.decode())
        #drawing.putTexts(frame, [vioScore], 130, 130, size=1, thick=1, color=(255,255,255))
        tracks = self.filter_for_crowd(yoloTracks)
        self.crowd(frame, tracks)
        if vioScore is not None:
            if self.scoreCount < 50:
                self.vioScoreAvg += vioScore
                self.scoreCount += 1
                if self.scoreCount == 50:
                    self.vioScoreAvg /= 50
            else:
                self.vioScoreAvg = (self.vioScoreAvg * 49 + vioScore) / 50
                #drawing.putTexts(frame, [self.vioScoreAvg], 30, 80, size=1, thick=1, color=(255,255,255))
                #drawing.putTexts(frame, [self.getDistMin(yoloTracks)], 30, 110, size=1, thick=1, color=(255,255,255))
                #if self.vioScoreAvg < .934 and self.vioScoreAvg > .930:
                #if self.vioScoreAvg > .934:
                #min_dist = self.getDistMin(yoloTracks)
                #drawing.putTexts(frame, [min_dist], 30, 230, size=1, thick=1, color=(255,255,255))
                #drawing.putTexts(frame, [self.vioScoreAvg], 130, 130, size=1, thick=1, color=(255,255,255))
                #drawing.putTexts(frame, [self.vioScoreAvg], 30, 130, size=1, thick=1, color=(255,255,255))
                #if self.vioScoreAvg < .934:
                if self.vioScoreAvg > .928:
                    min_dist = round(self.getDistMin(yoloTracks),2)
                    #drawing.putTexts(frame, [min_dist], 30, 230, size=1, thick=1, color=(255,255,255))
                    if min_dist < 1: # and min_dist > 0.39:

                        # if min_dist == .51 or min_dist == 0.33 or .36 <= min_dist <= .49:
                            
                        #    flag = 1
                        #else:
                        drawing.putTexts(frame, ['Violence detected!'], 50, 650, size=1.5, thick=2, color=(255,255,255))
                        self.send_alert(frame, vioScore) #TEMPORARILY COMMENTED

                #self.alertTrinity.updateVideo(frame)
        
        #self.fr.run(frame, stream=False)
        self.collapse.run(frame, yoloDets, stream=False)
        #self.weapon.run(frame, stream=False)
        #self.clothing.run2(frame, tracks)

        self.outstream.write(frame)
