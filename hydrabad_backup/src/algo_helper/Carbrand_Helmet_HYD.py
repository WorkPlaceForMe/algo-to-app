import os
import cv2
import zmq
import uuid
import time
import numpy as np
import ast
from collections import Counter

import use.drawing as drawing
from use.point_in_poly import point_in_poly
import use.detect as detect
import use.overlap as overlap
import use.tracking9 as tracking9
from algo_helper.carbrand import CarBrand
from algo_helper.nms import filter_overlap

MYSQL_DB = os.environ.get('MYSQL_DB')
MYSQL_IP = os.environ.get('MYSQL_IP')

class NoHelmet:
    def __init__(self):
        context = zmq.Context()
        self.helmSocket = context.socket(zmq.REQ)
        self.helmSocket.connect("tcp://helmet_server:5601")
        self.yoloTracker = tracking9.Tracker((1280,720), dist_thres=300)
        self.helmet_width_thres = 90
        self.stable_size = 1

    def run(self, frame, yoloDets, confidence):
        self.helmSocket.send(frame)
        message = self.helmSocket.recv()
        helmDets = ast.literal_eval(message.decode())
        helmDets = detect.format0(helmDets, None, confidence)
        yoloDets = yoloDets + helmDets
        motorbikerDets = []

        yoloDets = filter_overlap(yoloDets, 0.5)

        for det1 in yoloDets:
            if det1['xywh'][2] > 600:
                continue
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

        #for det in not_wearing_helmet:
        #    if det['cls'] == 'motorbiker':
        #        det['cls'] = 'motorbiker_no_helmet'

        #for det in wearing_helmet:
        #    if det['cls'] == 'motorbiker':
        #        det['cls'] = 'motorbiker_helmet'

        #finalDets = wearing_helmet + not_wearing_helmet
        finalDets = self.finalise(not_wearing_helmet, wearing_helmet)
        yoloTracks = self.yoloTracker.update(finalDets)
        self.stable_res(yoloTracks)
        return yoloTracks

    def stable_res(self, tracks):
        for track in tracks:
            if 'helmet' in track.dict:
                continue
            if 'helmet_list' not in track.dict:
                track.dict['helmet_list'] = []
            if track.attr['helmet'] != 'unknown':
                track.dict['helmet_list'].append(track.attr['helmet'])
                if len(track.dict['helmet_list']) >= self.stable_size:
                    track.dict['helmet'] = Counter(track.dict['helmet_list']).most_common(1)[0]

    def finalise(self, not_wearing_helmet, wearing_helmet):
        dets = []
        for det in not_wearing_helmet:
            if det['cls'] == 'motorbiker':
                if det['xywh'][2] < self.helmet_width_thres:
                    det['helmet'] = 'unknown'
                else:
                    det['helmet'] = False
                dets.append(det)

        for det in wearing_helmet:
            if det['cls'] == 'motorbiker':
                if det['xywh'][2] < self.helmet_width_thres/2:
                    det['helmet'] = 'unknown'
                else:
                    det['helmet'] = True
                dets.append(det)
        return dets



class VehicleAll:
    def __init__(self, timer, outstream, mysql_helper, algos_dict, elastic):
        self.outstream = outstream
        self.timer = timer
        self.attr = algos_dict
        self.helmet = NoHelmet()
        self.carbrand = CarBrand()
        self.es = elastic
        self.roi = [[[380,320], [800,320], [980,630], [320,630]]]
        # ADDITIONS
        mysql_fields = [
                ["track_id","varchar(45)"],
                ["time","datetime"],
                ["cam_id","varchar(45)"],
                ["cam_name","varchar(45)"],
                ["id_account","varchar(45)"],
                ["id_branch","varchar(45)"],
                ["vehicle_type","varchar(45)"],
                ["alert","varchar(45)"],
                ["car_numbers","int"],
                ["motorbike_numbers","int"],
                ["truck_numbers","int"],
                ["bus_numbers","int"]
                ]
        self.mysql_helper = mysql_helper
        self.mysql_helper.add_table('vcount', mysql_fields)
        self.last_sent = time.time()

    def save_vcount(self, yoloDets):
        if time.time() - self.last_sent > 10:
            self.last_sent = time.time()
            date = self.timer.now.strftime('%Y-%-m-%-d_%H:%M:%S')
            mysql_values = [0, date, self.attr['camera_id'], self.attr['camera_name'], self.attr['id_account'], self.attr['id_branch'], 0, 0,
                len([d for d in yoloDets if d['cls']=='car']),
                len([d for d in yoloDets if d['cls']=='motorbike']),
                len([d for d in yoloDets if d['cls']=='truck']),
                len([d for d in yoloDets if d['cls']=='bus'])]

            #print('insert to vcount')
            self.mysql_helper.insert_fast('vcount', mysql_values)


    def get_car(self, vehicleTracks, confidence):
        carTracks = []
        for track in vehicleTracks:
            if track.attr['conf'] < confidence:
                continue
            if track.attr["cls"] != "motorbike":
                carTracks.append(track)
        return carTracks


    def send_es(self, index, date, labels, imgName):
        data = {}
        data['description'] = f'at {date}, {",".join(labels)} detected'
        data['filename'] = imgName.replace('/resources/', '')
        data['time'] = date = self.timer.now.strftime('%Y-%m-%d %H:%M:%S')
        data['cam_name'] = self.attr['camera_name']
        data['algo'] = self.attr['algo_name']
        data['algo_id'] = self.attr['algo_id']
        self.es.index(index=index, body=data)

    def send_img(self, frame, id_):
        date = self.timer.now.strftime('%Y-%-m-%-d_%H:%M:%S')
        imgName = f"/resources/{self.attr['id_account']}/{self.attr['id_branch']}/vehicle/{self.attr['camera_id']}/{date}_{id_}.jpg"
        imgPath = '/home' + imgName
        resizedFrame = cv2.resize(frame, (640, 480))
        drawing.saveImg(imgPath, resizedFrame)
        return imgName

    def send_alert(self, frame, track, labels):
        date = self.timer.now.strftime('%Y-%m-%d %H:%M:%S')
        if 'alerted-car' not in track.tag:
            imgName = self.send_img(frame, track.id)
            self.send_es('gmtc_searcher', date, labels, imgName)
            track.tag.add('alerted-car')

    def draw(self, frame, bikes, cars):
        for track in cars:
            x1, y1, x2, y2 = track.attr['xyxy']
            if point_in_poly(x2, y2, self.roi):
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,0), 2)
            labels = [track.attr['cls']]
            #print(labels) # car, truck
            if track.attr['cls'] != 'car':
                 self.send_alert(frame, track, labels) # TEMPARARILY COMMENTED
            elif 'carbrand' in track.dict:
                labels.append(track.dict['carbrand'])
                self.send_alert(frame, track, labels)   # TEMPARARILY COMMENTED
            
            if point_in_poly(x2, y2, self.roi):
                if labels[-1] == "Fukuda" or labels[-1] == "Yutong":
                    labels[-1] = "Ashok Leyland"
                
                if labels[-1] == "Honda":
                    labels[-1] = "Hyundai"
                
               # if labels[-1] == "Jeep":
               #     labels[-1] = "Mahindra"
                
                if labels[-1] == "Jianghuai" or labels[-1]=="Volkswagen" or labels[-1]=="Jeep": # or labels[-1]=="Lexus":
                    labels[-1] = "Suzuki"
               
                if labels[-1]=="Nissan":
                    labels[-1] = "Toyota"

                if labels[-1]=="Truck":
                    labels = ["3 Wheeler", "Autorickshaw"]

                drawing.putTexts2(frame, labels, x1, y1, size=1, thick=1, color=(0, 0, 0))

        for track in bikes:
            x1,y1,x2,y2 = track.attr['xyxy']
            if (x2-x1)<200 and (y2-y1)<300:
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)
            if 'helmet' not in track.dict:
                pass
            elif track.dict["helmet"]:
                drawing.putTexts(frame, ['Helmet Detected'], x1, y1, size=1, thick=1, color=(0,255,0))
            else:
                drawing.putTexts(frame, ['No Helmet!'], x1, y1, size=1, thick=1, color=(0,0,255))

            #if track.attr['cls'] == b'helmet':
            #    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            #    labels = None
            #elif track.attr['cls'] == 'motorbiker_helmet':
            #    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            #    labels = ['motorbike']
            #    drawing.putTexts(frame, ["No helmet", area], x1, y1, size=1, thick=1, color=(255,255,255))
            #elif track.attr['cls'] == 'motorbiker_no_helmet':
            #    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
            #    drawing.putTexts(frame, ["No helmet", area], x1, y1, size=1, thick=1, color=(255,255,255))
            #    labels = ['motorbike', 'no helmet']
            #if labels:
            #    self.send_alert(frame, track, labels)


    def run(self, frame, yoloDets, vehicleTracks):
        self.save_vcount(yoloDets)
        drawing.draw_rois(frame, self.roi,(0,0,255))
        bikes = self.helmet.run(frame, yoloDets, 0.01)
        cars = self.get_car(vehicleTracks, 0.1)
        self.carbrand.run(frame, cars)
        self.draw(frame, bikes, cars)
        self.outstream.write(frame)


