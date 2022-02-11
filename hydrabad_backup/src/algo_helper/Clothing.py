import os
import cv2
import zmq
from collections import deque, Counter
import matplotlib.pyplot as plt 
import use.drawing as drawing
import time
from tqdm import tqdm
import numpy as np
import ast


MYSQL_DB = os.environ.get('MYSQL_DB')
MYSQL_IP = os.environ.get('MYSQL_IP')

class ML_client:
    def __init__(self, docker_url='tcp://localhost:5631', shape=(1280,720)):
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(docker_url)
        self.shape = shape

    def detect(self, frame):
        #frame = cv2.resize(frame, (self.shape), interpolation=cv2.INTER_LINEAR)
        self.socket.send_multipart(frame)
        detection = self.socket.recv()
        dets = ast.literal_eval(detection.decode())
        #dets = self.format(dets, rois, roi_cfg, conf_thres=conf_thres, **kwargs)
        return dets

class Clothing:
    def __init__(self, timer, outstream, mysql_helper, algos_dict, elastic):    
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
            ["bottom_colour", "varchar(45)"]
            ]
        self.mysql_helper = mysql_helper
    
        self.mysql_helper.add_table('clothing', mysql_fields)
        self.timer = timer
        self.attr = algos_dict
        self.average_len = 10
        self.es = elastic
        self.now_t = time.time()
        self.count = 0
        self.time_diff = 0
        self.pbar = tqdm(total=1)
        self.clothing = ML_client()

    def get_average(self, track, labels):
        if 'clothes' not in track.dict:
            track.dict['clothes'] = [deque(['Unknown'], maxlen=self.average_len) for i in range(5)]
            
        isFull = False
        for i, label in enumerate(labels):
            if len(track.dict['clothes'][i]) >= self.average_len:
                isFull = True
                continue
            if label != 'Unknown':
                track.dict['clothes'][i].append(label)
                #if len(track.dict['clothes'][i]) >= self.average_len:
                #    isFull = True
        if isFull:
            avg_labels = []
            for q in track.dict['clothes']:
                avg_labels.append(Counter(q).most_common(1)[0][0])
            return avg_labels
        else:
            return None

    def send_es(self, index, date, labels, imgName):
        data = {}
        data['description'] = f'person with {labels[0]}, {labels[1]} top, {labels[2]}, {labels[3]} bottom, {labels[4]} shoes detected at {date} at {self.attr["camera_name"]}'
        data['filename'] = imgName.replace('/resources/', '')
        data['time'] = date
        data['cam_name'] = self.attr['camera_name']
        data['algo'] = self.attr['algo_name']
        data['algo_id'] = self.attr['algo_id']
        self.es.index(index=index, body=data)

    def send_img(self, frame, id):
        date = self.timer.now.strftime('%Y-%-m-%-d_%H:%M:%S')
        imgName = f"/resources/{self.attr['id_account']}/{self.attr['id_branch']}/cloth/{self.attr['camera_id']}/{date}_{id}.jpg"
        imgPath = '/home' + imgName
        resizedFrame = cv2.resize(frame, self.castSize)
        drawing.saveImg(imgPath, resizedFrame)
        return imgName

    def save_alert(self, frame, track, date, labels):
        if 'mysql_clothes' not in track.tag:
            track.tag.add('mysql_clothes')
            mysql_values = (track.id, date, self.attr['camera_id'], self.attr['camera_name'], 
                    self.attr['id_account'], self.attr['id_branch'], labels[0], labels[1], labels[2], labels[3],labels[4])
            self.mysql_helper.insert_fast('clothing', mysql_values)
            
            imgName = self.send_img(frame, track.id)
            self.send_es('gmtc_searcher', date, labels, imgName)

    def run(self, frame, personDets,yoloTrack):
        date = self.timer.now.strftime('%Y-%m-%d %H:%M:%S')

        #cmap = plt.get_cmap('tab20b')
        #colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]
        xyxy = str([det['xyxy'] for det in personDets]).encode()
        #xyxy = str([det['xyxy'] for det in personDets]).encode()
        results = self.clothing.detect([xyxy, str(frame.shape).encode(), frame])

        for i, det in enumerate(yoloTrack):
            x1,y1,x2,y2 = det.attr['xyxy']
            x,y,w,h = det.attr['xywh']
            #if w > 80 or h < 90 or h > 400:
            if w < 80 or h < 120:# or w > 140 :
               continue
           #drawing.putTexts(frame, [w], 30,30, size=0.6, thick=1, color= (255,255,255))
            #drawing.putTexts(frame, [w,h], x1,y1, size=0.6, thick=1, color= (255,255,255))
            #track_id = det.id[-3:].split("_")
            #track_id =track_id[1]

            #color = colors[int(track_id) % len(colors)]
            #color = [i * 255 for i in color]
            #if len(results[i]) < 4:
            #    continue
            top_colour = 'NotClear'
            bottom_colour = 'NotClear'
            footwear = 'NotClear'
            footwear_colour = 'NotClear'

            hat = results[i][0]
            data = results[i][2].split("-")
           
            data = results[i][4].split("-")
            if data[0] == 'UpperGarment':
                top_colour = results[i][6].split("-")
                top_colour = top_colour[1]

                bottom_colour = results[i][7].split("-")
                bottom_colour = bottom_colour[1]

                footwear = results[i][8].split("-")
                footwear = footwear[1]

                footwear_colour = results[i][9].split("-")
                footwear_colour = footwear_colour[1]
            else:
                top_colour = results[i][7].split("-")
                top_colour = top_colour[1]

                bottom_colour = results[i][8].split("-")
                bottom_colour = bottom_colour[1]

                footwear = results[i][9].split("-")
                footwear = footwear[1]

                footwear_colour = results[i][10].split("-")
                footwear_colour = footwear_colour[1]

            cv2.rectangle(frame, (x1,y1), (x2,y2),(0,0,0), 2)
            #cv2.rectangle(frame, (x2,y2-30), (x2+len(results[i][:-1])*17,y2), color, -1)
            #cv2.putText(frame, results[i], (x2, y2), 0, 0.4,color, 2)

            #self.id_ += 1
            #cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,0), 2)
            drawing.putTexts(frame, results[i][:-1], x2, y2, size=0.6, thick=1, color= (255,255,255))
            # hat_color[1]
            track_list = [hat,top_colour,bottom_colour,footwear,footwear_colour]
            self.save_alert(frame, det, date, track_list)
        self.outstream.write(frame)
        


