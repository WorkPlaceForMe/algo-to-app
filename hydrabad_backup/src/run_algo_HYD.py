# ubuntu@office-am-4:~/hydrabad/src$ vi run_algo_Hyderabad.py
import os
import zmq
import ast
import ffmpeg
import numpy as np
import cv2
import time
from datetime import datetime
from elasticsearch import Elasticsearch
from tracking.tracker.byte_tracker import BYTETracker
from use.stream import createFileVideoStream
from use.mysql2 import Mysql
from algo_helper.Loitering_Crowd_Sociald_HYD import Loiter
from algo_helper.Crowd_Sociald_HYD import CrowdCounter
from algo_helper.Carbrand_Helmet_HYD import VehicleAll
from algo_helper.FR_HYD import FR
from algo_helper.Clothing_HYD_LKO import Clothing 
from algo_helper.ANPR_Weapon_HYD_LKO import ANPR
from algo_helper.AOD_HYD import AOD
from algo_helper.Violence_Collapsing_HYD import Merged
#from algo_helper.violence2_HYD import Merged
from utils import CFEVideoConf, image_resize
from algo_helper.Weapon2_HYD import Weapon

OUT_XY = (640,480)
MYSQL_DB = os.environ.get('MYSQL_DB')
MYSQL_IP = os.environ.get('MYSQL_IP')
ES_USER = "elastic"
ES_PASSWORD = "a73jhx59F0MC39OPtK9YrZOA"
ES_ENDPOINT = "https://e99459e530344a36b4236a899b32887a.westus2.azure.elastic-cloud.com:9243"

class Timer:
    def __init__(self):
        self.now = datetime.now()
        self.now_t = time.time()
        self.count = 0
        self.dt = 0
        self.last_sent = {}

    def update(self):
        self.now = datetime.now()
        self.count += 1
        self.dt = time.time() - self.now_t
        self.now_t = time.time()

    def hasExceed(self, name, duration):
        if name not in self.last_sent:
            self.last_sent[name] = self.now_t
            return False
        elif self.now_t - self.last_sent[name] > duration:
            self.last_sent[name] = self.now_t
            return True
        else:
            return False

class Yolo:
    def __init__(self):
        context = zmq.Context()
        self.yoloSocket = context.socket(zmq.REQ)
        self.yoloSocket.connect("tcp://deep_yolo:5601")
        self.yoloTracker = {}
        for cls in ['person', 'vehicle', 'backpack','suitcase']:
            #self.yoloTracker[cls] = Tracker((1280,720))
            self.yoloTracker[cls] = BYTETracker()

    def format(self, dets_all, mot=False):
        dets = {cls:[] for cls in {'person', 'vehicle', 'backpack', 'suitcase'}}
        for cls, conf, (x1,y1,x2,y2) in dets_all:
            if mot:
                det = [int(x1),int(y1),int(x2),int(y2),conf,cls]
            else:
                det = {}
                x, y = (x1+x2)/2, (y1+y2)/2
                w, h = x2-x1, y2-y1
                det['conf'] = conf
                det['cls'] = cls
                det['xyxy'] = [x1,y1,x2,y2]
                det['xywh'] = [x,y,w,h]
            if cls == 'person' and conf > .1:
                dets['person'].append(det)
            elif cls in {'car','truck','motorbike','bus','aeroplane','train','boat'} and conf > .8:
                dets['vehicle'].append(det)
            elif cls == 'backpack' and conf > .1:
                dets['backpack'].append(det)
            elif cls == 'suitcase' and conf > .1:
                dets['suitcase'].append(det)
        return dets

    def detect(self, frame):
        self.yoloSocket.send(frame)
        detection = self.yoloSocket.recv()
        yoloDets = ast.literal_eval(detection.decode())
        yoloMot = self.format(yoloDets, True)
        yoloDets = self.format(yoloDets)
        # yoloDets = filter_overlap(yoloDets, 0.8)
        return yoloDets, yoloMot

    def track(self, dets_all):
        tracks_all = {}
        for cls, tracker in self.yoloTracker.items():
            tracks = tracker.update(dets_all[cls])
            tracks_all[cls] = tracks
        return tracks_all

class OutStream:
    def __init__(self, outxy, output_path):
        self.output_path = output_path
        self.outxy = outxy
        self.process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(outxy[0], outxy[1]))
            .output('{}'.format(output_path))
            .overwrite_output()
            .global_args('-loglevel', 'error')
            .run_async(pipe_stdin=True))

    def write(self, frame):
        frame = cv2.resize(frame, self.outxy)
        self.process.stdin.write(frame)
        # print(self.output_path)

class AlgosMan:
    def __init__(self, timer, cam_id, algos_dict):
        self.algos_func = {}
        self.mysql_helper = Mysql({"ip":MYSQL_IP, "user":'graymatics', "pwd":'graymatics', "db":MYSQL_DB, "table":""})
        elastic = Elasticsearch([ES_ENDPOINT], http_auth=(ES_USER, ES_PASSWORD))
        self.timer = timer
        mysql_fields = [
                ['id','varchar(40)'],
                ['type','varchar(40)'],
                ["createdAt","datetime"],
                ["updatedAt","datetime"],
                ["assigned","varchar(40)"],
                ["id_account","varchar(40)"], 
                ["id_branch","varchar(40)"],
                ["level","int(11)"], 
                ['reviewed',"varchar(45)"]
                ]
        self.mysql_helper.add_table('tickets', mysql_fields)
        if 'crowd' in algos_dict:
            key = 'crowd'
            outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
            self.algos_func[key] = Weapon(timer, outstream, self.mysql_helper, algos_dict[key], elastic)
            #self.algos_func[key] = CrowdCounter(timer, outstream, self.mysql_helper, algos_dict[key], elastic)
        if 'fr' in algos_dict:
            key = 'fr'
            outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
            self.algos_func[key] = FR(timer, outstream, self.mysql_helper, algos_dict[key], elastic)
        if 'violence' in algos_dict:
            key = 'violence'
            outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
            self.algos_func[key] = Merged(timer, outstream, self.mysql_helper, algos_dict[key], elastic)
        if 'loiter' in algos_dict:
            key = 'loiter'
            outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
            self.algos_func[key] = Loiter(timer, outstream, self.mysql_helper, algos_dict[key], elastic)
        if 'aod' in algos_dict:
            key = 'aod'
            outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
            self.algos_func[key] = AOD([], timer, outstream, self.mysql_helper,algos_dict[key], elastic)
        if 'anpr' in algos_dict:
            key = 'anpr'
            outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
            self.algos_func[key] = ANPR(timer, outstream, self.mysql_helper, algos_dict[key], elastic)
#        if 'social' in algos_dict:
#            key = 'social'
#            outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
#            self.algos_func[key] = Weapon(timer, outstream, self.mysql_helper, algos_dict[key], elastic)
#            self.algos_func[key] = CrowdCounter(timer, outstream, self.mysql_helper, algos_dict[key], elastic)
        if 'clothing' in algos_dict:
            key = 'clothing'
            outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
            self.algos_func[key] = Clothing(timer, outstream, self.mysql_helper, algos_dict[key], elastic)
#        if 'vehicle_type' in algos_dict:
#            key = 'vehicle_type'
#            outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
#            self.algos_func[key] = VehicleAll(timer, outstream, self.mysql_helper, algos_dict[key], elastic)

    def run(self, frame, yoloDets, yoloTracks):
        if 'crowd' in self.algos_func:
            self.algos_func['crowd'].run(frame.copy(),yoloDets['person'])
        if 'fr' in self.algos_func:
            self.algos_func['fr'].run(frame.copy(), yoloTracks['person'])
        if 'violence' in self.algos_func:
            #self.algos_func['violence'].run(frame.copy(), yoloDets['person'],yoloTracks['person'])
            self.algos_func['violence'].run(frame.copy(),yoloDets['person'],yoloTracks['person'])
        if 'loiter' in self.algos_func:
            self.algos_func['loiter'].run(frame.copy(), yoloTracks['person'])
        if 'aod' in self.algos_func:
            self.algos_func['aod'].run(frame.copy(),yoloDets['backpack']+yoloDets['suitcase'])
        if 'anpr' in self.algos_func:
            self.algos_func['anpr'].run(frame.copy())
#        if 'social' in self.algos_func:
#            self.algos_func['social'].run(frame.copy(), yoloDets['person'], yoloTracks['person'])
#            self.algos_func['social'].run(frame.copy(),yoloDets['person'])
        if 'clothing' in self.algos_func:
            self.algos_func['clothing'].run(frame.copy(), yoloDets['person'], yoloTracks['person'])
#        if 'vehicle_type' in self.algos_func:
#            self.algos_func['vehicle_type'].run(frame.copy(), yoloDets['person']+yoloDets['vehicle'], yoloTracks['vehicle'])
        if self.timer.hasExceed('all_algos', 5):
            self.mysql_helper.commit_all()

#img_path = 'graymaticslogo.png'
#logo = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
#logoMark = cv2.imread(img_path, -1)

def main(cam_dict):
    cam_id, input_path, algos_dict = cam_dict['camera_id'], cam_dict['rtsp_in'], cam_dict['algos']
    print(f'starting {input_path}')
    fvs = createFileVideoStream('video', input_path, (1280,720), True, 0, skip=3)
    yolo = Yolo()
    timer = Timer()
    print(cam_id)
    algosMan = AlgosMan(timer, cam_id, algos_dict)

    print('run_algo_HYD.py')

    #WATERMARK----------------------------------------------
#    img_path = 'graymaticslogo.png'
#    logo = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
#    logo = cv2.imread(img_path, -1)
#    logo = logoMark
#    watermark = image_resize(logo, height=50)
#    watermark = cv2.cvtColor(logo, cv2.COLOR_BGR2BGRA)
    #---------------------------------------------------------

    while True:
        timer.update()
        frame = fvs.read()
        yoloDets, yoloMot = yolo.detect(frame)
        yoloTracks = yolo.track(yoloMot)

        #WATERMARK------------------------------------------------------
        img_path = 'graymaticslogo.png'
#       logo = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
        logo = cv2.imread(img_path, -1)
#       logo = logoMark
        watermark = image_resize(logo, height=50)
        watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2BGRA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        frame_h, frame_w, frame_c = frame.shape
        # overlay with 4 channels BGR and Alpha
        overlay = np.zeros((frame_h, frame_w, 4), dtype='uint8')
        watermark_h, watermark_w, watermark_c = watermark.shape
        # replace overlay pixels with watermark pixel values
        for i in range(0, watermark_h):
            for j in range(0, watermark_w):
                if watermark[i,j][3] != 0:
                    offset = 10
                    h_offset = frame_h - watermark_h - offset
                    w_offset = frame_w - watermark_w - offset
                    overlay[h_offset + i, w_offset+ j] = watermark[i,j]
        cv2.addWeighted(overlay, 1, frame, 1, 0.0, frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        #watermark = cv2.cvtColor(watermark, cv2.COLOR_BGRA2BGR)
        #---------------------------------------------------------------
        algosMan.run(frame, yoloDets, yoloTracks)




