
import os
import zmq
import ast
import ffmpeg
import cv2
import time
from datetime import datetime
from elasticsearch import Elasticsearch

from use.tracking9 import Tracker
from use.stream import createFileVideoStream
from use.mysql2 import Mysql
from algo_helper.yolo_algo import CrowdCounter, VehicleAll, Loiter
from algo_helper.fr import FR
from algo_helper.clothing import Clothing
from algo_helper.anpr import ANPR
from algo_helper.violence import Violence
from algo_helper.aod import AOD

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
        for cls in ['person', 'vehicle']:
            self.yoloTracker[cls] = Tracker((1280,720))

    def format(self, dets_all):
        dets = {cls:[] for cls in {'person', 'vehicle'}}
        for cls, conf, (x1,y1,x2,y2) in dets_all:
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
        return dets

    def detect(self, frame):
        self.yoloSocket.send(frame)
        detection = self.yoloSocket.recv()
        yoloDets = ast.literal_eval(detection.decode())
        yoloDets = self.format(yoloDets)
        return yoloDets

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
        print(self.output_path)
        frame = cv2.resize(frame, self.outxy)
        self.process.stdin.write(frame)

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
            self.algos_func[key] = CrowdCounter(timer, outstream, self.mysql_helper, algos_dict[key], elastic)
        if 'vehicle_type' in algos_dict:
            key = 'vehicle_type'
            outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
            self.algos_func[key] = VehicleAll(timer, outstream, self.mysql_helper, algos_dict[key], elastic)
        if 'fr' in algos_dict:
            key = 'fr'
            outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
            self.algos_func[key] = FR(timer, outstream, self.mysql_helper, algos_dict[key], elastic)
        if 'clothing' in algos_dict:
            key = 'clothing'
            outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
            self.algos_func[key] = Clothing(timer, outstream, self.mysql_helper, algos_dict[key], elastic)
        if 'anpr' in algos_dict:
            key = 'anpr'
            outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
            self.algos_func[key] = ANPR(timer, outstream, self.mysql_helper, algos_dict[key], elastic)
        if 'loiter' in algos_dict:
            key = 'loiter'
            outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
            self.algos_func[key] = Loiter(timer, outstream, self.mysql_helper, algos_dict[key], elastic)
        if 'violence' in algos_dict:
            key = 'violence'
            outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
            self.algos_func[key] = Violence(timer, outstream, self.mysql_helper, algos_dict[key], elastic)
        if 'aod' in algos_dict:
            key = 'aod'
            outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
            self.algos_func[key] = AOD([], timer, outstream, self.mysql_helper, algos_dict[key], elastic)

    def run(self, frame, yoloDets, yoloTracks):
        if 'crowd' in self.algos_func:
            self.algos_func['crowd'].run(frame.copy(), yoloDets['person'])
        if 'vehicle_type' in self.algos_func:
            self.algos_func['vehicle_type'].run(frame.copy(), yoloDets['person']+yoloDets['vehicle'], yoloTracks['vehicle'])
        if 'fr' in self.algos_func:
            self.algos_func['fr'].run(frame.copy())
        if 'clothing' in self.algos_func:
            self.algos_func['clothing'].run(frame.copy(), yoloTracks['person'])
        if 'anpr' in self.algos_func:
            self.algos_func['anpr'].run(frame.copy())
        if 'loiter' in self.algos_func:
            self.algos_func['loiter'].run(frame.copy(), yoloTracks['person'])
        if 'violence' in self.algos_func:
            self.algos_func['violence'].run(frame.copy(), yoloTracks['person'])
        if 'aod' in self.algos_func:
            self.algos_func['aod'].run(frame.copy(), yoloDets['person'])

        if self.timer.hasExceed('all_algos', 5):
            self.mysql_helper.commit_all()

def main(cam_dict):
    cam_id, input_path, algos_dict = cam_dict['camera_id'], cam_dict['rtsp_in'], cam_dict['algos']
    print(f'starting {input_path}')
    fvs = createFileVideoStream('live', input_path, (1280,720), False, 0)
    yolo = Yolo()
    timer = Timer()
    algosMan = AlgosMan(timer, cam_id, algos_dict)

    while True:
        #print('a')
        timer.update()
        frame = fvs.read()
        yoloDets = yolo.detect(frame)
        yoloTracks = yolo.track(yoloDets)
        algosMan.run(frame, yoloDets, yoloTracks)

