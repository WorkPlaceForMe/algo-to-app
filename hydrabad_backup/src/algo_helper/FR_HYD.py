import os
import cv2
import ast
import zmq
from collections import deque, Counter
import numpy as np
import glob
import ujson
import time
import matplotlib.pyplot as plt
import use.drawing as drawing
from use.tracking9 import Tracker

MYSQL_DB = os.environ.get('MYSQL_DB')
MYSQL_IP = os.environ.get('MYSQL_IP')

class FR:
    def __init__(self, timer, outstream, mysql_helper, algos_dict, elastic):
        context = zmq.Context()
        self.faceSocket = context.socket(zmq.REQ)
        self.faceSocket.connect("tcp://fr-server:5603")
        self.knownFace = KnownData('face_hydrabad/', "tcp://fr-server:5603")
        self.face_thres = 0.1
        self.knownface_thres = 0.5
        self.outstream = outstream
        self.tracker = Tracker((1280,720))
        self.average_len = 2
        self.castSize = (640, 480)
        mysql_fields = [
            ['id','varchar(45)'],
            ['time','datetime'],
            ['name','varchar(45)'],
            ['gender','varchar(20)'],
            ['age','varchar(20)'],
            ['cam_id','varchar(45)'],
            ['cam_name','varchar(45)'],
            ['id_account','varchar(45)'],
            ['id_branch','varchar(45)']
            ]
        self.mysql_helper = mysql_helper
        self.mysql_helper.add_table('faces', mysql_fields)
        
        self.timer = timer
        self.count = 0
        self.attr = algos_dict
        self.es = elastic
        self.classLists = {}
        self.classLists['gender'] = ['Male', 'Female']
        #self.classLists['age'] = ['Adult', 'Adult', 'Adult', 'Adult', 'Adult', 'Middle Age', 'Middle Age']
        self.classLists['age'] = ['Adult', 'Adult', 'Adult', 'Adult', 'Adult', 'Middle Age', 'Middle Age']


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
            det['age'] = np.array(age)
            det['gender'] = np.array(gender) + np.array([-.1, .4])

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

    def save_alert(self, frame, track, date, avg_name, avg_age, avg_gender):
        print("track:", track)
        if 'mysql_face' not in track.tag:
            track.tag.add('mysql_face')
            mysql_values = (track.id, date, avg_name, avg_gender, avg_age,"Null",
                self.attr['camera_id'], self.attr['camera_name'], self.attr['id_account'], self.attr['id_branch'])
            self.mysql_helper.insert_fast('faces', mysql_values)
            imgName = self.send_img(frame, track.id)
            self.send_es('gmtc_searcher', date, avg_name, imgName)

    def send_img(self, frame, id):
        date = self.timer.now.strftime('%Y-%-m-%-d_%H:%M:%S')
        #imgName = f"/resources/{self.attr['id_account']}/{self.attr['id_branch']}/fr/{self.attr['camera_id']}/{date}_{id}.jpg"
        imgName = f"/resources/{self.attr['id_account']}/{self.attr['id_branch']}/fr/{self.attr['camera_id']}/{date}.jpg"
        imgPath = '/home' + imgName
        resizedFrame = cv2.resize(frame, self.castSize)
        drawing.saveImg(imgPath, resizedFrame)
        return imgName

    def send_es(self, index, date, name, imgName):
        data = {}
        data['description'] = f'{name} at {date} at {self.attr["camera_name"]}'
        data['filename'] = imgName.replace('/resources/', '')
        data['time'] = date
        data['cam_name'] = self.attr['camera_name']
        data['algo'] = self.attr['algo_name']
        data['algo_id'] = self.attr['algo_id']
        self.es.index(index=index, body=data)

    def get_average_ag(self, track):
        if 'age_sum' not in track.dict:
            track.dict['age_count'] = 1
            track.dict['age_sum'] = track.attr['age']
            track.dict['gender_sum'] = track.attr['gender']
            return None, None
        elif 'age' in track.dict:
            return track.dict['age'], track.dict['gender']
        else:
            track.dict['age_count'] += 1
            track.dict['age_sum'] += track.attr['age']
            track.dict['gender_sum'] += track.attr['gender']
            if track.dict['age_count'] > self.average_len:
                for key in ('age', 'gender'):
                    max_score = np.amax(track.dict[key+'_sum'], axis=0)
                    max_ind = np.where(track.dict[key+'_sum']==max_score)[0][0]
                    track.dict[key] = self.classLists[key][max_ind]
            return None, None


    def run(self, frame, stream=True):
        self.faceSocket.send(frame)
        message = self.faceSocket.recv()
        faceDets = ast.literal_eval(message.decode())
        faceDets = self.format(faceDets)
        tracks = self.tracker.update(faceDets)
        date = self.timer.now.strftime('%Y-%m-%d %H:%M:%S')
        #tracks = self.filter_for_crowd(yoloTracks)
        #self.crowd(frame, tracks)
       # tracks2 = []
        for track in tracks:
            x1,y1,x2,y2 = track.attr['xyxy']
            x,y,w,h = track.attr['xywh']
            if w > 200:
                continue
          #  tracks2.append(track)
           # dets = [t.attr for t in tracks2]
           # dets = filter_overlap(dets, .5)
           # for det in dets:
           #     x1,y1,x2,y2 = det['xyxy']
        #    drawing.putTexts(frame, [f'people count: {len(tracks2)}'], 30, 30, size=1, thick=1, color=(255,255,255))

            #cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
            known_name, known_gender, known_age = self.knownFace.getName(track.attr['feature'], thres=self.knownface_thres)
            #avg_age, avg_gender = self.get_average_ag(track)
            #track_id = track.id[-3:].split("_")
            #drawing.putTexts(frame, [track_id], 30, 90, size=1, thick=2, color=(0,0,255))
            #self.count += 1
            #track_id =self.count
            


            if known_name is not None:
                self.count += 1
                track_id =self.count
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
                drawing.putTexts(frame, [known_name,known_gender,known_age], x1, y1, size=1, thick=1, color=(255,255,255))
                self.save_alert(frame, track, date, known_name, known_age, known_gender) #TEMPERARILY COMMENTED
            else:
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,0), 1)
                #self.save_alert(frame, track, date, known_name, avg_age, avg_gender)

            #tracks = self.filter_for_crowd(track)

        if stream:
            self.outstream.write(frame)
            
            
            
class KnownData():
    def __init__(self, dir0='face_hydrabad', port='tcp://localhost:5605'):
        self.dir = dir0
        self.attrs = {}
        self.features = []
        context = zmq.Context()
        self.faceSocket = context.socket(zmq.REQ)
        self.faceSocket.connect(port)
        self.fileList = []

        self.load()

    def isChanged(self):
        fileList = glob.glob('{}{}'.format(self.dir, '*/*'))
        if fileList != self.fileList:
            self.fileList = fileList
            return True
        else:
            return False

    def createImg(self, img):
        imgBG = np.zeros((720,1280,3), dtype=np.uint8)
        imgBG[:img.shape[0],:img.shape[1]] = img
        return imgBG

    def load(self, xy=(300,300)):
        nextId = 0
        userDirs = glob.glob(self.dir + '*')
        self.features = []
        for userDir in userDirs:
            if userDir[-3:] == 'txt':
                continue
            jsonPath = userDir + '/attr.json'
            with open(jsonPath, 'r') as f:
                print(jsonPath)
                attr = ujson.load(f)
            attr['nameid'] = attr['id']

            imgPaths = glob.glob(userDir + '/*.jpg')
            imgPaths.extend(glob.glob(userDir + '/*.png'))
            for imgPath in imgPaths:
                img = cv2.imread(imgPath)
                #img = cv2.resize(img, xy) # to improve: fix dimension of uploaded photo
                img = self.createImg(img)
                self.faceSocket.send(img)
                message = self.faceSocket.recv()
                message = ast.literal_eval(message.decode())
                if len(message) != 1:
                    continue
                feature = message[0][-1]
                self.features.append(feature)
                self.attrs[nextId] = attr
                nextId += 1
        self.features = np.array(self.features)

    def getName(self, feature, thres=0.65):
        # get similarity
        feature = np.array([feature])
        if len(self.features) == 0:
            name = None
            gender = None
            age = None
        else:
            similar = np.einsum('ij,kj->ik', feature, self.features) # [n,m]

            # get index of most similar feature
            max_score = np.amax(similar, axis=1) # n
            if max_score > thres:
                max_ind = np.where(similar==max_score[:,None])[1] # n
                #result = [True, {k:[v,int(max_score*100)] for k,v in self.attrs[max_ind[0]].items()}]
                #result = {k:[v,int(max_score*100)] for k,v in self.attrs[max_ind[0]].items()}['username']
                name = self.attrs[max_ind[0]]['username']
                gender = self.attrs[max_ind[0]]['gender']
                age = self.attrs[max_ind[0]]['age']
            else:
                name = None
                gender = None
                age = None
        return name,gender,age
