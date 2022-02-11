import cv2
import ffmpeg
import numpy as np
import use.drawing as drawing
import use.point_in_poly as point_in_poly
from use.tracking9 import Tracker
import uuid
import os

MYSQL_DB = os.environ.get('MYSQL_DB')
MYSQL_IP = os.environ.get('MYSQL_IP')

class AOD_worker:
    def __init__(self, mask, timer):
        self.bgSubtractor = cv2.createBackgroundSubtractorMOG2()
        #self.bgSubtractor.setNMixtures(5) # increase to reduce detection
        self.bgSubtractor.setNMixtures(12) # increase to reduce detection
        self.bgSubtractor.setDetectShadows(False)
        #self.bgSubtractor.setVarThreshold(200) # increase to reduce detection
        self.bgSubtractor.setVarThreshold(700) # increase to reduce detection
        #self.bgSubtractor.setVarThreshold(720) # increase to reduce detection
        self.minAODSize = 2
        self.maxAODSize = 5000
        self.minAODduration = 5
        self.mask = mask

        self.lastTracks = []
        self.iniCount = 0
        self.iniCountRequired = 5
        self.frameShape = (320,180)
        self.frameShapeOriginal = None
        self.bgImg = None

        self.tracker = Tracker((1280,720))
        self.aods_display = []
        self.timer = timer

    def isInitialized(self):
        if self.iniCount > self.iniCountRequired:
            return True
        else:
            self.iniCount += 1
            return False

    def removePeople(self, frame, yoloDets):
        for det in yoloDets:
            x1,y1,x2,y2 = self.resizeFromOriginal(det['xyxy'])
            frame[y1:y2,x1:x2] = self.bgImg[y1:y2,x1:x2]

    def maskArea(self, frame):
        frame = frame.copy()
        for x1,y1,x2,y2 in self.mask:
            frame[y1:y2, x1:x2] = 0
        return frame

    def initializeBG(self, frame, frame0):
        self.bgSubtractor.apply(frame, learningRate=.01)
        if self.iniCount >= self.iniCountRequired:
            self.frameShapeOriginal = (frame0.shape[1], frame0.shape[0])
            self.bgImg = frame.copy()

    def compareBG_with(self, frame):
        mask = self.bgSubtractor.apply(frame, learningRate=0)
        ret, mask = cv2.threshold(mask,200,255,cv2.THRESH_BINARY)
        kernel = np.ones((2,2), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.erode(mask, kernel, iterations=1)
        return mask

    def getAOD(self, mask):
        aods = []
        try:
            _, cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except:
            cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i, cnt in enumerate(cnts):
            aodSize = cv2.contourArea(cnt)
            if self.minAODSize < aodSize < self.maxAODSize:
                aod = {}
                x1,y1,w,h = cv2.boundingRect(cnt)
                x2 = x1 + w
                y2 = y1 + h
                x = int(x1 + w/2)
                y = int(y1 + h/2)
                aod['xyxy'] = [x1,y1,x2,y2]
                aod['xywh'] = [x,y,w,h]
                aod['size'] = aodSize

                x1 = int(max(0, x1-w*.1))
                y1 = int(max(0, y1-h*.1))
                x2 = int(min(mask.shape[1], x2+w*.1))
                y2 = int(min(mask.shape[0], y2+h*.1))
                aod['xyxy_enlarge'] = [x1,y1,x2,y2]
                aods.append(aod)
        return aods

    def resizeFromOriginal(self, xyxy):
        X, Y = self.frameShape
        X0, Y0 = self.frameShapeOriginal
        x1,y1,x2,y2 = xyxy

        x1 = int(x1/X0*X)
        y1 = int(y1/Y0*Y)
        x2 = int(x2/X0*X)
        y2 = int(y2/Y0*Y)
        return (x1,y1,x2,y2)

    def resizeToOriginal(self, xyxy):
        X, Y = self.frameShape
        X0, Y0 = self.frameShapeOriginal
        x1,y1,x2,y2 = xyxy

        x1 = int(x1/X*X0)
        y1 = int(y1/Y*Y0)
        x2 = int(x2/X*X0)
        y2 = int(y2/Y*Y0)
        return (x1,y1,x2,y2)

    def updateBG(self, frame, tracks):
        frameWithoutAOD = frame.copy()
        for track in tracks:
            if 'alerted' not in track.tag:
                x1,y1,x2,y2 = track.attr['xyxy_enlarge']
                frameWithoutAOD[y1:y2,x1:x2] = self.bgImg[y1:y2,x1:x2]
        self.bgImg = frameWithoutAOD
        self.bgSubtractor.apply(self.bgImg, learningRate=.01)

    def updateTrack(self, aods):
        X, Y = self.frameShape
        X0, Y0 = self.frameShapeOriginal
        dets_out = []
        for det in aods:
            det_out = det.copy()
            det_out['xyxy'] = self.resizeToOriginal(det['xyxy'])
            det_out['xywh'] = self.resizeToOriginal(det['xywh'])
            dets_out.append(det_out)
        tracks = self.tracker.update(dets_out)
        self.lastTracks = tracks
        return tracks

    def updateTrackDuration(self, tracks, dt):
        for track in tracks:
            if 'duration' not in track.dict:
                track.dict['duration'] = dt
                #track.dict['duration'] = 0
            else:
                track.dict['duration'] += dt
                #track.dict['duration'] += 1 

    def detect(self, frame, yoloDets):
        frame0 = frame.copy()
        frame = self.maskArea(frame)
        frame = cv2.resize(frame, self.frameShape)
        if self.isInitialized():
            self.removePeople(frame, yoloDets)
            mask = self.compareBG_with(frame)
            aods = self.getAOD(mask)
            tracks = self.updateTrack(aods)
            self.updateBG(frame, tracks)
            self.updateTrackDuration(tracks, self.timer.dt)
        else:
            self.initializeBG(frame, frame0)
            tracks = []
        return tracks


class AOD:
    def __init__(self, mask, timer, outstream, mysql_helper, algos_dict, elastic):
        self.outstream = outstream
        self.timer = timer
        self.attr = algos_dict

        mysql_fields_1 = [
            ["time","datetime"],
            ["zone", "int(11)"],
            ["cam_name","varchar(45)"],
            ["id", "varchar(45)"],
            ["cam_id","varchar(45)"],
            ["id_account","varchar(45)"],
            ["id_branch", "varchar(45)"],
            ["track_id", "varchar(45)"]
            ]

        self.mysql_helper = mysql_helper
        self.mysql_helper.add_table('aod', mysql_fields_1)
        self.es = elastic
        self.castSize = (640, 480)
        self.aod_worker = AOD_worker([], timer)

    def send_es(self, index, date, imgName):
        data = {}
        data['description'] = f'Abandoned Object detected at {date} at {self.attr["camera_name"]}'
        data['filename'] = imgName
        data['time'] = date
        data['cam_name'] = self.attr['camera_name']
        data['algo'] = self.attr['algo_name']
        data['algo_id'] = self.attr['algo_id']
        self.es.index(index=index, body=data)

    def send_img(self, frame, aod_id):
        date = self.timer.now.strftime('%Y-%-m-%-d_%H:%M:%S')
        imgName = f"/resources/{self.attr['id_account']}/{self.attr['id_branch']}/aod/{self.attr['camera_id']}/{date}_{aod_id}.jpg"
        imgPath = '/home' + imgName
        resizedFrame = cv2.resize(frame, self.castSize)
        drawing.saveImg(imgPath, resizedFrame)
        return imgName

    def send_alert(self, track, frame, date, uuid0):
        if 'mysql' not in track.tag:
            track.tag.add('mysql')
            mysql_values1 = (date, 0, self.attr['camera_name'], uuid0, self.attr['camera_id'],
                    self.attr['id_account'], self.attr['id_branch'], track.id)
            mysql_values2 = (uuid0, 'aod', date, date, 'NULL', self.attr['id_account'], self.attr['id_branch'], 0, 'NULL', 'NULL')
            self.mysql_helper.insert_fast('aod', mysql_values1)
            self.mysql_helper.insert_fast('tickets', mysql_values2)
            imgName = self.send_img(frame, track.id)
            self.send_es('gmtc_searcher', date, imgName)

    def run(self, frame, yoloDets):
        #print('aod recevei frame')
        date = self.timer.now.strftime('%Y-%m-%d %H:%M:%S')
        aodTracks = self.aod_worker.detect(frame, yoloDets)
        #print(aodTracks)

        for track in aodTracks:
            # drawing.putTexts(frame, [track.dict['duration']], 30, 80, size=1, thick=2, color=(0,0,255))
            #drawing.putTexts(frame, [self.aod_worker.minAODduration], 30, 100, size=1, thick=2, color=(0,0,255))
            if track.dict['duration'] >= self.aod_worker.minAODduration:
                uuid0 = str(uuid.uuid4())
                x1,y1,x2,y2 = track.attr['xyxy']
                #print("aod")
                cv2.rectangle(frame, (x1,y1), (x2, y2), (0, 0,255), 2)
                drawing.putTexts(frame, ["aod detected"], 30, 30, size=1, thick=2, color=(255,255,255))
#                drawing.putTexts(frame, ['unattended'], x1, y1, size=1, thick=2, color=(0,0,255))
                self.send_alert(track, frame, date, uuid0)

        self.outstream.write(frame)
