import os
import cv2
import ast
import zmq
from collections import deque, Counter

import use.drawing as drawing
#from use.tracking9 import Tracker as Tracker2
from algo_helper.anpr_use import detect
from algo_helper.anpr_use.cent_tracking import Tracker
import algo_helper.anpr_use.ocr.server1 as ocr_server1
import algo_helper.anpr_use.ocr.server2 as ocr_server2

MYSQL_DB = os.environ.get('MYSQL_DB')
MYSQL_IP = os.environ.get('MYSQL_IP')


class ANPR:
    def __init__(self, timer, outstream, mysql_helper, algos_dict, elastic):
        context = zmq.Context()
        self.plateSocket = context.socket(zmq.REQ)
        self.plateSocket.connect("tcp://anpr-server:5611")
        self.outstream = outstream
        self.castSize = (640, 480)
        self.confidence = 0.1
        self.screenXY = (1280, 720)
        self.yoloTracker = Tracker()
        mysql_fields = [
                ['track_id','varchar(45)'],
                ['time','datetime'],
                ['cam_id','varchar(45)'],
                ['cam_name','varchar(45)'],
                ['id_account','varchar(45)'],
                ['id_branch','varchar(45)'],
                ["plate","varchar(20)"]
                ]
        self.mysql_helper = mysql_helper
        self.mysql_helper.add_table('plate', mysql_fields)
        self.attr = algos_dict
        self.average_count = 10
        self.timer = timer
        self.es = elastic

    def send_mysql(self, date, track, lp_number):
        mysql_values = [track.id, date, self.attr['camera_id'], self.attr['camera_name'], 
                self.attr['id_account'], self.attr['id_branch'], lp_number]
        self.mysql_helper.insert_fast('plate', mysql_values)

    def send_es(self, index, date, lp_number, imgName):
        data = {}
        data['description'] = f'plate number {lp_number} detected at {date} at {self.attr["camera_name"]}'
        data['filename'] = imgName.replace('/resources/', '')
        data['time'] = date
        data['cam_name'] = self.attr['camera_name']
        data['algo'] = self.attr['algo_name']
        data['algo_id'] = self.attr['algo_id']
        self.es.index(index=index, body=data)

    def send_img(self, frame, id):
        date = self.timer.now.strftime('%Y-%-m-%-d_%H:%M:%S')
        imgName = f"/resources/{self.attr['id_account']}/{self.attr['id_branch']}/anpr/{self.attr['camera_id']}/{date}_{id}.jpg"
        imgPath = '/home' + imgName
        resizedFrame = cv2.resize(frame, self.castSize)
        drawing.saveImg(imgPath, resizedFrame)
        return imgName

    def send_alert(self, frame, lp_number, track):
        if lp_number is not None and 'plate' not in track.tag:
            date = self.timer.now.strftime('%Y-%m-%d %H:%M:%S')
            self.send_mysql(date, track, lp_number)
            imgName = self.send_img(frame, track.id)
            self.send_es('gmtc_searcher', date, lp_number, imgName)
            track.tag.add('plate')

    def run(self, frame):
        frame0 = frame.copy()
        frame_top = frame.copy()
        self.plateSocket.send(frame)
        message = self.plateSocket.recv()
        plateDets = ast.literal_eval(message.decode())
        plateDets = detect.formatPlate(plateDets, None, self.confidence, self.screenXY)
        yoloTracks = self.yoloTracker.update(plateDets)

        for i, track in enumerate(yoloTracks):
            x1,y1,x2,y2 = track.attr['xyxy']
            x,y,w,h = track.attr['xywh']
            lp_number = None
            if track.miss == 0:
                pass
                x_det = x2 - x1
                y_det = y2 - y1
                plate_ratio = x_det / y_det
                # Second Row, if exists
                if plate_ratio < 2.5:
                    y1, y2 = int(y1 + h*.35), int(y2 + h*.05) #h*.525 #.48 .05
                    mask_y1 = int(y1 + h*.48)
                    mask_y2 = int(y1 + h*.1)
                    mask_left = int(x1 + w*.14)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                    frame0[mask_y1:y2, x1:x2] = 0
                    frame0[y1:mask_y2, x1:x2] = 0
                    frame0[y1:y2, x1:mask_left] = 0

                    # if
                    second_row = frame0[y1:y2,x1:x2]
                    try:
                        img, plate = ocr_server1.main(second_row)
                    except:
                        break
                    try:
                        pass
                        #print(plate)
                    except:
                        pass
                    if 'plate' not in track.dict:
                        track.dict['plate'] = deque([''], maxlen=30)
                        track.dict['count_plate'] = 1
                        if len(plate) > 3 and len(plate) < 6:
                            track.dict['plate'].append(plate)
                    elif track.dict['count_plate'] < self.average_count:
                        if len(plate) > 3 and len(plate) < 6:
                            #plate = plate[:9]
                            #plate = plate.replace('Y','U')
                            track.dict['plate'].append(plate)
                            track.dict['count_plate'] += 1
                    #texts = [plate]
                    if track.dict['count_plate'] >= self.average_count:
                        true_plate = Counter(track.dict['plate']).most_common(1)[0][0]
                        drawing.putTexts(frame, [true_plate], x1, y1, size=1, thick=1, color=(255,255,255))
                        # track.dict['count_plate'] = average_count - 5
                # print(plate)
                #........
                # First Row
                x1,y1,x2,y2 = track.attr['xyxy']
                if plate_ratio < 2.5:
                    y1, y2 = int(y1 - h*.05), int(y1 + h*.47) #.53
                    mask_y1 = int(y1 + h*.48)
                    mask_y2 = int(y1 + h*.1)
                    mask_right = int(x2 - w*.15)
                    frame_top[mask_y1:y2, x1:x2] = 0
                    frame_top[y1:mask_y2, x1:x2] = 0
                    frame_top[y1:y2, mask_right:x2] = 0
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                try:
                    img2, plate2 = ocr_server2.main(frame_top[y1:y2,x1:x2])
                except:
                    break
                try:
                    pass
                    #print(plate2)
                except:
                   pass
                if 'plate2' not in track.dict:
                    track.dict['plate2'] = deque([''], maxlen=4)
                    track.dict['plate2'].append(plate2)
                    track.dict['count_plate2'] = 1
                elif track.dict['count_plate2'] < self.average_count:
                    if len(plate2) > 2 and plate2[0].upper() == 'T':
                        track.dict['plate2'].append(plate2)
                        track.dict['count_plate2'] += 1
                #if len(plate2) >= 8:
                #    plate2 = plate2[:9]
                #    #plate = plate.replace('Y','U')
                if track.dict['count_plate2'] >= self.average_count:
                    true_plate2 = Counter(track.dict['plate2']).most_common(1)[0][0]
                    drawing.putTexts(frame, [true_plate2], x1, y1, size=1, thick=1, color=(255,255,255))
                try:
                    # print(counter)
                    if (len(true_plate2) > 2):
                        if plate_ratio < 2.5 and (len(true_plate) > 3):
                            lp_number = true_plate2 + true_plate
                        else:
                            lp_number = true_plate2
                        #print("REAL NUMBER", lp_number)
                except:
                    pass

                self.send_alert(frame, lp_number, track)
            else:
                pass

        self.outstream.write(frame)

