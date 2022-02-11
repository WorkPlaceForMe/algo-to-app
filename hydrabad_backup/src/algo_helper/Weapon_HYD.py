import cv2
import zmq
import use.drawing as drawing
import ast
from use.tracking9 import Tracker
from use.detect import formatPlate  

class Weapon:
    def __init__(self, timer, outstream, mysql_helper, algos_dict, elastic):
        context = zmq.Context()
        self.weaponSocket = context.socket(zmq.REQ)
        self.weaponSocket.connect("tcp://weapon2-server:5615")
        self.yoloTracker = Tracker((1280,720))
        self.confidence = 0.01
        self.screenXY = (1280, 720)
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
        
    def send_mysql(self, date, track, count):
        mysql_values = [date, count, self.attr['camera_name'], self.attr['camera_id'], self.attr['id_branch'], self.attr['id_account']]
        if self.timer.now_t -self.last_sent > 15:
            self.mysql_helper.insert_fast('crowd_count', mysql_values)
            self.send_es('gmtc_searcher', date, count)
            self.last_sent = self.timer.now_t

        
    def send_alert(self, frame, track, count):
        self.last_sent = self.timer.now_t
        date = self.timer.now.strftime('%Y-%m-%d %H:%M:%S')
        self.send_mysql(date, track,count)

        
    def run(self, frame, yoloDets, stream=True):
        self.weaponSocket.send(frame)
        message = self.weaponSocket.recv()
        weaponDets = ast.literal_eval(message.decode())
        weaponDets = formatPlate(weaponDets, None,0.06,(1280,720))
        
        for det in yoloDets:
            x1, y1, x2, y2 = det['xyxy']
            x, y, w, h = det['xywh']
            if w > 100:
                continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 1)
        peopleCount = len(yoloDets)
        drawing.putTexts2(frame, [f"Count: {peopleCount}"], 650, 120, size=1, thick=2, color=(0,0,0))
        
        #MySql
        date = self.timer.now.strftime('%Y-%m-%d %H:%M:%S')
        mysql_values = [date, peopleCount, self.attr['camera_name'], self.attr['camera_id'], self.attr['id_branch'], self.attr['id_account']]
           # if self.timer.now_t -self.last_sent > 15:
        self.mysql_helper.insert_fast('crowd_count', mysql_values)
        self.send_es('gmtc_searcher', date, peopleCount)
        self.last_sent = self.timer.now_t


        for i,track  in enumerate(weaponDets):
            x1,y1,x2,y2 = track['xyxy']
            x,y,w,h = track['xywh']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 3)
            drawing.putTexts2(frame, ['Weapon Detected!'], x1, y1, size=1, thick=2, color=(0,0,0))
            #self.send_alert(frame, track, peopleCount)

            #MySql
           # date = self.timer.now.strftime('%Y-%m-%d %H:%M:%S')
           # mysql_values = [date, peopleCount, self.attr['camera_name'], self.attr['camera_id'], self.attr['id_branch'], self.attr['id_account']]
           # if self.timer.now_t -self.last_sent > 15:
           #     self.mysql_helper.insert_fast('crowd_count', mysql_values)
           #     self.send_es('gmtc_searcher', date, peopleCount)
           #     self.last_sent = self.timer.now_t
        if stream:
           self.outstream.write(frame)
