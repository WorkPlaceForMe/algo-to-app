import os
import cv2

import use.drawing as drawing
import use.point_in_poly as point_in_poly
from algo_helper.old_files.sociald import SocialDistancing
from algo_helper.nms import filter_overlap
from algo_helper.old_files.mask import Mask


MYSQL_DB = os.environ.get('MYSQL_DB')
MYSQL_IP = os.environ.get('MYSQL_IP')

class CrowdCounter:
    def __init__(self, timer, outstream, mysql_helper, algos_dict, elastic):
        self.outstream = outstream
        mysql_fields = [
           ['id','varchar(45)'],
           ['time',"datetime"],
           ['cam_name','varchar(45)'],
           ['cam_id','varchar(45)'],
           ['id_account','varchar(45)'],
           ['id_branch','varchar(45)']]
        self.mysql_helper = mysql_helper
        self.mysql_helper.add_table('social', mysql_fields)
        self.timer = timer
        self.attr = algos_dict
        self.last_sent = self.timer.now_t
        self.es = elastic
        self.sociald = SocialDistancing(timer)
        self.mask = Mask()
        self.repeater = Repeater(5)
        
    def send_es(self, index, date, peopleCount):
        data = {}
        data['description'] = f'{peopleCount} people detected at {date} at {self.attr["camera_name"]}'
        #data['filename'] = imgName
        data['time'] = date
        data['cam_name'] = self.attr['camera_name']
        data['algo'] = self.attr['algo_name']
        data['algo_id'] = self.attr['algo_id']
        self.es.index(index=index, body=data)
        
    def filterDets(self, personDets):
        dets_out = []
        for det in personDets:
            x1, y1, x2, y2 = det['xyxy']
            x, y, w, h = det['xywh']
            if w > 100:
                continue
            if det['conf'] < (0.01 * self.attr['atributes'][0]['conf']):
                continue
            if (x2 -x1)/(y2-y1) > 0.8 or (y2-y1)>(0.5*720):
                continue

            if self.attr['rois'] is None or point_in_poly.point_in_poly(x, y, self.attr['rois']):
                dets_out.append(det)
        return dets_out
        
    def run(self, frame, personDets, yoloTracks):
        personDets = self.filterDets(personDets)
        personDets = self.repeater.run(personDets)
        if (self.attr['rois'] is not None):
            drawing.draw_rois(frame, self.attr['rois'], (255,0,0))
        for track in yoloTracks:
            x1, y1, x2, y2 = track.attr['xyxy']
            x, y, w, h = track.attr['xywh']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        peopleCount = len(yoloTracks)
        drawing.putTexts2(frame, [f"Count: {peopleCount}"], 650, 120, size=1, thick=2, color=(0,0,0))
        
        # social distancing
        frame = self.sociald.run(frame, personDets)

        # mask
        frame = self.mask.run(frame)
        self.outstream.write(frame)
        
        date = self.timer.now.strftime('%Y-%m-%d %H:%M:%S')
        mysql_values = [track.id, date, self.attr['camera_name'], self.attr['camera_id'],
                self.attr['id_account'], self.attr['id_branch']]       
        if self.timer.now_t -self.last_sent > 15:
            self.mysql_helper.insert_fast('social', mysql_values)
            self.send_es('gmtc_searcher', date, peopleCount)
            self.last_sent = self.timer.now_t
        
class Repeater:
    def __init__(self, repeat):
        self.repeat = repeat
        self.dets = []

    def update(self):
        dets = []
        for det in self.dets:
            det['count'] += 1
            if det['count'] <= self.repeat:
                dets.append(det)
        self.dets = dets

    def addNew(self, dets):
        for det in dets:
            det['count'] = 0
            self.dets.append(det)

    def run(self, dets):
        self.update()
        self.addNew(dets)
        dets_out = filter_overlap(self.dets, .5)
        return dets_out
