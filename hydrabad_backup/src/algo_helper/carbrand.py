import cv2
import numpy as np
import zmq
import ast

context = zmq.Context()

class CarBrand:
    def __init__(self):
        self.carcrop = np.zeros((100,100,3), dtype=np.uint8)
        self.yoloSocket = context.socket(zmq.REQ)
        self.yoloSocket.connect("tcp://carbrand_server:5601")

    def run(self, frame, vehicleTracks):
        frames_to_average = 5
        for track in vehicleTracks:
            if track.attr['cls'] != 'car':
                continue
            if 'predicted_brands' not in track.dict:
                track.dict['predicted_brands'] = {}
            #if track.miss > 0:
            #    continue

            x1,y1,x2,y2 = track.attr['xyxy']
            x,y,w,h = track.attr['xywh']

            carcrop = frame[y1:y2, x1:x2]
            carbrand = ''
            carshape = ''
            final_carbrand = ''
            if 'frame_count' not in track.dict:
                track.dict['frame_count'] = 1
            if carcrop.shape[0] >= 60 and carcrop.shape[1] >= 60:
                if 'predicted' not in track.tag and track.dict['frame_count'] < frames_to_average:
                    carcrop = np.ascontiguousarray(carcrop, dtype=np.uint8)
                    self.yoloSocket.send_multipart([str(carcrop.shape).encode(), carcrop])
                    pred_res = self.yoloSocket.recv()
                    pred_res = ast.literal_eval(pred_res.decode())
                    
                    carbrand = pred_res[2][0][0]
                    brand_scores = pred_res[2][0][1]

                    carbrand = carbrand.encode('utf-8')

                    if carbrand not in track.dict['predicted_brands']:
                        track.dict['predicted_brands'][carbrand] = 0
                    track.dict['predicted_brands'][carbrand] += round(brand_scores/frames_to_average,2)
                    track.dict['frame_count'] += 1

                if track.dict['frame_count'] == frames_to_average:
                    track.tag.add('predicted')
                    final_carbrand = max(track.dict['predicted_brands'], key=track.dict['predicted_brands'].get)
                    track.dict['carbrand'] = final_carbrand.decode('utf-8')
                    track.dict['score'] = track.dict['predicted_brands'][final_carbrand]

            #if 'carbrand' in track.dict and track.dict['score'] > 0.3:
            #    if "tracked" not in track.tag:
            #        track.tag.add("tracked")
            #    return track.dict['carbrand']
            #else:
            #    return None
