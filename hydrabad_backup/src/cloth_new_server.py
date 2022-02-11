import os
import numpy as np
import cv2
import zmq
import ast
from collections import deque, Counter

import sys
sys.path.append('/home/src/cloth/api/model')
sys.path.append('/home/src/cloth/api/demo')
from net import Backbone_FC
import torch
import torch.utils.data as data
from classify import classify, ImageList

class Clothing:
    def __init__(self):
        #context = zmq.Context()
        #self.pubSocket = context.socket(zmq.REQ)
        ##self.pubSocket.connect(f'tcp://test_clothing:5612')
        #self.pubSocket.connect("tcp://127.0.0.1:5631")
        #self.outstream = outstream
        #self.castSize = (640, 480)
        #mysql_fields = [
        #    ["track_id", "varchar(45)"],
        #    ["time","datetime"],
        #    ["cam_id","varchar(45)"],
        #    ["cam_name","varchar(45)"],
        #    ["id_account","varchar(45)"],
        #    ["id_branch", "varchar(45)"],
        #    ["sleeve_length", "varchar(45)"],
        #    ["top_colour", "varchar(45)"],
        #    ["bottom_length", "varchar(45)"],
        #    ["bottom_colour", "varchar(45)"]
        #    ]
        #self.mysql_helper = mysql_helper
        #self.mysql_helper.add_table('clothing', mysql_fields)
        #self.timer = timer
        #self.attr = algos_dict
        #self.average_len = 10
        #self.es = elastic
        self.test_model = 'cloth/api/model/checkpoint054.pth'
        self.logger = open('results.txt', 'w')
        #self.now_t = time.time()
        #self.count = 0
        #self.time_diff = 0
        #self.pbar = tqdm(total=1)
        self.model = Backbone_FC(132)
        self.model.load_state_dict(torch.load(self.test_model))
        self.model = self.model.cuda()

    def run(self, img_list):
        data_loader = data.DataLoader(ImageList(img_list), batch_size=128, num_workers=1, shuffle=False)
        result = classify(self.model, data_loader, self.logger)
        return result


clothing = Clothing()
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.connect('tcp://127.0.0.1:5632')

while True:
    xyxy_b, shape_b, frame_b = socket.recv_multipart()
    xyxy = ast.literal_eval(xyxy_b.decode())
    shape = ast.literal_eval(shape_b.decode())
    frame = np.frombuffer(frame_b, dtype=np.uint8).reshape(shape)

    img_list = [frame[y1:y2,x1:x2] for (x1,y1,x2,y2) in xyxy]
    dets = clothing.run(img_list)
    socket.send(str(dets).encode())

