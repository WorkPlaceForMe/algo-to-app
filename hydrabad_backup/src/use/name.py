#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 12:39:48 2021

@author: fibish
"""

import os
import numpy as np
import cv2
import glob
import ujson
import zmq
import ast
import time

class UnknownData():
    def __init__(self, dir0='/home/latest5/data/faceData/unknownFace/'):
        self.dir = dir0
        self.id_base = str(int(time.time()))
        self.id_now = 1
        self.features = []
        self.ids = []
        self.load()

    def load(self):
        self.features = []
        featurePaths = glob.glob(f'{self.dir}*.npy')
        for path in featurePaths:
            feature = np.load(path)[0] # change from (1,512) to (512)
            id_ = os.path.basename(path).split('.')[0]
            self.features.append(feature)
            self.ids.append(id_)
        self.features = np.array(self.features, dtype=np.float32)

    def addFeature(self, feature, faceImg):
        if len(self.features) == 0:
            self.features = feature
        else:
            self.features = np.concatenate((self.features, feature))
        id_ = int(self.id_base + str(self.id_now))
        self.ids.append(id_)
        self.id_now += 1
        # save feature
        np.save(f'{self.dir}{id_}.npy', feature)
        cv2.imwrite(f'{self.dir}{id_}.jpg', faceImg)
        return id_

    def getName(self, feature, faceImg, thres=.3):
        # get similarity
        feature = np.array([feature], dtype=np.float32)
        #print(self.features)
        if len(self.features) == 0: # empty database
            faceid = self.addFeature(feature, faceImg)
            result = faceid
        else:
            # get index of most similar feature
            similar = np.einsum('ij,kj->ik', feature, self.features) # [n,m]
            max_score = np.amax(similar, axis=1) # n
            if max_score > thres: # match is found
                max_ind = np.where(similar==max_score[:,None])[1][0] # n
                faceid = self.ids[max_ind]
                result = faceid
            else: # no match
                faceid = self.addFeature(feature, faceImg)
                result = faceid
        return result

class UnknownData2():
    def __init__(self, dir0='../../data/unknownFace/'):
        self.attrs = {}
        self.dir = dir0

        # read data
        try:
            self.ids = list(np.load(self.dir + 'ids.npy'))
            self.features = np.load(self.dir + 'features.npy')
            with open('{}nextId.txt'.format(self.dir), "r") as f:
                self.nextId = int(f.readline().strip()) + 1
        except OSError:
            self.features = []
            self.ids = []
            self.nextId = 1

    def save(self):
        np.save(self.dir + 'ids.npy', np.array(self.ids))
        np.save(self.dir + 'features.npy', self.features)
        with open('{}nextId.txt'.format(self.dir), "w") as f:
            f.write(str(self.nextId))

    def genNextId(self):
        self.ids.append(self.nextId)
        self.nextId += 1

    def matchWithKnown(self, featuresKnown, attrKnown, mysql1):
        if len(self.features) == 0:
            return 0
        self.attrs = {}
        for i in range(featuresKnown.shape[0]):
            feature = featuresKnown[None,i,:]
            similar = np.einsum('ij,kj->ik', feature, self.features) # [n,m]

            matchIds = np.where(similar>0.35)[1] # n
            if matchIds.any():
                # update self.attrs
                for id_ in matchIds:
                    #print('--------------------')
                    print(self.attrs)
                    #print(self.ids[id_])
                    #print(i)
                    #print(attrKnown)
                    #print(featuresKnown.shape)
                    self.attrs[self.ids[id_]] = attrKnown[i]

                    # update sql
                    for attr2 in ('age', 'gender', 'category', 'email', 'phone', 'img_path'):
                        mysql1.cursor.execute("update tam.test set {}='{}' where faceid='unknown_{}'".format(attr2,attrKnown[i][attr2], self.ids[id_]))
                    mysql1.cursor.execute("update tam.test set car_preference='{}' where faceid='unknown_{}'".format(attrKnown[i]['car'], self.ids[id_]))
                    mysql1.cursor.execute("update tam.test set name='{}' where faceid='unknown_{}'".format(attrKnown[i]['username'], self.ids[id_]))
                    mysql1.cursor.execute("update tam.test set faceid='{}' where faceid='unknown_{}'".format(attrKnown[i]['nameid'], self.ids[id_]))

    def getName(self, feature):
        # get similarity
        feature = np.array([feature])
        if len(self.features) == 0:
            self.features = feature
            self.genNextId()
        similar = np.einsum('ij,kj->ik', feature, self.features) # [n,m]

        # get index of most similar feature
        max_score = np.amax(similar, axis=1) # n
        if max_score > 0.3:
            max_ind = np.where(similar==max_score[:,None])[1] # n

            # return name, age, gender, from knownDatabase
            if self.ids[max_ind[0]] in self.attrs:
                res = [True, {k:[v,int(max_score*100)] for k,v in self.attrs[self.ids[max_ind[0]]].items()}]
            else:
                res = ['unknown_{}'.format(self.ids[max_ind[0]]), int(max_score*100)]
        else:
            res = ['unknown_{}'.format(self.nextId), 100]
            self.features = np.concatenate((self.features, feature))
            self.genNextId()
        return res

class KnownData():
    def __init__(self, dir0='/home/latest5/data/faceData/knownFace1/', port='tcp://localhost:5605'):
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
                img = cv2.resize(img, xy) # to improve: fix dimension of uploaded photo
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
        
    def getName(self, feature, thres=0.4):
        # get similarity
        feature = np.array([feature])
        if len(self.features) == 0:
            result = None
        else:
            similar = np.einsum('ij,kj->ik', feature, self.features) # [n,m]

            # get index of most similar feature
            max_score = np.amax(similar, axis=1) # n
            if max_score > thres:
                max_ind = np.where(similar==max_score[:,None])[1] # n
                #result = [True, {k:[v,int(max_score*100)] for k,v in self.attrs[max_ind[0]].items()}]
                #result = {k:[v,int(max_score*100)] for k,v in self.attrs[max_ind[0]].items()}['username']
                result = self.attrs[max_ind[0]]['username']
            else:
                result = None
        return result
