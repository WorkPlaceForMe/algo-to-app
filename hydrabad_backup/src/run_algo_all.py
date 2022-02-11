
import os
import multiprocessing as mp
from use.mysql2 import Mysql
import run_algo
import time
import json

MYSQL_DB = os.environ.get('MYSQL_DB')
MYSQL_IP = os.environ.get('MYSQL_IP')
ALGO_DICT = {'19':'violence', '2':'loiter', '12':'crowd', '32':'clothing', '26':'vehicle_type', '0':'fr', '16':'aod', '13':'anpr'}
HTTP_OUT_IP = os.environ.get('SERVER_IP')
HTTP_OUT_PORT = 8091
STREAM_IP = 'broadcast'
STREAM_PORT = 8090

mysql = Mysql({"ip":MYSQL_IP, "user":'graymatics', "pwd":'graymatics', "db":MYSQL_DB, "table":""})

def get_rtsp_dict():
    def edit_path(rtsp):
        if rtsp[:4] in {'rtsp', 'http'}:
            return rtsp
        else:
            #return '/home/videos/' + rtsp[27:]
            return rtsp.replace('/usr/src/app/resources/', '/home/videos/').replace('/home/nodejs/app/resources/', '/home/videos/')

    cmd = 'select id, name, rtsp_in from cameras'
    things = mysql.run_fetch(cmd)
    rtsp_dict = {}
    for camera_id, name, rtsp_in in things:
        rtsp_dict[camera_id] = [name, edit_path(rtsp_in)]
    return rtsp_dict

def reset_stream_url(num, id_, camera_id, algo_id):
    #mysql.run(f'update relations set http_out=concat("http://{HTTP_OUT_IP}:{PORT}/stream",id,".mjpeg")')
    mysql.run(f'update relations set http_out="http://{HTTP_OUT_IP}:{HTTP_OUT_PORT}/stream{num}.mjpeg" where id="{id_}" and camera_id="{camera_id}" and algo_id="{algo_id}"')

def get_cam_dict(rtsp_dict):
    stream_num = 0
    cmd = 'select id,camera_id,algo_id, roi_id, atributes, id_account, id_branch, stream, createdAt, updatedAt, http_out from relations'
    things = mysql.run_fetch(cmd)
    cam_dict = {}
    for id_, camera_id, algo_id, roi_id, atributes, id_account, id_branch, stream, createdAt, updatedAt, http_out in things:
        if algo_id not in ALGO_DICT:
            print(f'algo_id: {algo_id} not present')
            continue
        algo_name = ALGO_DICT[algo_id]
        reset_stream_url(stream_num, id_, camera_id, algo_id)
        if camera_id not in cam_dict:
            cam_dict[camera_id] = {}
            cam_dict[camera_id]['camera_id'] = camera_id
            cam_dict[camera_id]['rtsp_in'] = rtsp_dict[camera_id][1]
            cam_dict[camera_id]['algos'] = {}
        cam_dict[camera_id]['algos'][algo_name] = {}
        if (roi_id is not None):
            cam_dict[camera_id]['algos'][algo_name]['rois'] = json.loads(roi_id)
        else:
            cam_dict[camera_id]['algos'][algo_name]['rois'] = roi_id
        cam_dict[camera_id]['algos'][algo_name]['atributes'] = json.loads(atributes)
        cam_dict[camera_id]['algos'][algo_name]['algo_name'] = algo_name
        cam_dict[camera_id]['algos'][algo_name]['algo_id'] = algo_id
        cam_dict[camera_id]['algos'][algo_name]['camera_id'] = camera_id
        cam_dict[camera_id]['algos'][algo_name]['camera_name'] = rtsp_dict[camera_id][0]
        cam_dict[camera_id]['algos'][algo_name]['id_account'] = id_account
        cam_dict[camera_id]['algos'][algo_name]['id_branch'] = id_branch
        #cam_dict[camera_id]['algos'][algo_name]['http_out'] = http_out
        cam_dict[camera_id]['algos'][algo_name]['stream_in'] = f"http://{STREAM_IP}:{STREAM_PORT}/feed{stream_num}.ffm"
        #cam_dict[camera_id]['algos'][algo_name]['roi_id'] = rois
        #cam_dict[camera_id]['algos'][algo_name]['atributes'] = id_
        stream_num += 1
    return cam_dict


if __name__ == '__main__':
    rtsp_dict = get_rtsp_dict()
    cam_dict = get_cam_dict(rtsp_dict)
    mysql.close()
    print('--algos setting--')
    for camera_id, values in cam_dict.items():
        print(values)
        print(f'starting camera_id {camera_id} ...\n')
        p = mp.Process(target=run_algo.main, args=(values,))
        p.daemon = True
        p.start()

        