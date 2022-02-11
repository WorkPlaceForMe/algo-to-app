
import MySQLdb
from datetime import datetime

db = MySQLdb.connect('ubuntu_db_1', 'graymatics', 'graymatics', db="multi_tenant")
cursor = db.cursor()

# cameras
# cursor.execute('delete from cameras where id="cam1"')
# cursor.execute('delete from cameras where id="cam2"')
# cursor.execute('delete from cameras where id="cam3"')
# cursor.execute('delete from cameras where id="cam4"')
# cursor.execute('delete from cameras where id="cam5"')
# cursor.execute('delete from cameras where id="cam6"')
# cursor.execute('delete from cameras where id="cam7"')
cursor.execute('delete from cameras')
db.commit()

cmd = "insert into cameras (id, name, rtsp_in, cam_width, cam_height, createdAt, updatedAt) values ('cam1','camName','/home/videos/walk.mp4','1280','720', '{0}','{0}')".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
cursor.execute(cmd)


# relation
cursor.execute('delete from relations')
date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
idacc = idbranch = 'cc730777'

# crowd
camid = 'cam1'
rois = '[{"x":0,"y":360},{"x":1280,"y":360},{"x":1280,"y":720},{"x":0,"y":720}]'
attr = '[{"conf": 10, "save": true, "time": 0}]'
cmd = f"insert into relations (camera_id, algo_id, roi_id, atributes, id_account, id_branch, stream, createdAt, updatedAt) \
        values ('{camid}','50','{rois}','{attr}','{idacc}','{idbranch}', 0, '{date}','{date}')"
print(cmd)
cursor.execute(cmd)

# crowd
camid = 'cam1'
rois = '[{"x":0,"y":360},{"x":1280,"y":360},{"x":1280,"y":720},{"x":0,"y":720}]'
attr = '[{"conf": 10, "save": true, "time": 0}]'
cmd = f"insert into relations (camera_id, algo_id, roi_id, atributes, id_account, id_branch, stream, createdAt, updatedAt) \
        values ('{camid}','51','{rois}','{attr}','{idacc}','{idbranch}', 0, '{date}','{date}')"
print(cmd)
cursor.execute(cmd)

# crowd
camid = 'cam1'
rois = '[{"x":0,"y":360},{"x":1280,"y":360},{"x":1280,"y":720},{"x":0,"y":720}]'
attr = '[{"conf": 10, "save": true, "time": 0}]'
cmd = f"insert into relations (camera_id, algo_id, roi_id, atributes, id_account, id_branch, stream, createdAt, updatedAt) \
        values ('{camid}','52','{rois}','{attr}','{idacc}','{idbranch}', 0, '{date}','{date}')"
print(cmd)
cursor.execute(cmd)

db.commit()