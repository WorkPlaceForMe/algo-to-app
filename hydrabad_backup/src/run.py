

import pymysql
pymysql.install_as_MySQLdb()
import os
import subprocess
from datetime import datetime, timedelta
import time
from use.mysql2 import Mysql

MYSQL_DB = os.environ.get('MYSQL_DB')
MYSQL_IP = os.environ.get('MYSQL_IP')
CHECK_DURATION = 5


class Update:
    def __init__(self):
        self.last_update = datetime.now() - timedelta(weeks=100)

    def need_restart(self):
        cmd = 'select id,camera_id,algo_id, roi_id, atributes, id_account, id_branch, stream, createdAt, updatedAt, http_out from relations'
        things = mysql.run_fetch(cmd)
        for id_, camera_id, algo_id, roi_id, atributes, id_account, id_branch, stream, createdAt, updatedAt, http_out in things:
            if updatedAt > self.last_update:
                self.last_update = updatedAt
                return True
        return False


mysql = Mysql({"ip":MYSQL_IP, "user":'graymatics', "pwd":'graymatics', "db":MYSQL_DB, "table":""})
update = Update()
last_check = 0

while True:
    if update.need_restart():
        print('restarting ...')
        subprocess.run('pkill -f run_algo_all.py -9', shell=True)
        subprocess.Popen('python3 run_algo_all.py', shell=True)
    else:
        pass
        #print('no restarting ...')
    time.sleep(CHECK_DURATION)