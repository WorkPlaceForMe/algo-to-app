

from mysql2a import Mysql
  
mysql_args = {
    "ip":'ubuntu_db_1',
    "user":'graymatics',
    "pwd":'graymatics',
    "db":'multi_tenant',
    "table":"violence",
    "column": [
        ['time','datetime'],
        ['violent','tinyint(1)'],
        ['camera_name','varchar(40)'],
        ['cam_id','varchar(40)'],
        ['id_branch','varchar(40)'],
        ['id_account','varchar(40)'],
        ['id','varchar(40)']
        ]
    }
mysql = Mysql(mysql_args)

mysql_args = {
    "ip":'ubuntu_db_1',
    "user":'graymatics',
    "pwd":'graymatics',
    "db":'multi_tenant',
    "table":"crowd_count",
    "column": [
        ['time','datetime'],
        ['number_of_ppl','int(11)'],
        ['camera_name','varchar(40)'],
        ['cam_id','varchar(40)'],
        ['id_branch','varchar(40)'],
        ['id_account','varchar(40)'],
        ]
    }
mysql = Mysql(mysql_args)
mysql_args = {
    "ip":'ubuntu_db_1',
    "user":'graymatics',
    "pwd":'graymatics',
    "db":'multi_tenant',
    "table":"vehicles",
    "column": [
        ['time','datetime'],
        ['type','varchar(40)'],
        ['plate_number','varchar(40)'],
        ['camera_name','varchar(40)'],
        ['cam_id','varchar(40)'],
        ['id_branch','varchar(40)'],
        ['id_account','varchar(40)'],
        ]
    }
mysql = Mysql(mysql_args)
mysql_args = {
    "ip":'ubuntu_db_1',
    "user":'graymatics',
    "pwd":'graymatics',
    "db":'multi_tenant',
    "table":"fr",
    "column": [
        ['time','datetime'],
        ['name','varchar(40)'],
        ['camera_name','varchar(40)'],
        ['cam_id','varchar(40)'],
        ['id_branch','varchar(40)'],
        ['id_account','varchar(40)'],
        ]
    }
mysql = Mysql(mysql_args)

mysql_args = {
    "ip":'ubuntu_db_1',
    "user":'graymatics',
    "pwd":'graymatics',
    "db":'multi_tenant',
    "table":"aod",
    "column": [
        ['time','datetime'],
        ['camera_name','varchar(40)'],
        ['cam_id','varchar(40)'],
        ['id_branch','varchar(40)'],
        ['id_account','varchar(40)'],
        ['id','varchar(40)']
        ]
    }
mysql = Mysql(mysql_args)