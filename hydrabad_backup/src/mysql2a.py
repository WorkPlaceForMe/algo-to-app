
import MySQLdb
import traceback
import time

class Mysql:
    def __init__(self, args):
        self.ip = args["ip"]
        self.user = args["user"]
        self.passwd = args["pwd"]
        self.db = args['db']
        self.table = args["table"]
        self.values = {}

        self.connect()
        if 'column' in args:
            self.fields, self.field_and_type = self.get_columns_info(args['column'])
            self.cursor.execute(f'create database if not exists {self.db}')
            self.database.commit()
            self.cursor.execute(f'drop table if exists {self.db}.{self.table}')
            self.cursor.execute(f'create table if not exists {self.db}.{self.table} ({self.field_and_type})')
        else:
            print('read-only')

    def connect(self):
        self.database = MySQLdb.connect(self.ip, self.user, self.passwd, self.db)
        self.cursor = self.database.cursor()

    def get_columns_info(self, columns):
        fields = []
        field_and_type = []

        for field, _type in columns:
            fields.append(field)
            field_and_type.append(f'{field} {_type}')

        fields = ','.join(fields)
        field_and_type = (','.join(field_and_type))
        return fields, field_and_type

    # -------------------------------------------------------

    def add_table(self, table):
        self.values[table] = []

    def set_table(self, table):
        self.table = table

    def insert(self, values): # 0.05s for each insert, 99% of time used by commit
        values = [f'"{v}"' for v in values]
        values = ','.join(values)
        #cmd = f'insert into {self.db}.{self.table} ({self.fields}) values ({values})'
        cmd = f'insert into {self.db}.{self.table} values ({values})'
        self.run(cmd)

    def insert_fast(self, table, values): # may cause error if used together with alter table
        values = [f'"{v}"' for v in values]
        values = ','.join(values)
        self.values[table].append(f'({values})')

    def commit_insert(self, table):
        if self.values[table]:
            values = ','.join(self.values[table])
            cmd = f'insert into {self.db}.{table} values {values}'
            self.run(cmd)
            self.values[table] = []

    def fetch(self, fields, wherein=None):
        fields = ','.join(fields)
        cmd = f'select {fields} from {self.db}.{self.table}'
        if wherein:
            field, values = wherein
            values = [f'"{v}"' for v in values]
            values = ','.join(values)
            cmd += f' where {field} in ({values})'
        #print(cmd)
        self.run(cmd)

        result = self.cursor.fetchall()
        return result

    def run_fetch(self, cmd):
        try:
            self.cursor.execute(cmd)
            self.database.commit()
            result = self.cursor.fetchall()
            return result
        except Exception:
            #print(traceback.format_exc())
            print(f'ERROR:{cmd}')
            time.sleep(1)
            self.close()
            self.connect()
            self.cursor.execute(cmd)
            self.database.commit()
            return result

    def run(self, cmd):
        try:
            self.cursor.execute(cmd)
            self.database.commit()
        except Exception:
            print(traceback.format_exc())
            print(cmd)
            time.sleep(1)
            self.close()
            self.connect()
            self.cursor.execute(cmd)
            self.database.commit()

    def close(self):
        self.cursor.close()
        self.database.close()

if __name__ == '__main__':
    mysql_args = {
        "ip":'127.0.0.1',
        "user":'graymatics',
        "pwd":'graymatics',
        "db":'test0',
        "table":"test1",
        #"column": [
        #    #['c1','INT NOT NULL AUTO_INCREMENT, primary key(c1)'],
        #    ['c1','INT NOT NULL'],
        #    ['c2','varchar(40)']
        #    ]
        }

    mysql = Mysql(mysql_args)
    #for i in range(100):
    #    mysql.insert_fast([12345, 'abcde'])
    #mysql.commit_insert()
    t0 = time.time()
    for i in range(1):
        print(mysql.fetch(['c2,c1']))
    print(time.time() -t0)
