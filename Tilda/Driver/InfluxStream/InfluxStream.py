"""
Created on 2023-03

@author: Tim Lellinger

Module Description: Receive data from an InfluxDB database
"""

import time
import Tilda.Driver.DataLoggerBase
from Tilda.Application.Importer import InfluxConfig

from influxdb import InfluxDBClient
import requests
requests.packages.urllib3.disable_warnings()#this disables the unverified SSL warning


class InfluxStream(Tilda.Driver.DataLoggerBase.DataLoggerBase):

    def __init__(self, infl_cfg=InfluxConfig.Influx_CFG):
        super().__init__('sql')
        self.infl_cfg = infl_cfg

    def db_query(self, strrequest):
        dbClient = InfluxDBClient(host=self.infl_cfg['host'],
                                  port=self.infl_cfg['port'],
                                  username=self.infl_cfg['username'],
                                  password=self.infl_cfg['password'],
                                  database=self.infl_cfg['database'],
                                  ssl=True,
                                  # verify_ssl=False,
                                  )
        print(strrequest)
        result = dbClient.query(strrequest,epoch="ns")
        dbClient.close()
        return result


    def get_channels_from_db(self):
        if self.infl_cfg['useinflux']:
            result = self.db_query('SHOW FIELD KEYS ON "'+self.infl_cfg["database"]+'"')
            channelsafields = []
            for series in result.raw["series"]:
                chname = series["name"]
                fields = series["values"]
                for field in fields:
                    if field[1] == 'float':
                        channelsafields.append(self.makestringXMLcompatible(chname + ":" + field[0]))
            return channelsafields
        else:
            return [self.makestringXMLcompatible("nodbdev:value")]

    def setupChtimes(self, pre_dur_post_str):
        if pre_dur_post_str == "duringScan":
            self.ch_time = {ch: time.time() for ch in self.log.keys()}
        else:  # for pre/post scan it is allowed to specify a time tolerance
            self.ch_time = {ch: int(time.time() - self.infl_cfg['notolderthan_s']) for ch in self.log.keys()}

    def write_run_to_db(self, unix_time, xml_file):
        pass

    def update_run_status_in_db(self, status):
        pass

    def conclude_run_in_db(self, unix_time_stop, status):
        pass

    def aquireChannel(self, chname, lastaqu):
        data = []
        lastaquns = int(lastaqu*1E9)#influx runs in integer ns instead of float seconds
        if self.infl_cfg["useinflux"]:
            measname, field = self.undoXMLcompatible(chname).split(":")
            query = 'SELECT "' + field + '" FROM "' + measname + '" WHERE time > ' + str(lastaquns)
            result = self.db_query(query)
            if len(result.raw["series"]) > 0:
                data = result.raw["series"][0]["values"]
        else:
            data = [[time.time_ns(), 0.12345]]  # dummy mode
        return data

    def makestringXMLcompatible(self,inp):
        return inp.replace("/","...").replace(":",".")

    def undoXMLcompatible(self,inp):
        return inp.replace("...","/").replace(".",":")