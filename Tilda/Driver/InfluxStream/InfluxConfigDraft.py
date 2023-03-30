"""Module description: Draft Config for the Influx class."""

Influx_CFG = {
    'useinflux' : False, #if set to False, use a dummy mode. set to true it will actually try to connect to a DB
    'host': '188.184.30.238',
    'port': 8080,
    'username': 'admin',
    'password': 'pw',
    'database': 'mydatabase',
    'notolderthan_s' : 10 #this value specifies a time offset which is allowed for Pre/Post scan measurements in seconds. Within this window TILDA does not need to wait for a new value.
}