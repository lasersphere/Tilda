#include "picorocdll.h"


struct IPCDATA
{
    int     step=0;
    double  voltage=0;
    int     scan=0;
    bool    endOfFile=false;
    bool    inRun = false;
    char    fileName[1024]="noFile";
};

IPCDATA     ipcData;
QUdpSocket  socket;
QByteArray  data;

extern "C"
{

void attachSharedMemory()
{
    socket.bind(QHostAddress("192.168.0.2"),1234);
}

void startScan(int step, double voltage, int scan, char* filename)
{
    ipcData.step=step;
    ipcData.voltage=voltage;
    ipcData.scan=scan;
    ipcData.endOfFile=false;
    ipcData.inRun=true;
    strncpy(ipcData.fileName, filename, strlen(filename));
    writeData();
}

void stepScan(int step, double voltage, int scan)
{
    ipcData.step=step;
    ipcData.voltage=voltage;
    ipcData.scan=scan;
    ipcData.inRun=true;
    writeData();
}

void stopScan()
{
    ipcData.inRun=false;
    ipcData.endOfFile=true;
    writeData();
}

void detachSharedMemory()
{
    socket.disconnectFromHost();
}
}

void writeData()
{
    data.clear();
    QDataStream out (&data, QIODevice::WriteOnly);
    out.device()->seek(0); //rewind

    struct IPCDATA
    {
        int step;
        double voltage;
        int scan;
        bool endOfFile;
        char fileName[1024];
    };

    out << (int) ipcData.step
        << (double) ipcData.voltage
        << (int) ipcData.scan
        << (bool) ipcData.endOfFile
        << (bool) ipcData.inRun
        << (QString) QString(ipcData.fileName);

    socket.writeDatagram(data, QHostAddress("192.168.0.255"), 1234);
}


