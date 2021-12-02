#ifndef IPCCLIENT_H
#define IPCCLIENT_H

#include <QThread>
#include <QCoreApplication>
#include <QString>
#include <QtCore>
#include <QObject>
#include <QSharedMemory>
#include <QSystemSemaphore>
#include <iostream>

struct IPCDATA
{
    int     step;
    double  voltage;
    int     scan;
    bool    endOfFile;
    char fileName[1024];
};

class ipcClient: public QThread
{
    Q_OBJECT

public:

    ipcClient(QWaitCondition* wc, QMutex* mu, QObject *parent = NULL);
    ~ipcClient();
    void run() override;
    bool runThread=true;

    QSystemSemaphore memoryUpdate;

private:
    QWaitCondition* startAcq;
    QMutex* mutexAcq;
    QSharedMemory* memory;
    IPCDATA ipcData;

};

#endif // IPCCLIENT_H
