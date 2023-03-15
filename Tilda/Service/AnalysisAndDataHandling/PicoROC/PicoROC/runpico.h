#ifndef RUNPICO_H
#define RUNPICO_H
#include <QCoreApplication>
#include <QString>
#include <QtCore>
#include "picosettings.h"
#include "flatbuffer.h"
#include <ps6000Api.h>
#ifndef PICO_STATUS
#include <PicoStatus.h>
#include <processpico.h>
#include "ipcclient.h"
//#include "callbackfn.h"
#endif

class RunPico: public QThread
{
public:
    inline static QWaitCondition startAcq;
    inline static QMutex mutexAcq;
    bool runThread = true;

    bool closeProcessandIPCThreads();


    bool charge;
    bool g_ready;
    int16_t * handle= nullptr;




    void CallBackBlock(	int16_t handle,PICO_STATUS status,void * pParameter);

    RunPico(PicoSettings &picoSettings, bool charge, QObject *parent = NULL);
    ~RunPico();

    void run() override;

    FlatBuffer *buffer;
    PicoSettings &settings;
    ProcessPico  *processPico;
    inline static ipcClient* ipcout;

private:




};

#endif // RUNPICO_H
