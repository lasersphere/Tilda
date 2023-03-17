#ifndef PROCESSPICO_H
#define PROCESSPICO_H
#include <QCoreApplication>
#include <QString>
#include <QtCore>
#include <QObject>
#include "picosettings.h"
#include "flatbuffer.h"
#include <ps6000Api.h>
#ifndef PICO_STATUS
#include <PicoStatus.h>
#endif

class ProcessPico: public QThread
{
    Q_OBJECT


public:

    QWaitCondition startProc;
    QMutex mutexProc;
    QMutex mutexCopy;
    QMutex mutexDataOut;


    bool procFin = false;
    bool runThread=true;

    ProcessPico(FlatBuffer *fb, QObject *parent = NULL);
    ~ProcessPico();

    void run() override;

    FlatBuffer *fb;
    QVector<double> v_validEvents;

signals:
    void dataReady();


private:




};

#endif // PROCESSPICO_H
