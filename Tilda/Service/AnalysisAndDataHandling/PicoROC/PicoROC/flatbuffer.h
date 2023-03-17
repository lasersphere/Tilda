#ifndef FLATBUFFER_H
#define FLATBUFFER_H
#include <QCoreApplication>
#include <QVector>
#include <iostream>
#include "picosettings.h"

class FlatBuffer
{
public:
    FlatBuffer(PicoSettings &picoSettings, bool charge);
    ~FlatBuffer();
    int16_t* flatbufferA;
    int16_t* flatbufferP;
    void copyFlatBuffer(uint32_t events);
    struct  alignas(8)            Event {uint8_t energyInner; uint8_t energyOuter; uint8_t time;};
    struct alignas(8)             event2 {int8_t energyInner; int8_t energyOuter; int8_t time;};
    void deleteFlatBuffer();
    PS6000_TRIGGER_INFO *triggers;
    PICO_STATUS         setPicoBuffer(int16_t handle);
    bool                ion;
    QVector<Event>      vEvents;
    event2* vEvents2;

    uint64_t            nEvent;
    int                 validEvents;
    void                clearFlatBuffer();
    PicoSettings        &settings;
    const uint64_t      nCaptures               =   250000;


};

#endif // FLATBUFFER_H
