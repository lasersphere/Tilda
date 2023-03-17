#ifndef PICOSETTINGS_H
#define PICOSETTINGS_H
#include <QCoreApplication>
#include <QFile>
#include <QJsonObject>
#include <QJsonDocument>

#include <ps6000Api.h>
#ifndef PICO_STATUS
#include <PicoStatus.h>
#endif

class PicoSettings
{
public:
                        PicoSettings();
                        ~PicoSettings();
    double              getSampleTime();
    void                savePicoSettings(QFile file);
    void                loadPicoSettings(QFile file);
    int16_t             delay;
    int16_t             measTime;
    int16_t             timebase;
    int16_t             samples;
    int16_t             trigAtoms;
    int16_t             trigIons;
    double              calAtomIn;
    double              calAtomOut;
    double              calIonIn;
    double              calIonOut;
    int                 tmin;
    int                 tmax;
    double              eMin;
    double              eMax;


    const uint16_t      rangeIndex   = 8;                                                  //5000 mV range
    const uint16_t      mV           = 5000;


    int16_t * handle_ion = nullptr;
    int16_t * handle_atom = nullptr;

    int8_t serial_number1[10] = "GW086/026";
    int8_t serial_number2[10] = "GW086/026";






};

#endif // PICOSETTINGS_H
