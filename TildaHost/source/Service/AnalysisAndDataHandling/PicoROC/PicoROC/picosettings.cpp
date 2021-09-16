#include "picosettings.h"

PicoSettings::PicoSettings()
{
    PicoSettings::delay         =   10;
    PicoSettings::measTime      =   300;
    PicoSettings::timebase      =   4;
    PicoSettings::samples       =   30;
    PicoSettings::trigAtoms     =   0x1000;
    PicoSettings::trigIons      =   -10;
    PicoSettings::calAtomIn     =   -1;
    PicoSettings::calAtomOut    =   -1;
    PicoSettings::calIonIn      =   -1;
    PicoSettings::calIonOut     =   -1;
    PicoSettings::tmin          =   0;
    PicoSettings::tmax          =   10;
    PicoSettings::eMin          =   0;
    PicoSettings::eMax          =   10;

    handle_atom= new int16_t;
    handle_ion = new int16_t;
}

PicoSettings::~PicoSettings()
{
    delete handle_atom;
    delete handle_ion;
}

double PicoSettings::getSampleTime()
{
    if (timebase<5)
    {
        return pow(2,timebase)/5000000000;
    }
    else
    {
        return (timebase-4.0)/156250000;
    }
}

void PicoSettings::savePicoSettings(QFile file)
{
    file.open(QIODevice::WriteOnly | QIODevice::Text);
    QJsonObject json
    {
        {"delay"        ,   delay       },
        {"measTime"     ,   measTime    },
        {"timebase"     ,   timebase    },
        {"samples"      ,   samples     },
        {"trigAtoms"    ,   trigAtoms   },
        {"trigIons"     ,   trigIons    },
        {"calAtomIn"    ,   calAtomIn   },
        {"calAtomOut"   ,   calAtomOut  },
        {"calIonIn"     ,   calIonIn    },
        {"calIonOut"    ,   calIonOut   },
        {"tmin"         ,   tmin        },
        {"tmax"         ,   tmax        },
        {"eMin"         ,   eMin        },
        {"eMax"         ,   eMax        }
    };
    QJsonDocument document(json);
    file.write(document.toJson(QJsonDocument::JsonFormat::Compact));
    file.close();
}

void PicoSettings::loadPicoSettings(QFile file)
{
    file.open(QIODevice::ReadOnly | QIODevice::Text);
    QString         val     =   file.readAll();
    QJsonDocument   doc     =   QJsonDocument::fromJson(val.toUtf8());
    QJsonObject     json    =   doc.object();
    file.close();

    delay       =   json.take   ("delay")       .toInt();
    measTime    =   json.take   ("measTime")    .toInt();
    timebase    =   json.take   ("timebase")    .toInt();
    samples     =   json.take   ("samples")     .toInt();
    trigAtoms   =   json.take   ("trigAtoms")   .toInt();
    trigIons    =   json.take   ("trigIons")    .toInt();
    calAtomIn   =   json.take   ("calAtomIn")   .toDouble();
    calAtomOut  =   json.take   ("calAtomOut")  .toDouble();
    calIonIn    =   json.take   ("calIonIn")    .toDouble();
    calIonOut   =   json.take   ("calIonOut")   .toDouble();
    tmin        =   json.take   ("tmin")        .toInt();
    tmax        =   json.take   ("tmax")        .toInt();
    eMin        =   json.take   ("eMin")        .toDouble();
    eMax        =   json.take   ("eMax")        .toDouble();
}

