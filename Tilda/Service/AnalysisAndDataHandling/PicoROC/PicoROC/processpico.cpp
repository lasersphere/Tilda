#include "processpico.h"



ProcessPico::ProcessPico(FlatBuffer *fb, QObject *parent) : QThread(parent), fb(fb)
{


}



void ProcessPico::run()
{
    QFile output("output.dat");
    output.open(QIODevice::ReadWrite );
    QTextStream datout(&output);

    auto start = std::chrono::high_resolution_clock::now();
    double calIn;
    double calOut;
    if (fb->ion==true)
    {
        calIn   =   fb->settings.calIonIn;
        calOut  =   fb->settings.calIonOut;
    }
    else
    {
        calIn   =   fb->settings.calAtomIn;
        calOut  =   fb->settings.calAtomOut;
    }

    double sampletime   =   fb->settings.getSampleTime();
    int samples         =   fb->settings.samples;

    double factorIn     =   sampletime*calIn*fb->settings.mV/(1000*PS6000_MAX_VALUE);
    double factorOut    =   sampletime*calOut*fb->settings.mV/(1000*PS6000_MAX_VALUE);
    while(runThread)
    {
        mutexProc.lock  ();
        startProc.wait  (&mutexProc);
        procFin             =               false;
        mutexProc.unlock();
        if (runThread==false)
        {
            break;
        }

        //mutexDataOut.lock();
        //dataReady=false;
        //mutexDataOut.lock();

        mutexCopy.lock();
        auto startProc      =   std::chrono::high_resolution_clock::now();


        int validEvents     =   0;
        fb->vEvents.resize(fb->nEvent);
        std::cout << "resize " << fb->nEvent << std::endl;

        FlatBuffer::Event event;
        for (uint32_t capture = 0; capture < fb->nEvent; capture++)
        {
            event.energyInner=0;
            event.energyOuter=0;
            if(procFin == true)
            {
                std::cout << "process: exit" << std::endl;
                break;
            }

             event.time  =   (fb->flatbufferP[capture*3*samples+2*samples]/256+128);
             uint8_t etime= (uint16_t)(fb->flatbufferP[capture*3*samples+2*samples]/256+128);

            //std::cout<<"Event time  "<< (uint16_t)(event.time) << " " <<
            //           fb->flatbufferP[capture*3*samples+2*samples] <<   " " << (uint16_t)etime << std::endl;

            for (int16_t sample = 0; sample < fb->settings.samples; sample++)
            {
                event.energyInner +=  (fb->flatbufferP[capture*3*samples+sample]);//*factorIn;
                event.energyOuter +=  (fb->flatbufferP[capture*3*samples+samples+sample]);//*factorOut;
            }
            datout<<  (event.energyInner/256)<<" , "<<event.energyOuter/256<<" , "<<etime<<"\n";

            event.energyInner = event.energyInner/256;
            event.energyOuter = event.energyOuter/256;


            fb->vEvents[capture] = event;
            if (
                    event.energyInner*factorIn+event.energyOuter*factorOut         >   fb->settings.eMin
                    &&  event.energyInner*factorIn+event.energyOuter*factorOut     <   fb->settings.eMax
                    &&  event.time                              >   fb->settings.tmin
                    &&  event.time                              <   fb->settings.tmax
                    )
            {
                validEvents++;
            }
            //to be used as an array index, add 128 to create a positive number
            event.energyInner = event.energyInner+ 128;
            event.energyOuter = event.energyOuter+ 128;


            fb->vEvents[capture] = event;

        }


        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        auto durationProc = std::chrono::duration_cast<std::chrono::milliseconds>(stop - startProc);

        std::cout <<"process: " <<v_validEvents.size()+1 << "  "<< duration.count() << ": " << durationProc.count() << std::endl;
        datout<<"\n"<<"\n"<<"Process :"<<v_validEvents.size()+1<<"\n"<<"\n";
        v_validEvents.push_back(validEvents);

        emit dataReady();

        mutexCopy.unlock();

    }
    output.close();
   //std::cout<<"file closed"<<std::endl;


}


ProcessPico::~ProcessPico()
{
    delete fb;
}




