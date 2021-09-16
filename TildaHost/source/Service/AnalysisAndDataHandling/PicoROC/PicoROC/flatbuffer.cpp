#include "flatbuffer.h"
#include <chrono>
FlatBuffer::FlatBuffer(PicoSettings &picoSettings,bool charge): settings(picoSettings)
{
    ion =charge;

    flatbufferP = new int16_t[nCaptures*3*settings.samples];
    flatbufferA = new int16_t[nCaptures*3*settings.samples];
    std::cout<< "Buffer "<<flatbufferP[0]<<std::endl;
    for (uint32_t i=0;i<nCaptures*3*settings.samples;i++)
    {
        flatbufferP[i]=0;
        flatbufferA[i]=0;
    }

    triggers    = new PS6000_TRIGGER_INFO[nCaptures];
    vEvents2    = new event2[nCaptures];



}

void FlatBuffer::copyFlatBuffer(uint32_t events)
{
    nEvent = events;
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Events " << nEvent << std::endl;
    if(nEvent > 0 ) {
        memcpy_s(flatbufferP,nEvent*3*settings.samples*sizeof (int16_t), flatbufferA,nEvent*3*settings.samples*sizeof (int16_t));
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout <<"memcpy " << duration.count()<<std::endl;


}

PICO_STATUS FlatBuffer::setPicoBuffer(int16_t handle)
{
    PICO_STATUS status;
    int samples=settings.samples;
    for (uint32_t capt = 0; capt < nCaptures; capt++)
    {
        status=ps6000SetDataBufferBulk (handle,PS6000_CHANNEL_A,&(flatbufferA[capt*3*samples]),samples,capt,PS6000_RATIO_MODE_NONE);
        //std::cout<<"Buffer status:""" << status<<std::endl;
        status=ps6000SetDataBufferBulk (handle,PS6000_CHANNEL_B,&(flatbufferA[capt*3*samples+samples]),samples,capt,PS6000_RATIO_MODE_NONE);
        //std::cout<<"Buffer status:""" << status<<std::endl;
        status=ps6000SetDataBufferBulk (handle,PS6000_CHANNEL_C,&(flatbufferA[capt*3*samples+2*samples]),samples,capt,PS6000_RATIO_MODE_NONE);
    }
    std::cout<<"Buffer status:""" << status<<std::endl;
    return status;
}


void FlatBuffer::deleteFlatBuffer()
{
    delete[] flatbufferP;
    delete[] flatbufferA;
    delete[] vEvents2;
}

FlatBuffer::~FlatBuffer()
{
    FlatBuffer::deleteFlatBuffer();
}
