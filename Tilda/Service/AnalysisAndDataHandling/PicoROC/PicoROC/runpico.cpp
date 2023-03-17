#include "runpico.h"

//Strange stuff for calback?///////////////////////////////////////////////////////////
template <typename T>
struct Callback;
template <typename Ret, typename... Params>
struct Callback<Ret(Params...)>
{
    template <typename... Args>
    static Ret callback(Args... args) { return func(args...); }
    static std::function<Ret(Params...)> func;
};
// Initialize the static member.
template <typename Ret, typename... Params>
std::function<Ret(Params...)> Callback<Ret(Params...)>::func;

void RunPico::CallBackBlock(	int16_t handle,PICO_STATUS status, void * pParameter)
{
   if (status != PICO_CANCELLED)
   {
       g_ready = true;
   }
}
//END Strange stuff for calback?///////////////////////////////////////////////////////////

RunPico::RunPico(PicoSettings &picoSettings, bool charge, QObject *parent) : QThread(parent),settings(picoSettings)
{

    RunPico::charge     =   charge;
    buffer              =   new FlatBuffer(settings,charge);
    std::cout<<"FlatBuffer Created"<<std::endl;

    processPico = new ProcessPico(buffer);
    ipcout = new ipcClient(&startAcq,&mutexAcq,0);
    ipcout->start();
    uint32_t        *nMaxSamples        =   new uint32_t;
    float           *timeIntervalNano   =   new float;

    if (charge==false)
    {
        handle=settings.handle_atom;
        ps6000OpenUnit (handle, picoSettings.serial_number1);
    }
    else
    {
        handle=settings.handle_ion;
        ps6000OpenUnit (handle, picoSettings.serial_number2);
    }



    std::cout<<"SetChannel: "<<ps6000SetChannel        (*handle,PS6000_CHANNEL_A,true ,PS6000_AC,PS6000_2V,0.,PS6000_BW_FULL)<<std::endl;
    std::cout<<"SetChannel: "<<ps6000SetChannel        (*handle,PS6000_CHANNEL_B,true ,PS6000_AC,PS6000_50MV,0.,PS6000_BW_FULL)<<std::endl;
    std::cout<<"SetChannel: "<<ps6000SetChannel        (*handle,PS6000_CHANNEL_C,true ,PS6000_DC_1M,PS6000_1V,-1.0,PS6000_BW_FULL)<<std::endl;
    std::cout<<"SetChannel: "<<ps6000SetChannel        (*handle,PS6000_CHANNEL_D,false,PS6000_AC,PS6000_5V,0.,PS6000_BW_FULL)<<std::endl;
    std::cout<<"SetTrigger: " << ps6000SetSimpleTrigger  (*handle,1,PS6000_CHANNEL_A,settings.trigAtoms,PS6000_FALLING,0,0)<<std::endl;;
    std::cout<<"Mem Segm : " << ps6000MemorySegments    (*handle,250000,nMaxSamples)<<"  MaxSamples "<<*nMaxSamples<<std::endl;
    std::cout<<"Num Cap : "  <<ps6000SetNoOfCaptures   (*handle,250000)<<std::endl;
    std::cout<<"Timebase : " <<ps6000GetTimebase2      (*handle,settings.timebase,settings.samples, timeIntervalNano,0, nMaxSamples, 0)<<"  MaxSamples "<<*nMaxSamples<<std::endl;
    std::cout<<"Timebase  :  "<<settings.getSampleTime()*settings.samples<<std::endl;
    float freq = 20000000;

    std::cout<<"Siggen  "<<ps6000SetSigGenBuiltInV2(*handle,0,2000000,PS6000_SINE,freq,freq,0,0,PS6000_UP,PS6000_ES_OFF,0,0,PS6000_SIGGEN_RISING,PS6000_SIGGEN_SOFT_TRIG,0)<<std::endl;
    buffer->setPicoBuffer   (*handle);

}

bool RunPico::closeProcessandIPCThreads()
{
    ipcout->runThread=false;
    std::cout<<ipcout->runThread<<std::endl;
    ipcout->memoryUpdate.release(1);
    ipcout->exit();
    while(!ipcout->isFinished())
    {
        QThread::msleep(10);
    }
    if(ipcout != nullptr)
    {
        delete ipcout;
        ipcout = nullptr;
    }

    processPico->runThread=false;
    processPico->mutexProc.lock();
    processPico->startProc.wakeAll();
    processPico->mutexProc.unlock();
    processPico->exit();
    while(!processPico->isFinished())
    {
        QThread::msleep(10);
    }
    if(processPico != nullptr)
    {
        delete processPico;
        processPico = nullptr;
    }
    return true;
}

void RunPico::run()
{

    int32_t         * timeIndisposedMs  =new int32_t;
    uint32_t        *nCaptures          =new uint32_t;
    uint32_t        *samples            =new uint32_t;
    int16_t         * overflow          =new int16_t[buffer->nCaptures*2];

    Callback<void(int16_t,uint32_t,void*)>::func=std::bind(&RunPico::CallBackBlock,this,std::placeholders::_1,std::placeholders::_2,std::placeholders::_3 );
    void (*c_func)(int16_t,uint32_t,void*) = static_cast<decltype(c_func)>(Callback<void(int16_t,uint32_t,void*)>::callback);
    processPico->start();
    int i=1 ;

    while(runThread)
    {

        RunPico::mutexAcq.lock     ();
        RunPico::startAcq.wait  (&mutexAcq);
        RunPico::mutexAcq.unlock   ();
        if (runThread==false)
        {
            break;
        }

        std::cout<<std::endl<<"run "<<i << " " << settings.delay << " " << settings.measTime << std::endl;
        i++;

        QThread::msleep         (settings.delay);

        //std::cout<<"Trig Gen: "<<ps6000SigGenSoftwareControl(*handle, 1)<<std::endl;
        //QThread::usleep         (10000);


        std::cout<<"Run Block: "<<ps6000RunBlock          (*handle,(settings.samples/3),settings.samples-settings.samples/3,settings.timebase,0,timeIndisposedMs,0,c_func,nullptr)<<std::endl;
        std::cout<<"Trig Gen: "<<ps6000SigGenSoftwareControl(*handle, 1)<<std::endl;

        QThread::msleep         (settings.measTime);

        std::cout<<"Stop : "<<ps6000Stop              (*handle)<<std::endl;

        std::cout<<"GetNoOfCaptures " << ps6000GetNoOfCaptures   (*handle,nCaptures ) << std::endl;

        std::cout<<"Captures:  "<<*nCaptures<<std::endl;
        if(*nCaptures > 0) {
            std::cout<<"Get trig info :"<<ps6000GetTriggerInfoBulk(*handle,buffer->triggers,0,*nCaptures-1)<<std::endl;
        }
        auto start = std::chrono::high_resolution_clock::now();

        *samples=settings.samples;
        ps6000GetValuesBulk     (*handle, samples,0,*nCaptures-1,1,PS6000_RATIO_MODE_NONE,overflow);

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        std::cout<<"Transfering data took "<< duration.count()/1000<<"milli s"<<std::endl;

        processPico->mutexProc.lock();
        processPico->procFin=true;
        processPico->mutexProc.unlock();

        processPico->mutexCopy.lock();
        {buffer->copyFlatBuffer  (*nCaptures);}
        processPico->mutexCopy.unlock();

        processPico->mutexProc.lock();
        processPico->startProc.wakeAll();
        processPico->mutexProc.unlock();

        std::cout<<"run: copied buffer "<< buffer->ion << std::endl;

    }
    delete  timeIndisposedMs    ;
    delete  nCaptures           ;
    delete  samples             ;
    delete[]overflow            ;
    runThread=false;

}
RunPico::~RunPico()
{
    /*
    if(processPico != nullptr) {
        delete processPico;
        processPico = nullptr;
    }
    if(ipcout != nullptr) {
        delete ipcout;
        ipcout = nullptr;
    }
    */
}

