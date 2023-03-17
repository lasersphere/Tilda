#include "ipcclient.h"
#include <cstdio>
ipcClient::ipcClient(QWaitCondition* wc, QMutex* mu, QObject *parent) :
    QThread(parent), startAcq(wc), mutexAcq(mu),  memoryUpdate("SemaphoreKeyForROCPicoscope", 0, QSystemSemaphore::Open)
{

    memory = new QSharedMemory("MyVerySafeKeyForROCPicoscope");
    if (!memory->attach())
    {
        std::cout << "Shared memory attach error" << std::endl;
        if(memory->create(sizeof(IPCDATA)) == false)
        {
            std::cout << "Shared memory create error: " << sizeof(IPCDATA) << " bytes" << std::endl;
        }
        else
        {
            std::cout << "Shared memory created!" << std::endl;
        }
    }
    else
    {
        std::cout << "Shared memory attached!" << std::endl;
    }

}

ipcClient::~ipcClient(){
    memory->detach();
    delete memory;
    std::cout << "Shared memory detached!" << std::endl;
}

void ipcClient::run()
{
   while(runThread==true)
   {
        memoryUpdate.acquire();
        if (runThread==false)
        {
            break;
        }
        memory->lock();

        IPCDATA* data = (IPCDATA*)memory->data();
        ipcData = *data;
        memory->unlock();

        mutexAcq->lock();
        startAcq->wakeAll();
        mutexAcq->unlock();

        std::cout<<"Step : "<<ipcData.step<<std::endl;

        std::cout<<"voltage : "<<ipcData.voltage<<std::endl;
        std::cout<<"scan : "<<ipcData.scan<<std::endl;
        std::cout<<"endOfFile : "<<ipcData.endOfFile<<std::endl;
        std::cout<<"fileName : "<<ipcData.fileName <<std::endl;
    }

}
