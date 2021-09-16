#include <QCoreApplication>
#include <QString>
#include <QSharedMemory>
#include <iostream>
#include <QSystemSemaphore>
#include "PicoROCdll_global.h"

int main(int argc, char *argv[])
{
   // QCoreApplication a(argc, argv);
   char fileName[] = "myFileABC";

    attachSharedMemory();
    startScan(1,1.0,1,fileName);
    //std::cout<<"share size"<<memory.size() <<std::endl;

    stepScan(10,11,12);

    //std::cout<<"share size"<<memory.size() <<std::endl;
    std::string any;
    std::cin>>any;
    detachSharedMemory();
    //return a.exec();
}
