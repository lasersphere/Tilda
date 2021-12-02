#ifndef PICOROCDLL_GLOBAL_H
#define PICOROCDLL_GLOBAL_H

#include <QtCore/qglobal.h>

extern "C"
{
 __declspec(dllexport) void attachSharedMemory();
 __declspec(dllexport) void startScan(int step, double voltage, int scan, char* filename);
 __declspec(dllexport) void stepScan(int step, double voltage, int scan);
 __declspec(dllexport) void stopScan();
 __declspec(dllexport) void detachSharedMemory();

}

#endif // PICOROCDLL_GLOBAL_H
