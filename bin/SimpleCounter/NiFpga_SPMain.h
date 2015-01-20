/*
 * Generated with the FPGA Interface C API Generator 14.0.00
 * for NI-RIO 14.0.0 or later.
 * Test
 */

#ifndef __NiFpga_SPMain_h__
#define __NiFpga_SPMain_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1400
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_SPMain_Bitfile;
 */
#define NiFpga_SPMain_Bitfile "D:\\Workspace\\Eclipse\\Tilda\\TildaTarget\\bin\\SimpleCounter\\NiFpga_SPMain.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_SPMain_Signature = "0FCF79DACC611C97C420E1D3B7260086";

typedef enum
{
   NiFpga_SPMain_IndicatorBool_Overflow = 0x810E,
} NiFpga_SPMain_IndicatorBool;

typedef enum
{
   NiFpga_SPMain_ControlBool_stop = 0x8112,
} NiFpga_SPMain_ControlBool;

typedef enum
{
   NiFpga_SPMain_TargetToHostFifoU32_SimpleCounterDMA = 0,
} NiFpga_SPMain_TargetToHostFifoU32;

#endif
