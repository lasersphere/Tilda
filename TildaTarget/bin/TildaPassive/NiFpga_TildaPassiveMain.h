/*
 * Generated with the FPGA Interface C API Generator 14.0.0
 * for NI-RIO 14.0.0 or later.
 */

#ifndef __NiFpga_TildaPassiveMain_h__
#define __NiFpga_TildaPassiveMain_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1400
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_TildaPassiveMain_Bitfile;
 */
#define NiFpga_TildaPassiveMain_Bitfile "NiFpga_TildaPassiveMain.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_TildaPassiveMain_Signature = "BE610634C7571F6D016C6F4F66749B41";

typedef enum
{
   NiFpga_TildaPassiveMain_IndicatorU16_TildaPassiveStateInd = 0x8112,
} NiFpga_TildaPassiveMain_IndicatorU16;

typedef enum
{
   NiFpga_TildaPassiveMain_ControlU16_TildaPassiveStateCtrl = 0x810E,
} NiFpga_TildaPassiveMain_ControlU16;

typedef enum
{
   NiFpga_TildaPassiveMain_ControlU32_delay_10ns_ticks = 0x8114,
   NiFpga_TildaPassiveMain_ControlU32_nOfBins = 0x8118,
} NiFpga_TildaPassiveMain_ControlU32;

typedef enum
{
   NiFpga_TildaPassiveMain_TargetToHostFifoU32_transferToHost = 0,
} NiFpga_TildaPassiveMain_TargetToHostFifoU32;

#endif
