/*
 * Generated with the FPGA Interface C API Generator 14.0.0
 * for NI-RIO 14.0.0 or later.
 */

#ifndef __NiFpga_SimpleCounterV100_h__
#define __NiFpga_SimpleCounterV100_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1400
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_SimpleCounterV100_Bitfile;
 */
#define NiFpga_SimpleCounterV100_Bitfile "NiFpga_SimpleCounterV100.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_SimpleCounterV100_Signature = "1927BCDAF87BC3C5B46E16D2F712C497";

typedef enum
{
   NiFpga_SimpleCounterV100_IndicatorU8_postAccOffsetVoltState = 0x8122,
} NiFpga_SimpleCounterV100_IndicatorU8;

typedef enum
{
   NiFpga_SimpleCounterV100_IndicatorU16_DacState = 0x8112,
} NiFpga_SimpleCounterV100_IndicatorU16;

typedef enum
{
   NiFpga_SimpleCounterV100_IndicatorU32_actDACRegister = 0x8114,
} NiFpga_SimpleCounterV100_IndicatorU32;

typedef enum
{
   NiFpga_SimpleCounterV100_ControlU8_postAccOffsetVoltControl = 0x811E,
} NiFpga_SimpleCounterV100_ControlU8;

typedef enum
{
   NiFpga_SimpleCounterV100_ControlU16_DacStateCmdByHost = 0x811A,
} NiFpga_SimpleCounterV100_ControlU16;

typedef enum
{
   NiFpga_SimpleCounterV100_ControlU32_setDACRegister = 0x810C,
} NiFpga_SimpleCounterV100_ControlU32;

typedef enum
{
   NiFpga_SimpleCounterV100_TargetToHostFifoU32_transferToHost = 0,
} NiFpga_SimpleCounterV100_TargetToHostFifoU32;

#endif
