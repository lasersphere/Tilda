/*
 * Generated with the FPGA Interface C API Generator 14.0.0
 * for NI-RIO 14.0.0 or later.
 */

#ifndef __NiFpga_SimpleCounterV252_h__
#define __NiFpga_SimpleCounterV252_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1400
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_SimpleCounterV252_Bitfile;
 */
#define NiFpga_SimpleCounterV252_Bitfile "NiFpga_SimpleCounterV252.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_SimpleCounterV252_Signature = "DC5A7BCCA258451DB55BBF8398B5BCD0";

typedef enum
{
   NiFpga_SimpleCounterV252_IndicatorU8_postAccOffsetVoltState = 0x8122,
} NiFpga_SimpleCounterV252_IndicatorU8;

typedef enum
{
   NiFpga_SimpleCounterV252_IndicatorU16_DacState = 0x8112,
} NiFpga_SimpleCounterV252_IndicatorU16;

typedef enum
{
   NiFpga_SimpleCounterV252_IndicatorU32_actDACRegister = 0x8114,
} NiFpga_SimpleCounterV252_IndicatorU32;

typedef enum
{
   NiFpga_SimpleCounterV252_ControlU8_postAccOffsetVoltControl = 0x811E,
} NiFpga_SimpleCounterV252_ControlU8;

typedef enum
{
   NiFpga_SimpleCounterV252_ControlU16_DacStateCmdByHost = 0x811A,
} NiFpga_SimpleCounterV252_ControlU16;

typedef enum
{
   NiFpga_SimpleCounterV252_ControlU32_setDACRegister = 0x810C,
} NiFpga_SimpleCounterV252_ControlU32;

typedef enum
{
   NiFpga_SimpleCounterV252_TargetToHostFifoU32_transferToHost = 0,
} NiFpga_SimpleCounterV252_TargetToHostFifoU32;

#endif
