/*
 * Generated with the FPGA Interface C API Generator 14.0.0
 * for NI-RIO 14.0.0 or later.
 */

#ifndef __NiFpga_SimpleCounterV200_h__
#define __NiFpga_SimpleCounterV200_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1400
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_SimpleCounterV200_Bitfile;
 */
#define NiFpga_SimpleCounterV200_Bitfile "NiFpga_SimpleCounterV200.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_SimpleCounterV200_Signature = "2BE2BD956E75A0E9F14F6B1A6761B2FB";

typedef enum
{
   NiFpga_SimpleCounterV200_IndicatorU8_postAccOffsetVoltState = 0x8122,
} NiFpga_SimpleCounterV200_IndicatorU8;

typedef enum
{
   NiFpga_SimpleCounterV200_IndicatorU16_DacState = 0x8112,
} NiFpga_SimpleCounterV200_IndicatorU16;

typedef enum
{
   NiFpga_SimpleCounterV200_IndicatorU32_actDACRegister = 0x8114,
} NiFpga_SimpleCounterV200_IndicatorU32;

typedef enum
{
   NiFpga_SimpleCounterV200_ControlU8_postAccOffsetVoltControl = 0x811E,
} NiFpga_SimpleCounterV200_ControlU8;

typedef enum
{
   NiFpga_SimpleCounterV200_ControlU16_DacStateCmdByHost = 0x811A,
} NiFpga_SimpleCounterV200_ControlU16;

typedef enum
{
   NiFpga_SimpleCounterV200_ControlU32_setDACRegister = 0x810C,
} NiFpga_SimpleCounterV200_ControlU32;

typedef enum
{
   NiFpga_SimpleCounterV200_TargetToHostFifoU32_transferToHost = 0,
} NiFpga_SimpleCounterV200_TargetToHostFifoU32;

#endif
