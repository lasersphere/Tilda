/*
 * Generated with the FPGA Interface C API Generator 14.0.0
 * for NI-RIO 14.0.0 or later.
 */

#ifndef __NiFpga_SimpleCounter_7841_h__
#define __NiFpga_SimpleCounter_7841_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1400
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_SimpleCounter_7841_Bitfile;
 */
#define NiFpga_SimpleCounter_7841_Bitfile "NiFpga_SimpleCounter_7841.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_SimpleCounter_7841_Signature = "E87F746B4D75FAC83D98D9825FB513AE";

typedef enum
{
   NiFpga_SimpleCounter_7841_IndicatorU8_postAccOffsetVoltState = 0x8122,
} NiFpga_SimpleCounter_7841_IndicatorU8;

typedef enum
{
   NiFpga_SimpleCounter_7841_IndicatorU16_DacState = 0x8112,
} NiFpga_SimpleCounter_7841_IndicatorU16;

typedef enum
{
   NiFpga_SimpleCounter_7841_IndicatorU32_actDACRegister = 0x8114,
} NiFpga_SimpleCounter_7841_IndicatorU32;

typedef enum
{
   NiFpga_SimpleCounter_7841_ControlU8_postAccOffsetVoltControl = 0x811E,
} NiFpga_SimpleCounter_7841_ControlU8;

typedef enum
{
   NiFpga_SimpleCounter_7841_ControlU16_DacStateCmdByHost = 0x811A,
} NiFpga_SimpleCounter_7841_ControlU16;

typedef enum
{
   NiFpga_SimpleCounter_7841_ControlU32_setDACRegister = 0x810C,
} NiFpga_SimpleCounter_7841_ControlU32;

typedef enum
{
   NiFpga_SimpleCounter_7841_TargetToHostFifoU32_transferToHost = 0,
} NiFpga_SimpleCounter_7841_TargetToHostFifoU32;

#endif
