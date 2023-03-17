/*
 * Generated with the FPGA Interface C API Generator 14.0.0
 * for NI-RIO 14.0.0 or later.
 */

#ifndef __NiFpga_NiFpga_SimpleCounterV254_h__
#define __NiFpga_NiFpga_SimpleCounterV254_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1400
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_NiFpga_SimpleCounterV254_Bitfile;
 */
#define NiFpga_NiFpga_SimpleCounterV254_Bitfile "NiFpga_NiFpga_SimpleCounterV254.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_NiFpga_SimpleCounterV254_Signature = "7EB244FCD5A1C780261ED48E2A1E370C";

typedef enum
{
   NiFpga_NiFpga_SimpleCounterV254_IndicatorU8_postAccOffsetVoltState = 0x813A,
} NiFpga_NiFpga_SimpleCounterV254_IndicatorU8;

typedef enum
{
   NiFpga_NiFpga_SimpleCounterV254_IndicatorU16_DacState = 0x8126,
} NiFpga_NiFpga_SimpleCounterV254_IndicatorU16;

typedef enum
{
   NiFpga_NiFpga_SimpleCounterV254_IndicatorU32_actDACRegister = 0x8128,
} NiFpga_NiFpga_SimpleCounterV254_IndicatorU32;

typedef enum
{
   NiFpga_NiFpga_SimpleCounterV254_ControlBool_softwareTrigger = 0x8116,
} NiFpga_NiFpga_SimpleCounterV254_ControlBool;

typedef enum
{
   NiFpga_NiFpga_SimpleCounterV254_ControlU8_postAccOffsetVoltControl = 0x8136,
   NiFpga_NiFpga_SimpleCounterV254_ControlU8_selectTrigger = 0x8122,
   NiFpga_NiFpga_SimpleCounterV254_ControlU8_triggerEdge = 0x811A,
} NiFpga_NiFpga_SimpleCounterV254_ControlU8;

typedef enum
{
   NiFpga_NiFpga_SimpleCounterV254_ControlU16_DacStateCmdByHost = 0x8132,
   NiFpga_NiFpga_SimpleCounterV254_ControlU16_triggerTypes = 0x810E,
} NiFpga_NiFpga_SimpleCounterV254_ControlU16;

typedef enum
{
   NiFpga_NiFpga_SimpleCounterV254_ControlU32_dwellTime10ns = 0x8110,
   NiFpga_NiFpga_SimpleCounterV254_ControlU32_setDACRegister = 0x812C,
   NiFpga_NiFpga_SimpleCounterV254_ControlU32_trigDelay10ns = 0x811C,
} NiFpga_NiFpga_SimpleCounterV254_ControlU32;

typedef enum
{
   NiFpga_NiFpga_SimpleCounterV254_TargetToHostFifoU32_transferToHost = 0,
} NiFpga_NiFpga_SimpleCounterV254_TargetToHostFifoU32;

#endif
