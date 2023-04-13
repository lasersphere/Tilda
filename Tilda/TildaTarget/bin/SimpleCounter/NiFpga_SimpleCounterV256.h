/*
 * Generated with the FPGA Interface C API Generator 14.0.0
 * for NI-RIO 14.0.0 or later.
 */

#ifndef __NiFpga_SimpleCounterV256_h__
#define __NiFpga_SimpleCounterV256_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1400
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_SimpleCounterV256_Bitfile;
 */
#define NiFpga_SimpleCounterV256_Bitfile "NiFpga_SimpleCounterV256.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_SimpleCounterV256_Signature = "26A900D8A0DB783ECFE2C08779A02D4B";

typedef enum
{
   NiFpga_SimpleCounterV256_IndicatorU8_postAccOffsetVoltState = 0x813E,
} NiFpga_SimpleCounterV256_IndicatorU8;

typedef enum
{
   NiFpga_SimpleCounterV256_IndicatorU16_DacState = 0x812A,
} NiFpga_SimpleCounterV256_IndicatorU16;

typedef enum
{
   NiFpga_SimpleCounterV256_IndicatorU32_actDACRegister = 0x812C,
} NiFpga_SimpleCounterV256_IndicatorU32;

typedef enum
{
   NiFpga_SimpleCounterV256_ControlBool_StartSignal = 0x8112,
   NiFpga_SimpleCounterV256_ControlBool_softwareTrigger = 0x811A,
} NiFpga_SimpleCounterV256_ControlBool;

typedef enum
{
   NiFpga_SimpleCounterV256_ControlU8_postAccOffsetVoltControl = 0x813A,
   NiFpga_SimpleCounterV256_ControlU8_selectTrigger = 0x8126,
   NiFpga_SimpleCounterV256_ControlU8_triggerEdge = 0x811E,
} NiFpga_SimpleCounterV256_ControlU8;

typedef enum
{
   NiFpga_SimpleCounterV256_ControlU16_DacStateCmdByHost = 0x8136,
   NiFpga_SimpleCounterV256_ControlU16_triggerTypes = 0x810E,
} NiFpga_SimpleCounterV256_ControlU16;

typedef enum
{
   NiFpga_SimpleCounterV256_ControlU32_dwellTime10ns = 0x8114,
   NiFpga_SimpleCounterV256_ControlU32_setDACRegister = 0x8130,
   NiFpga_SimpleCounterV256_ControlU32_trigDelay10ns = 0x8120,
} NiFpga_SimpleCounterV256_ControlU32;

typedef enum
{
   NiFpga_SimpleCounterV256_TargetToHostFifoU32_transferToHost = 0,
} NiFpga_SimpleCounterV256_TargetToHostFifoU32;

#endif
