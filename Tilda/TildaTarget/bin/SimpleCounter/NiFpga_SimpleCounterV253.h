/*
 * Generated with the FPGA Interface C API Generator 14.0.0
 * for NI-RIO 14.0.0 or later.
 */

#ifndef __NiFpga_SimpleCounterV253_h__
#define __NiFpga_SimpleCounterV253_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1400
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_SimpleCounterV253_Bitfile;
 */
#define NiFpga_SimpleCounterV253_Bitfile "NiFpga_SimpleCounterV253.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_SimpleCounterV253_Signature = "7EB244FCD5A1C780261ED48E2A1E370C";

typedef enum
{
   NiFpga_SimpleCounterV253_IndicatorU8_postAccOffsetVoltState = 0x813A,
} NiFpga_SimpleCounterV253_IndicatorU8;

typedef enum
{
   NiFpga_SimpleCounterV253_IndicatorU16_DacState = 0x8126,
} NiFpga_SimpleCounterV253_IndicatorU16;

typedef enum
{
   NiFpga_SimpleCounterV253_IndicatorU32_actDACRegister = 0x8128,
} NiFpga_SimpleCounterV253_IndicatorU32;

typedef enum
{
   NiFpga_SimpleCounterV253_ControlBool_softwareTrigger = 0x8116,
} NiFpga_SimpleCounterV253_ControlBool;

typedef enum
{
   NiFpga_SimpleCounterV253_ControlU8_postAccOffsetVoltControl = 0x8136,
   NiFpga_SimpleCounterV253_ControlU8_selectTrigger = 0x8122,
   NiFpga_SimpleCounterV253_ControlU8_triggerEdge = 0x811A,
} NiFpga_SimpleCounterV253_ControlU8;

typedef enum
{
   NiFpga_SimpleCounterV253_ControlU16_DacStateCmdByHost = 0x8132,
   NiFpga_SimpleCounterV253_ControlU16_triggerTypes = 0x810E,
} NiFpga_SimpleCounterV253_ControlU16;

typedef enum
{
   NiFpga_SimpleCounterV253_ControlU32_dwellTime10ns = 0x8110,
   NiFpga_SimpleCounterV253_ControlU32_setDACRegister = 0x812C,
   NiFpga_SimpleCounterV253_ControlU32_trigDelay10ns = 0x811C,
} NiFpga_SimpleCounterV253_ControlU32;

typedef enum
{
   NiFpga_SimpleCounterV253_TargetToHostFifoU32_transferToHost = 0,
} NiFpga_SimpleCounterV253_TargetToHostFifoU32;

#endif