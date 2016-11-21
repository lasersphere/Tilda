/*
 * Generated with the FPGA Interface C API Generator 14.0.0
 * for NI-RIO 14.0.0 or later.
 */

#ifndef __NiFpga_TRS_DAF_204_h__
#define __NiFpga_TRS_DAF_204_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1400
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_TRS_DAF_204_Bitfile;
 */
#define NiFpga_TRS_DAF_204_Bitfile "NiFpga_TRS_DAF_204.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_TRS_DAF_204_Signature = "F581751C62A1DC80D72103FACEB7FCDE";

typedef enum
{
   NiFpga_TRS_DAF_204_IndicatorBool_DACQuWriteTimeout = 0x8126,
   NiFpga_TRS_DAF_204_IndicatorBool_MCSQuWriteTimeout = 0x812A,
} NiFpga_TRS_DAF_204_IndicatorBool;

typedef enum
{
   NiFpga_TRS_DAF_204_IndicatorI8_MCSerrorcount = 0x8122,
} NiFpga_TRS_DAF_204_IndicatorI8;

typedef enum
{
   NiFpga_TRS_DAF_204_IndicatorU8_postAccOffsetVoltState = 0x811A,
} NiFpga_TRS_DAF_204_IndicatorU8;

typedef enum
{
   NiFpga_TRS_DAF_204_IndicatorU16_MCSstate = 0x8132,
   NiFpga_TRS_DAF_204_IndicatorU16_measVoltState = 0x812E,
   NiFpga_TRS_DAF_204_IndicatorU16_seqState = 0x8156,
} NiFpga_TRS_DAF_204_IndicatorU16;

typedef enum
{
   NiFpga_TRS_DAF_204_ControlBool_VoltOrScaler = 0x810E,
   NiFpga_TRS_DAF_204_ControlBool_abort = 0x814E,
   NiFpga_TRS_DAF_204_ControlBool_halt = 0x814A,
   NiFpga_TRS_DAF_204_ControlBool_hostConfirmsHzOffsetIsSet = 0x8146,
   NiFpga_TRS_DAF_204_ControlBool_invertScan = 0x8172,
   NiFpga_TRS_DAF_204_ControlBool_timedOutWhileHandshake = 0x815A,
} NiFpga_TRS_DAF_204_ControlBool;

typedef enum
{
   NiFpga_TRS_DAF_204_ControlU8_postAccOffsetVoltControl = 0x816E,
   NiFpga_TRS_DAF_204_ControlU8_selectTrigger = 0x8142,
   NiFpga_TRS_DAF_204_ControlU8_triggerEdge = 0x8116,
} NiFpga_TRS_DAF_204_ControlU8;

typedef enum
{
   NiFpga_TRS_DAF_204_ControlU16_cmdByHost = 0x8152,
   NiFpga_TRS_DAF_204_ControlU16_triggerTypes = 0x811E,
   NiFpga_TRS_DAF_204_ControlU16_waitAfterReset25nsTicks = 0x8166,
   NiFpga_TRS_DAF_204_ControlU16_waitForKepco25nsTicks = 0x816A,
} NiFpga_TRS_DAF_204_ControlU16;

typedef enum
{
   NiFpga_TRS_DAF_204_ControlI32_dacStartRegister18Bit = 0x817C,
   NiFpga_TRS_DAF_204_ControlI32_dacStepSize18Bit = 0x8180,
   NiFpga_TRS_DAF_204_ControlI32_measVoltPulseLength25ns = 0x8160,
   NiFpga_TRS_DAF_204_ControlI32_measVoltTimeout10ns = 0x815C,
   NiFpga_TRS_DAF_204_ControlI32_nOfBunches = 0x8134,
   NiFpga_TRS_DAF_204_ControlI32_nOfScans = 0x8174,
   NiFpga_TRS_DAF_204_ControlI32_nOfSteps = 0x8178,
} NiFpga_TRS_DAF_204_ControlI32;

typedef enum
{
   NiFpga_TRS_DAF_204_ControlU32_dac0VRegister = 0x8110,
   NiFpga_TRS_DAF_204_ControlU32_nOfBins = 0x8138,
   NiFpga_TRS_DAF_204_ControlU32_trigDelay10ns = 0x813C,
} NiFpga_TRS_DAF_204_ControlU32;

typedef enum
{
   NiFpga_TRS_DAF_204_TargetToHostFifoU32_transferToHost = 0,
} NiFpga_TRS_DAF_204_TargetToHostFifoU32;

#endif
