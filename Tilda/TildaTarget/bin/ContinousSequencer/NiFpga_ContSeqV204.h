/*
 * Generated with the FPGA Interface C API Generator 14.0.0
 * for NI-RIO 14.0.0 or later.
 */

#ifndef __NiFpga_ContSeqV204_h__
#define __NiFpga_ContSeqV204_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1400
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_ContSeqV204_Bitfile;
 */
#define NiFpga_ContSeqV204_Bitfile "NiFpga_ContSeqV204.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_ContSeqV204_Signature = "52BCD4A1D8B95585DAD347F0477C59EC";

typedef enum
{
   NiFpga_ContSeqV204_IndicatorBool_DACQuWriteTimeout = 0x8186,
   NiFpga_ContSeqV204_IndicatorBool_SPCtrQuWriteTimeout = 0x8142,
} NiFpga_ContSeqV204_IndicatorBool;

typedef enum
{
   NiFpga_ContSeqV204_IndicatorU8_SPerrorCount = 0x8136,
   NiFpga_ContSeqV204_IndicatorU8_postAccOffsetVoltState = 0x812E,
} NiFpga_ContSeqV204_IndicatorU8;

typedef enum
{
   NiFpga_ContSeqV204_IndicatorU16_SPstate = 0x813A,
   NiFpga_ContSeqV204_IndicatorU16_measVoltState = 0x8182,
   NiFpga_ContSeqV204_IndicatorU16_seqState = 0x816E,
} NiFpga_ContSeqV204_IndicatorU16;

typedef enum
{
   NiFpga_ContSeqV204_ControlBool_VoltOrScaler = 0x810E,
   NiFpga_ContSeqV204_ControlBool_abort = 0x8176,
   NiFpga_ContSeqV204_ControlBool_halt = 0x817A,
   NiFpga_ContSeqV204_ControlBool_hostConfirmsHzOffsetIsSet = 0x817E,
   NiFpga_ContSeqV204_ControlBool_invertScan = 0x8156,
   NiFpga_ContSeqV204_ControlBool_stopVoltMeas = 0x8112,
   NiFpga_ContSeqV204_ControlBool_timedOutWhileHandshake = 0x816A,
} NiFpga_ContSeqV204_ControlBool;

typedef enum
{
   NiFpga_ContSeqV204_ControlU8_postAccOffsetVoltControl = 0x8132,
   NiFpga_ContSeqV204_ControlU8_selectTrigger = 0x8126,
   NiFpga_ContSeqV204_ControlU8_triggerEdge = 0x811E,
} NiFpga_ContSeqV204_ControlU8;

typedef enum
{
   NiFpga_ContSeqV204_ControlU16_cmdByHost = 0x8172,
   NiFpga_ContSeqV204_ControlU16_measVoltCompleteDest = 0x8116,
   NiFpga_ContSeqV204_ControlU16_triggerTypes = 0x812A,
   NiFpga_ContSeqV204_ControlU16_waitAfterReset25nsTicks = 0x815E,
   NiFpga_ContSeqV204_ControlU16_waitForKepco25nsTicks = 0x815A,
} NiFpga_ContSeqV204_ControlU16;

typedef enum
{
   NiFpga_ContSeqV204_ControlI32_dacStartRegister18Bit = 0x8148,
   NiFpga_ContSeqV204_ControlI32_dacStepSize18Bit = 0x8144,
   NiFpga_ContSeqV204_ControlI32_measVoltPulseLength25ns = 0x8160,
   NiFpga_ContSeqV204_ControlI32_measVoltTimeout10ns = 0x8164,
   NiFpga_ContSeqV204_ControlI32_nOfScans = 0x8150,
   NiFpga_ContSeqV204_ControlI32_nOfSteps = 0x814C,
} NiFpga_ContSeqV204_ControlI32;

typedef enum
{
   NiFpga_ContSeqV204_ControlU32_dac0VRegister = 0x8118,
   NiFpga_ContSeqV204_ControlU32_dwellTime10ns = 0x813C,
   NiFpga_ContSeqV204_ControlU32_trigDelay10ns = 0x8120,
} NiFpga_ContSeqV204_ControlU32;

typedef enum
{
   NiFpga_ContSeqV204_TargetToHostFifoU32_transferToHost = 0,
} NiFpga_ContSeqV204_TargetToHostFifoU32;

#endif
