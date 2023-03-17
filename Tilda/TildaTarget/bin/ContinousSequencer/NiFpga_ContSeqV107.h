/*
 * Generated with the FPGA Interface C API Generator 14.0.0
 * for NI-RIO 14.0.0 or later.
 */

#ifndef __NiFpga_ContSeqV107_h__
#define __NiFpga_ContSeqV107_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1400
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_ContSeqV107_Bitfile;
 */
#define NiFpga_ContSeqV107_Bitfile "NiFpga_ContSeqV107.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_ContSeqV107_Signature = "52CC5E3A5AA826A7532D2637CD2DACFA";

typedef enum
{
   NiFpga_ContSeqV107_IndicatorBool_DACQuWriteTimeout = 0x8176,
   NiFpga_ContSeqV107_IndicatorBool_SPCtrQuWriteTimeout = 0x8132,
} NiFpga_ContSeqV107_IndicatorBool;

typedef enum
{
   NiFpga_ContSeqV107_IndicatorU8_SPerrorCount = 0x8126,
   NiFpga_ContSeqV107_IndicatorU8_postAccOffsetVoltState = 0x811E,
} NiFpga_ContSeqV107_IndicatorU8;

typedef enum
{
   NiFpga_ContSeqV107_IndicatorU16_SPstate = 0x812A,
   NiFpga_ContSeqV107_IndicatorU16_measVoltState = 0x8172,
   NiFpga_ContSeqV107_IndicatorU16_seqState = 0x815E,
} NiFpga_ContSeqV107_IndicatorU16;

typedef enum
{
   NiFpga_ContSeqV107_ControlBool_VoltOrScaler = 0x810E,
   NiFpga_ContSeqV107_ControlBool_abort = 0x8166,
   NiFpga_ContSeqV107_ControlBool_halt = 0x816A,
   NiFpga_ContSeqV107_ControlBool_hostConfirmsHzOffsetIsSet = 0x816E,
   NiFpga_ContSeqV107_ControlBool_invertScan = 0x8146,
   NiFpga_ContSeqV107_ControlBool_timedOutWhileHandshake = 0x815A,
} NiFpga_ContSeqV107_ControlBool;

typedef enum
{
   NiFpga_ContSeqV107_ControlU8_postAccOffsetVoltControl = 0x8122,
   NiFpga_ContSeqV107_ControlU8_selectTrigger = 0x8116,
} NiFpga_ContSeqV107_ControlU8;

typedef enum
{
   NiFpga_ContSeqV107_ControlU16_cmdByHost = 0x8162,
   NiFpga_ContSeqV107_ControlU16_triggerTypes = 0x811A,
   NiFpga_ContSeqV107_ControlU16_waitAfterReset25nsTicks = 0x814E,
   NiFpga_ContSeqV107_ControlU16_waitForKepco25nsTicks = 0x814A,
} NiFpga_ContSeqV107_ControlU16;

typedef enum
{
   NiFpga_ContSeqV107_ControlI32_dacStartRegister18Bit = 0x8138,
   NiFpga_ContSeqV107_ControlI32_dacStepSize18Bit = 0x8134,
   NiFpga_ContSeqV107_ControlI32_measVoltPulseLength25ns = 0x8150,
   NiFpga_ContSeqV107_ControlI32_measVoltTimeout10ns = 0x8154,
   NiFpga_ContSeqV107_ControlI32_nOfScans = 0x8140,
   NiFpga_ContSeqV107_ControlI32_nOfSteps = 0x813C,
} NiFpga_ContSeqV107_ControlI32;

typedef enum
{
   NiFpga_ContSeqV107_ControlU32_delay_10ns_ticks = 0x8110,
   NiFpga_ContSeqV107_ControlU32_dwellTime10ns = 0x812C,
} NiFpga_ContSeqV107_ControlU32;

typedef enum
{
   NiFpga_ContSeqV107_TargetToHostFifoU32_transferToHost = 0,
} NiFpga_ContSeqV107_TargetToHostFifoU32;

#endif
