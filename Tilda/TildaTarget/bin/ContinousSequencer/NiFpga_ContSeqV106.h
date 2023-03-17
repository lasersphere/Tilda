/*
 * Generated with the FPGA Interface C API Generator 14.0.0
 * for NI-RIO 14.0.0 or later.
 */

#ifndef __NiFpga_ContSeqV106_h__
#define __NiFpga_ContSeqV106_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1400
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_ContSeqV106_Bitfile;
 */
#define NiFpga_ContSeqV106_Bitfile "NiFpga_ContSeqV106.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_ContSeqV106_Signature = "E0E658970DBC3F1F39674D0887522CAC";

typedef enum
{
   NiFpga_ContSeqV106_IndicatorBool_DACQuWriteTimeout = 0x816A,
   NiFpga_ContSeqV106_IndicatorBool_SPCtrQuWriteTimeout = 0x8126,
} NiFpga_ContSeqV106_IndicatorBool;

typedef enum
{
   NiFpga_ContSeqV106_IndicatorU8_SPerrorCount = 0x811A,
   NiFpga_ContSeqV106_IndicatorU8_postAccOffsetVoltState = 0x8112,
} NiFpga_ContSeqV106_IndicatorU8;

typedef enum
{
   NiFpga_ContSeqV106_IndicatorU16_SPstate = 0x811E,
   NiFpga_ContSeqV106_IndicatorU16_measVoltState = 0x8166,
   NiFpga_ContSeqV106_IndicatorU16_seqState = 0x8152,
} NiFpga_ContSeqV106_IndicatorU16;

typedef enum
{
   NiFpga_ContSeqV106_ControlBool_VoltOrScaler = 0x810E,
   NiFpga_ContSeqV106_ControlBool_abort = 0x815A,
   NiFpga_ContSeqV106_ControlBool_halt = 0x815E,
   NiFpga_ContSeqV106_ControlBool_hostConfirmsHzOffsetIsSet = 0x8162,
   NiFpga_ContSeqV106_ControlBool_invertScan = 0x813A,
   NiFpga_ContSeqV106_ControlBool_timedOutWhileHandshake = 0x814E,
} NiFpga_ContSeqV106_ControlBool;

typedef enum
{
   NiFpga_ContSeqV106_ControlU8_postAccOffsetVoltControl = 0x8116,
} NiFpga_ContSeqV106_ControlU8;

typedef enum
{
   NiFpga_ContSeqV106_ControlU16_cmdByHost = 0x8156,
   NiFpga_ContSeqV106_ControlU16_waitAfterReset25nsTicks = 0x8142,
   NiFpga_ContSeqV106_ControlU16_waitForKepco25nsTicks = 0x813E,
} NiFpga_ContSeqV106_ControlU16;

typedef enum
{
   NiFpga_ContSeqV106_ControlI32_dacStartRegister18Bit = 0x812C,
   NiFpga_ContSeqV106_ControlI32_dacStepSize18Bit = 0x8128,
   NiFpga_ContSeqV106_ControlI32_measVoltPulseLength25ns = 0x8144,
   NiFpga_ContSeqV106_ControlI32_measVoltTimeout10ns = 0x8148,
   NiFpga_ContSeqV106_ControlI32_nOfScans = 0x8134,
   NiFpga_ContSeqV106_ControlI32_nOfSteps = 0x8130,
} NiFpga_ContSeqV106_ControlI32;

typedef enum
{
   NiFpga_ContSeqV106_ControlU32_dwellTime10ns = 0x8120,
} NiFpga_ContSeqV106_ControlU32;

typedef enum
{
   NiFpga_ContSeqV106_TargetToHostFifoU32_transferToHost = 0,
} NiFpga_ContSeqV106_TargetToHostFifoU32;

#endif
