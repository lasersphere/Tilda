/*
 * Generated with the FPGA Interface C API Generator 14.0.0
 * for NI-RIO 14.0.0 or later.
 */

#ifndef __NiFpga_ContSeqV104_h__
#define __NiFpga_ContSeqV104_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1400
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_ContSeqV104_Bitfile;
 */
#define NiFpga_ContSeqV104_Bitfile "NiFpga_ContSeqV104.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_ContSeqV104_Signature = "2F607F7C7F5AC2066074FC066258B6C3";

typedef enum
{
   NiFpga_ContSeqV104_IndicatorBool_DACQuWriteTimeout = 0x8166,
   NiFpga_ContSeqV104_IndicatorBool_SPCtrQuWriteTimeout = 0x811E,
} NiFpga_ContSeqV104_IndicatorBool;

typedef enum
{
   NiFpga_ContSeqV104_IndicatorU8_SPerrorCount = 0x8112,
} NiFpga_ContSeqV104_IndicatorU8;

typedef enum
{
   NiFpga_ContSeqV104_IndicatorU16_SPstate = 0x8116,
   NiFpga_ContSeqV104_IndicatorU16_measVoltState = 0x8162,
   NiFpga_ContSeqV104_IndicatorU16_seqState = 0x814E,
} NiFpga_ContSeqV104_IndicatorU16;

typedef enum
{
   NiFpga_ContSeqV104_ControlBool_VoltOrScaler = 0x810E,
   NiFpga_ContSeqV104_ControlBool_abort = 0x8156,
   NiFpga_ContSeqV104_ControlBool_halt = 0x815A,
   NiFpga_ContSeqV104_ControlBool_hostConfirmsHzOffsetIsSet = 0x815E,
   NiFpga_ContSeqV104_ControlBool_invertScan = 0x8132,
   NiFpga_ContSeqV104_ControlBool_timedOutWhileHandshake = 0x814A,
} NiFpga_ContSeqV104_ControlBool;

typedef enum
{
   NiFpga_ContSeqV104_ControlU8_heinzingerControl = 0x8136,
} NiFpga_ContSeqV104_ControlU8;

typedef enum
{
   NiFpga_ContSeqV104_ControlU16_cmdByHost = 0x8152,
   NiFpga_ContSeqV104_ControlU16_waitAfterReset25nsTicks = 0x813E,
   NiFpga_ContSeqV104_ControlU16_waitForKepco25nsTicks = 0x813A,
} NiFpga_ContSeqV104_ControlU16;

typedef enum
{
   NiFpga_ContSeqV104_ControlI32_measVoltPulseLength25ns = 0x8140,
   NiFpga_ContSeqV104_ControlI32_measVoltTimeout10ns = 0x8144,
   NiFpga_ContSeqV104_ControlI32_nOfScans = 0x812C,
   NiFpga_ContSeqV104_ControlI32_nOfSteps = 0x8128,
   NiFpga_ContSeqV104_ControlI32_start = 0x8124,
   NiFpga_ContSeqV104_ControlI32_stepSize = 0x8120,
} NiFpga_ContSeqV104_ControlI32;

typedef enum
{
   NiFpga_ContSeqV104_ControlU32_dwellTime = 0x8118,
} NiFpga_ContSeqV104_ControlU32;

typedef enum
{
   NiFpga_ContSeqV104_TargetToHostFifoU32_transferToHost = 0,
} NiFpga_ContSeqV104_TargetToHostFifoU32;

#endif
