/*
 * Generated with the FPGA Interface C API Generator 14.0.0
 * for NI-RIO 14.0.0 or later.
 */

#ifndef __NiFpga_CS_h__
#define __NiFpga_CS_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1400
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_CS_Bitfile;
 */
#define NiFpga_CS_Bitfile "D:\\Workspace\\Eclipse\\Tilda\\TildaTarget\\bin\\ContinousSequencer\\NiFpga_CS.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_CS_Signature = "85C054988B741A8519365445F387CA1D";

typedef enum
{
   NiFpga_CS_IndicatorBool_DACQuWriteTimeout = 0x8166,
   NiFpga_CS_IndicatorBool_SPCtrQuWriteTimeout = 0x811E,
} NiFpga_CS_IndicatorBool;

typedef enum
{
   NiFpga_CS_IndicatorU8_SPerrorCount = 0x8112,
} NiFpga_CS_IndicatorU8;

typedef enum
{
   NiFpga_CS_IndicatorU16_SPstate = 0x8116,
   NiFpga_CS_IndicatorU16_measVoltState = 0x8162,
   NiFpga_CS_IndicatorU16_seqState = 0x814E,
} NiFpga_CS_IndicatorU16;

typedef enum
{
   NiFpga_CS_ControlBool_VoltOrScaler = 0x810E,
   NiFpga_CS_ControlBool_abort = 0x8156,
   NiFpga_CS_ControlBool_halt = 0x815A,
   NiFpga_CS_ControlBool_hostConfirmsHzOffsetIsSet = 0x815E,
   NiFpga_CS_ControlBool_invertScan = 0x8132,
   NiFpga_CS_ControlBool_timedOutWhileHandshake = 0x814A,
} NiFpga_CS_ControlBool;

typedef enum
{
   NiFpga_CS_ControlU8_heinzingerControl = 0x8136,
} NiFpga_CS_ControlU8;

typedef enum
{
   NiFpga_CS_ControlU16_cmdByHost = 0x8152,
   NiFpga_CS_ControlU16_waitAfterReset25nsTicks = 0x813E,
   NiFpga_CS_ControlU16_waitForKepco25nsTicks = 0x813A,
} NiFpga_CS_ControlU16;

typedef enum
{
   NiFpga_CS_ControlI32_measVoltPulseLength25ns = 0x8140,
   NiFpga_CS_ControlI32_measVoltTimeout10ns = 0x8144,
   NiFpga_CS_ControlI32_nOfScans = 0x812C,
   NiFpga_CS_ControlI32_nOfSteps = 0x8128,
   NiFpga_CS_ControlI32_start = 0x8124,
   NiFpga_CS_ControlI32_stepSize = 0x8120,
} NiFpga_CS_ControlI32;

typedef enum
{
   NiFpga_CS_ControlU32_dwellTime = 0x8118,
} NiFpga_CS_ControlU32;

typedef enum
{
   NiFpga_CS_TargetToHostFifoU32_transferToHost = 0,
} NiFpga_CS_TargetToHostFifoU32;

#endif
