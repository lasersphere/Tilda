/*
 * Generated with the FPGA Interface C API Generator 14.0.0
 * for NI-RIO 14.0.0 or later.
 */

#ifndef __NiFpga_TRS_DAF_103_h__
#define __NiFpga_TRS_DAF_103_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1400
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_TRS_DAF_103_Bitfile;
 */
#define NiFpga_TRS_DAF_103_Bitfile "NiFpga_TRS_DAF_103.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_TRS_DAF_103_Signature = "BF31570369009FA00617B7055FD697C8";

typedef enum
{
   NiFpga_TRS_DAF_103_IndicatorBool_DACQuWriteTimeout = 0x8116,
   NiFpga_TRS_DAF_103_IndicatorBool_MCSQuWriteTimeout = 0x811A,
} NiFpga_TRS_DAF_103_IndicatorBool;

typedef enum
{
   NiFpga_TRS_DAF_103_IndicatorI8_MCSerrorcount = 0x8112,
} NiFpga_TRS_DAF_103_IndicatorI8;

typedef enum
{
   NiFpga_TRS_DAF_103_IndicatorU16_MCSstate = 0x8122,
   NiFpga_TRS_DAF_103_IndicatorU16_measVoltState = 0x811E,
   NiFpga_TRS_DAF_103_IndicatorU16_seqState = 0x8146,
} NiFpga_TRS_DAF_103_IndicatorU16;

typedef enum
{
   NiFpga_TRS_DAF_103_ControlBool_VoltOrScaler = 0x810E,
   NiFpga_TRS_DAF_103_ControlBool_abort = 0x813E,
   NiFpga_TRS_DAF_103_ControlBool_halt = 0x813A,
   NiFpga_TRS_DAF_103_ControlBool_hostConfirmsHzOffsetIsSet = 0x8136,
   NiFpga_TRS_DAF_103_ControlBool_invertScan = 0x8162,
   NiFpga_TRS_DAF_103_ControlBool_timedOutWhileHandshake = 0x814A,
} NiFpga_TRS_DAF_103_ControlBool;

typedef enum
{
   NiFpga_TRS_DAF_103_ControlU8_MCSSelectTrigger = 0x8132,
   NiFpga_TRS_DAF_103_ControlU8_heinzingerControl = 0x815E,
} NiFpga_TRS_DAF_103_ControlU8;

typedef enum
{
   NiFpga_TRS_DAF_103_ControlU16_cmdByHost = 0x8142,
   NiFpga_TRS_DAF_103_ControlU16_waitAfterReset25nsTicks = 0x8156,
   NiFpga_TRS_DAF_103_ControlU16_waitForKepco25nsTicks = 0x815A,
} NiFpga_TRS_DAF_103_ControlU16;

typedef enum
{
   NiFpga_TRS_DAF_103_ControlI32_measVoltPulseLength25ns = 0x8150,
   NiFpga_TRS_DAF_103_ControlI32_measVoltTimeout10ns = 0x814C,
   NiFpga_TRS_DAF_103_ControlI32_nOfBunches = 0x8124,
   NiFpga_TRS_DAF_103_ControlI32_nOfScans = 0x8164,
   NiFpga_TRS_DAF_103_ControlI32_nOfSteps = 0x8168,
   NiFpga_TRS_DAF_103_ControlI32_start = 0x816C,
   NiFpga_TRS_DAF_103_ControlI32_stepSize = 0x8170,
} NiFpga_TRS_DAF_103_ControlI32;

typedef enum
{
   NiFpga_TRS_DAF_103_ControlU32_delayticks = 0x812C,
   NiFpga_TRS_DAF_103_ControlU32_nOfBins = 0x8128,
} NiFpga_TRS_DAF_103_ControlU32;

typedef enum
{
   NiFpga_TRS_DAF_103_TargetToHostFifoU32_transferToHost = 0,
} NiFpga_TRS_DAF_103_TargetToHostFifoU32;

#endif
