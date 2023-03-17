/*
 * Generated with the FPGA Interface C API Generator 14.0.0
 * for NI-RIO 14.0.0 or later.
 */

#ifndef __NiFpga_TRS_DAF_240_h__
#define __NiFpga_TRS_DAF_240_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1400
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_TRS_DAF_240_Bitfile;
 */
#define NiFpga_TRS_DAF_240_Bitfile "NiFpga_TRS_DAF_240.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_TRS_DAF_240_Signature = "CF170A704FBF3A62DAA5416FF2A31A4A";

typedef enum
{
   NiFpga_TRS_DAF_240_IndicatorBool_DACQuWriteTimeout = 0x817A,
   NiFpga_TRS_DAF_240_IndicatorBool_MCSQuWriteTimeout = 0x817E,
} NiFpga_TRS_DAF_240_IndicatorBool;

typedef enum
{
   NiFpga_TRS_DAF_240_IndicatorI8_MCSerrorcount = 0x8176,
} NiFpga_TRS_DAF_240_IndicatorI8;

typedef enum
{
   NiFpga_TRS_DAF_240_IndicatorU8_postAccOffsetVoltState = 0x816E,
} NiFpga_TRS_DAF_240_IndicatorU8;

typedef enum
{
   NiFpga_TRS_DAF_240_IndicatorU16_MCSstate = 0x8186,
   NiFpga_TRS_DAF_240_IndicatorU16_OutBitsState = 0x8152,
   NiFpga_TRS_DAF_240_IndicatorU16_measVoltState = 0x8182,
   NiFpga_TRS_DAF_240_IndicatorU16_seqState = 0x81AA,
} NiFpga_TRS_DAF_240_IndicatorU16;

typedef enum
{
   NiFpga_TRS_DAF_240_IndicatorU32_nOfCmdsOutbit0 = 0x814C,
   NiFpga_TRS_DAF_240_IndicatorU32_nOfCmdsOutbit1 = 0x8148,
   NiFpga_TRS_DAF_240_IndicatorU32_nOfCmdsOutbit2 = 0x8144,
} NiFpga_TRS_DAF_240_IndicatorU32;

typedef enum
{
   NiFpga_TRS_DAF_240_ControlBool_LaserSet = 0x8112,
   NiFpga_TRS_DAF_240_ControlBool_VoltOrScaler = 0x810E,
   NiFpga_TRS_DAF_240_ControlBool_abort = 0x81A2,
   NiFpga_TRS_DAF_240_ControlBool_halt = 0x819E,
   NiFpga_TRS_DAF_240_ControlBool_hostConfirmsHzOffsetIsSet = 0x819A,
   NiFpga_TRS_DAF_240_ControlBool_invertScan = 0x81C6,
   NiFpga_TRS_DAF_240_ControlBool_pause = 0x815A,
   NiFpga_TRS_DAF_240_ControlBool_softwareScanTrigger = 0x8136,
   NiFpga_TRS_DAF_240_ControlBool_softwareStepTrigger = 0x8122,
   NiFpga_TRS_DAF_240_ControlBool_softwareTrigger = 0x8156,
   NiFpga_TRS_DAF_240_ControlBool_stopVoltMeas = 0x815E,
   NiFpga_TRS_DAF_240_ControlBool_timedOutWhileHandshake = 0x81AE,
} NiFpga_TRS_DAF_240_ControlBool;

typedef enum
{
   NiFpga_TRS_DAF_240_ControlU8_postAccOffsetVoltControl = 0x81C2,
   NiFpga_TRS_DAF_240_ControlU8_scanTriggerEdge = 0x813A,
   NiFpga_TRS_DAF_240_ControlU8_selectScanTrigger = 0x8142,
   NiFpga_TRS_DAF_240_ControlU8_selectStepTrigger = 0x812E,
   NiFpga_TRS_DAF_240_ControlU8_selectTrigger = 0x8196,
   NiFpga_TRS_DAF_240_ControlU8_stepTriggerEdge = 0x8126,
   NiFpga_TRS_DAF_240_ControlU8_triggerEdge = 0x816A,
} NiFpga_TRS_DAF_240_ControlU8;

typedef enum
{
   NiFpga_TRS_DAF_240_ControlU16_ScanDevice = 0x811A,
   NiFpga_TRS_DAF_240_ControlU16_cmdByHost = 0x81A6,
   NiFpga_TRS_DAF_240_ControlU16_measVoltCompleteDest = 0x8162,
   NiFpga_TRS_DAF_240_ControlU16_scanTriggerTypes = 0x813E,
   NiFpga_TRS_DAF_240_ControlU16_stepTriggerTypes = 0x812A,
   NiFpga_TRS_DAF_240_ControlU16_triggerTypes = 0x8172,
} NiFpga_TRS_DAF_240_ControlU16;

typedef enum
{
   NiFpga_TRS_DAF_240_ControlI32_LaserSetTimeout10ns = 0x8114,
   NiFpga_TRS_DAF_240_ControlI32_dacStartRegister18Bit = 0x81D0,
   NiFpga_TRS_DAF_240_ControlI32_dacStepSize18Bit = 0x81D4,
   NiFpga_TRS_DAF_240_ControlI32_measVoltPulseLength25ns = 0x81B4,
   NiFpga_TRS_DAF_240_ControlI32_measVoltTimeout10ns = 0x81B0,
   NiFpga_TRS_DAF_240_ControlI32_nOfBunches = 0x8188,
   NiFpga_TRS_DAF_240_ControlI32_nOfScans = 0x81C8,
   NiFpga_TRS_DAF_240_ControlI32_nOfSteps = 0x81CC,
} NiFpga_TRS_DAF_240_ControlI32;

typedef enum
{
   NiFpga_TRS_DAF_240_ControlU32_dac0VRegister = 0x8164,
   NiFpga_TRS_DAF_240_ControlU32_nOfBins = 0x818C,
   NiFpga_TRS_DAF_240_ControlU32_scanTrigDelay10ns = 0x8130,
   NiFpga_TRS_DAF_240_ControlU32_stepTrigDelay10ns = 0x811C,
   NiFpga_TRS_DAF_240_ControlU32_trigDelay10ns = 0x8190,
   NiFpga_TRS_DAF_240_ControlU32_waitAfterResetus = 0x81B8,
   NiFpga_TRS_DAF_240_ControlU32_waitForKepcous = 0x81BC,
} NiFpga_TRS_DAF_240_ControlU32;

typedef enum
{
   NiFpga_TRS_DAF_240_TargetToHostFifoU32_transferToHost = 0,
} NiFpga_TRS_DAF_240_TargetToHostFifoU32;

typedef enum
{
   NiFpga_TRS_DAF_240_HostToTargetFifoU32_OutbitsCMD = 1,
} NiFpga_TRS_DAF_240_HostToTargetFifoU32;

#endif
