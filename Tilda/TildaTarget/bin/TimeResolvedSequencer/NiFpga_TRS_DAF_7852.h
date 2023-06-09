/*
 * Generated with the FPGA Interface C API Generator 14.0.0
 * for NI-RIO 14.0.0 or later.
 */

#ifndef __NiFpga_TRS_DAF_7852_h__
#define __NiFpga_TRS_DAF_7852_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1400
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_TRS_DAF_7852_Bitfile;
 */
#define NiFpga_TRS_DAF_7852_Bitfile "NiFpga_TRS_DAF_7852.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_TRS_DAF_7852_Signature = "570404AF38BC448E410B52CC8C899484";

typedef enum
{
   NiFpga_TRS_DAF_7852_IndicatorBool_DACQuWriteTimeout = 0x817E,
   NiFpga_TRS_DAF_7852_IndicatorBool_MCSQuWriteTimeout = 0x8182,
   NiFpga_TRS_DAF_7852_IndicatorBool_internalDacAvailable = 0x8112,
} NiFpga_TRS_DAF_7852_IndicatorBool;

typedef enum
{
   NiFpga_TRS_DAF_7852_IndicatorI8_MCSerrorcount = 0x817A,
} NiFpga_TRS_DAF_7852_IndicatorI8;

typedef enum
{
   NiFpga_TRS_DAF_7852_IndicatorU8_postAccOffsetVoltState = 0x8172,
} NiFpga_TRS_DAF_7852_IndicatorU8;

typedef enum
{
   NiFpga_TRS_DAF_7852_IndicatorU16_MCSstate = 0x818A,
   NiFpga_TRS_DAF_7852_IndicatorU16_OutBitsState = 0x8156,
   NiFpga_TRS_DAF_7852_IndicatorU16_measVoltState = 0x8186,
   NiFpga_TRS_DAF_7852_IndicatorU16_seqState = 0x81AE,
} NiFpga_TRS_DAF_7852_IndicatorU16;

typedef enum
{
   NiFpga_TRS_DAF_7852_IndicatorU32_nOfCmdsOutbit0 = 0x8150,
   NiFpga_TRS_DAF_7852_IndicatorU32_nOfCmdsOutbit1 = 0x814C,
   NiFpga_TRS_DAF_7852_IndicatorU32_nOfCmdsOutbit2 = 0x8148,
} NiFpga_TRS_DAF_7852_IndicatorU32;

typedef enum
{
   NiFpga_TRS_DAF_7852_ControlBool_VoltOrScaler = 0x810E,
   NiFpga_TRS_DAF_7852_ControlBool_abort = 0x81A6,
   NiFpga_TRS_DAF_7852_ControlBool_halt = 0x81A2,
   NiFpga_TRS_DAF_7852_ControlBool_hostConfirmsHzOffsetIsSet = 0x819E,
   NiFpga_TRS_DAF_7852_ControlBool_invertScan = 0x81CA,
   NiFpga_TRS_DAF_7852_ControlBool_pause = 0x815E,
   NiFpga_TRS_DAF_7852_ControlBool_scanDevSet = 0x811A,
   NiFpga_TRS_DAF_7852_ControlBool_softwareScanTrigger = 0x813A,
   NiFpga_TRS_DAF_7852_ControlBool_softwareStepTrigger = 0x8126,
   NiFpga_TRS_DAF_7852_ControlBool_softwareTrigger = 0x815A,
   NiFpga_TRS_DAF_7852_ControlBool_stopVoltMeas = 0x8162,
   NiFpga_TRS_DAF_7852_ControlBool_timedOutWhileHandshake = 0x81B2,
} NiFpga_TRS_DAF_7852_ControlBool;

typedef enum
{
   NiFpga_TRS_DAF_7852_ControlU8_postAccOffsetVoltControl = 0x81C6,
   NiFpga_TRS_DAF_7852_ControlU8_scanTriggerEdge = 0x813E,
   NiFpga_TRS_DAF_7852_ControlU8_selectScanTrigger = 0x8146,
   NiFpga_TRS_DAF_7852_ControlU8_selectStepTrigger = 0x8132,
   NiFpga_TRS_DAF_7852_ControlU8_selectTrigger = 0x819A,
   NiFpga_TRS_DAF_7852_ControlU8_stepTriggerEdge = 0x812A,
   NiFpga_TRS_DAF_7852_ControlU8_triggerEdge = 0x816E,
} NiFpga_TRS_DAF_7852_ControlU8;

typedef enum
{
   NiFpga_TRS_DAF_7852_ControlU16_ScanDevice = 0x8116,
   NiFpga_TRS_DAF_7852_ControlU16_cmdByHost = 0x81AA,
   NiFpga_TRS_DAF_7852_ControlU16_measVoltCompleteDest = 0x8166,
   NiFpga_TRS_DAF_7852_ControlU16_scanTriggerTypes = 0x8142,
   NiFpga_TRS_DAF_7852_ControlU16_stepTriggerTypes = 0x812E,
   NiFpga_TRS_DAF_7852_ControlU16_triggerTypes = 0x8176,
} NiFpga_TRS_DAF_7852_ControlU16;

typedef enum
{
   NiFpga_TRS_DAF_7852_ControlI32_dacStartRegister20Bit = 0x81D4,
   NiFpga_TRS_DAF_7852_ControlI32_dacStepSize20Bit = 0x81D8,
   NiFpga_TRS_DAF_7852_ControlI32_measVoltPulseLength25ns = 0x81B8,
   NiFpga_TRS_DAF_7852_ControlI32_measVoltTimeout10ns = 0x81B4,
   NiFpga_TRS_DAF_7852_ControlI32_nOfBunches = 0x818C,
   NiFpga_TRS_DAF_7852_ControlI32_nOfScans = 0x81CC,
   NiFpga_TRS_DAF_7852_ControlI32_nOfSteps = 0x81D0,
   NiFpga_TRS_DAF_7852_ControlI32_scanDevTimeout10ns = 0x811C,
} NiFpga_TRS_DAF_7852_ControlI32;

typedef enum
{
   NiFpga_TRS_DAF_7852_ControlU32_dac0VRegister = 0x8168,
   NiFpga_TRS_DAF_7852_ControlU32_nOfBins = 0x8190,
   NiFpga_TRS_DAF_7852_ControlU32_scanTrigDelay10ns = 0x8134,
   NiFpga_TRS_DAF_7852_ControlU32_stepTrigDelay10ns = 0x8120,
   NiFpga_TRS_DAF_7852_ControlU32_trigDelay10ns = 0x8194,
   NiFpga_TRS_DAF_7852_ControlU32_waitAfterResetus = 0x81BC,
   NiFpga_TRS_DAF_7852_ControlU32_waitForKepcous = 0x81C0,
} NiFpga_TRS_DAF_7852_ControlU32;

typedef enum
{
   NiFpga_TRS_DAF_7852_TargetToHostFifoU32_transferToHost = 0,
} NiFpga_TRS_DAF_7852_TargetToHostFifoU32;

typedef enum
{
   NiFpga_TRS_DAF_7852_HostToTargetFifoU32_OutbitsCMD = 1,
} NiFpga_TRS_DAF_7852_HostToTargetFifoU32;

#endif
