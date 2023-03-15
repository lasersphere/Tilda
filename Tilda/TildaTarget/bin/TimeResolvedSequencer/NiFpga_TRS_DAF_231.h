/*
 * Generated with the FPGA Interface C API Generator 14.0.0
 * for NI-RIO 14.0.0 or later.
 */

#ifndef __NiFpga_TRS_DAF_231_h__
#define __NiFpga_TRS_DAF_231_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1400
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_TRS_DAF_231_Bitfile;
 */
#define NiFpga_TRS_DAF_231_Bitfile "NiFpga_TRS_DAF_231.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_TRS_DAF_231_Signature = "08557AE0FB8520D28576FA23989A63E2";

typedef enum
{
   NiFpga_TRS_DAF_231_IndicatorBool_DACQuWriteTimeout = 0x816E,
   NiFpga_TRS_DAF_231_IndicatorBool_MCSQuWriteTimeout = 0x8172,
} NiFpga_TRS_DAF_231_IndicatorBool;

typedef enum
{
   NiFpga_TRS_DAF_231_IndicatorI8_MCSerrorcount = 0x816A,
} NiFpga_TRS_DAF_231_IndicatorI8;

typedef enum
{
   NiFpga_TRS_DAF_231_IndicatorU8_postAccOffsetVoltState = 0x8162,
} NiFpga_TRS_DAF_231_IndicatorU8;

typedef enum
{
   NiFpga_TRS_DAF_231_IndicatorU16_MCSstate = 0x817A,
   NiFpga_TRS_DAF_231_IndicatorU16_OutBitsState = 0x8146,
   NiFpga_TRS_DAF_231_IndicatorU16_measVoltState = 0x8176,
   NiFpga_TRS_DAF_231_IndicatorU16_seqState = 0x819E,
} NiFpga_TRS_DAF_231_IndicatorU16;

typedef enum
{
   NiFpga_TRS_DAF_231_IndicatorU32_nOfCmdsOutbit0 = 0x8140,
   NiFpga_TRS_DAF_231_IndicatorU32_nOfCmdsOutbit1 = 0x813C,
   NiFpga_TRS_DAF_231_IndicatorU32_nOfCmdsOutbit2 = 0x8138,
} NiFpga_TRS_DAF_231_IndicatorU32;

typedef enum
{
   NiFpga_TRS_DAF_231_ControlBool_VoltOrScaler = 0x810E,
   NiFpga_TRS_DAF_231_ControlBool_abort = 0x8196,
   NiFpga_TRS_DAF_231_ControlBool_halt = 0x8192,
   NiFpga_TRS_DAF_231_ControlBool_hostConfirmsHzOffsetIsSet = 0x818E,
   NiFpga_TRS_DAF_231_ControlBool_invertScan = 0x81BA,
   NiFpga_TRS_DAF_231_ControlBool_pause = 0x814E,
   NiFpga_TRS_DAF_231_ControlBool_softwareScanTrigger = 0x812A,
   NiFpga_TRS_DAF_231_ControlBool_softwareStepTrigger = 0x8116,
   NiFpga_TRS_DAF_231_ControlBool_softwareTrigger = 0x814A,
   NiFpga_TRS_DAF_231_ControlBool_stopVoltMeas = 0x8152,
   NiFpga_TRS_DAF_231_ControlBool_timedOutWhileHandshake = 0x81A2,
} NiFpga_TRS_DAF_231_ControlBool;

typedef enum
{
   NiFpga_TRS_DAF_231_ControlU8_postAccOffsetVoltControl = 0x81B6,
   NiFpga_TRS_DAF_231_ControlU8_scanTriggerEdge = 0x812E,
   NiFpga_TRS_DAF_231_ControlU8_selectScanTrigger = 0x8136,
   NiFpga_TRS_DAF_231_ControlU8_selectStepTrigger = 0x8122,
   NiFpga_TRS_DAF_231_ControlU8_selectTrigger = 0x818A,
   NiFpga_TRS_DAF_231_ControlU8_stepTriggerEdge = 0x811A,
   NiFpga_TRS_DAF_231_ControlU8_triggerEdge = 0x815E,
} NiFpga_TRS_DAF_231_ControlU8;

typedef enum
{
   NiFpga_TRS_DAF_231_ControlU16_cmdByHost = 0x819A,
   NiFpga_TRS_DAF_231_ControlU16_measVoltCompleteDest = 0x8156,
   NiFpga_TRS_DAF_231_ControlU16_scanTriggerTypes = 0x8132,
   NiFpga_TRS_DAF_231_ControlU16_stepTriggerTypes = 0x811E,
   NiFpga_TRS_DAF_231_ControlU16_triggerTypes = 0x8166,
} NiFpga_TRS_DAF_231_ControlU16;

typedef enum
{
   NiFpga_TRS_DAF_231_ControlI32_dacStartRegister18Bit = 0x81C4,
   NiFpga_TRS_DAF_231_ControlI32_dacStepSize18Bit = 0x81C8,
   NiFpga_TRS_DAF_231_ControlI32_measVoltPulseLength25ns = 0x81A8,
   NiFpga_TRS_DAF_231_ControlI32_measVoltTimeout10ns = 0x81A4,
   NiFpga_TRS_DAF_231_ControlI32_nOfBunches = 0x817C,
   NiFpga_TRS_DAF_231_ControlI32_nOfScans = 0x81BC,
   NiFpga_TRS_DAF_231_ControlI32_nOfSteps = 0x81C0,
} NiFpga_TRS_DAF_231_ControlI32;

typedef enum
{
   NiFpga_TRS_DAF_231_ControlU32_dac0VRegister = 0x8158,
   NiFpga_TRS_DAF_231_ControlU32_nOfBins = 0x8180,
   NiFpga_TRS_DAF_231_ControlU32_scanTrigDelay10ns = 0x8124,
   NiFpga_TRS_DAF_231_ControlU32_stepTrigDelay10ns = 0x8110,
   NiFpga_TRS_DAF_231_ControlU32_trigDelay10ns = 0x8184,
   NiFpga_TRS_DAF_231_ControlU32_waitAfterResetus = 0x81AC,
   NiFpga_TRS_DAF_231_ControlU32_waitForKepcous = 0x81B0,
} NiFpga_TRS_DAF_231_ControlU32;

typedef enum
{
   NiFpga_TRS_DAF_231_TargetToHostFifoU32_transferToHost = 0,
} NiFpga_TRS_DAF_231_TargetToHostFifoU32;

typedef enum
{
   NiFpga_TRS_DAF_231_HostToTargetFifoU32_OutbitsCMD = 1,
} NiFpga_TRS_DAF_231_HostToTargetFifoU32;

#endif
