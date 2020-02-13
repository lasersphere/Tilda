/*
 * Generated with the FPGA Interface C API Generator 14.0.0
 * for NI-RIO 14.0.0 or later.
 */

#ifndef __NiFpga_TRS_DAF_220_h__
#define __NiFpga_TRS_DAF_220_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1400
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_TRS_DAF_220_Bitfile;
 */
#define NiFpga_TRS_DAF_220_Bitfile "NiFpga_TRS_DAF_220.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_TRS_DAF_220_Signature = "BF2C97C9C6E452D9ED5574234F590C09";

typedef enum
{
   NiFpga_TRS_DAF_220_IndicatorBool_DACQuWriteTimeout = 0x815A,
   NiFpga_TRS_DAF_220_IndicatorBool_MCSQuWriteTimeout = 0x815E,
} NiFpga_TRS_DAF_220_IndicatorBool;

typedef enum
{
   NiFpga_TRS_DAF_220_IndicatorI8_MCSerrorcount = 0x8156,
} NiFpga_TRS_DAF_220_IndicatorI8;

typedef enum
{
   NiFpga_TRS_DAF_220_IndicatorU8_postAccOffsetVoltState = 0x814E,
} NiFpga_TRS_DAF_220_IndicatorU8;

typedef enum
{
   NiFpga_TRS_DAF_220_IndicatorU16_MCSstate = 0x8166,
   NiFpga_TRS_DAF_220_IndicatorU16_OutBitsState = 0x8132,
   NiFpga_TRS_DAF_220_IndicatorU16_measVoltState = 0x8162,
   NiFpga_TRS_DAF_220_IndicatorU16_seqState = 0x818A,
} NiFpga_TRS_DAF_220_IndicatorU16;

typedef enum
{
   NiFpga_TRS_DAF_220_IndicatorU32_nOfCmdsOutbit0 = 0x812C,
   NiFpga_TRS_DAF_220_IndicatorU32_nOfCmdsOutbit1 = 0x8128,
   NiFpga_TRS_DAF_220_IndicatorU32_nOfCmdsOutbit2 = 0x8124,
} NiFpga_TRS_DAF_220_IndicatorU32;

typedef enum
{
   NiFpga_TRS_DAF_220_ControlBool_VoltOrScaler = 0x810E,
   NiFpga_TRS_DAF_220_ControlBool_abort = 0x8182,
   NiFpga_TRS_DAF_220_ControlBool_halt = 0x817E,
   NiFpga_TRS_DAF_220_ControlBool_hostConfirmsHzOffsetIsSet = 0x817A,
   NiFpga_TRS_DAF_220_ControlBool_invertScan = 0x81A6,
   NiFpga_TRS_DAF_220_ControlBool_pause = 0x813A,
   NiFpga_TRS_DAF_220_ControlBool_softwareScanTrigger = 0x8116,
   NiFpga_TRS_DAF_220_ControlBool_softwareTrigger = 0x8136,
   NiFpga_TRS_DAF_220_ControlBool_stopVoltMeas = 0x813E,
   NiFpga_TRS_DAF_220_ControlBool_timedOutWhileHandshake = 0x818E,
} NiFpga_TRS_DAF_220_ControlBool;

typedef enum
{
   NiFpga_TRS_DAF_220_ControlU8_postAccOffsetVoltControl = 0x81A2,
   NiFpga_TRS_DAF_220_ControlU8_scanTriggerEdge = 0x811A,
   NiFpga_TRS_DAF_220_ControlU8_selectScanTrigger = 0x8122,
   NiFpga_TRS_DAF_220_ControlU8_selectTrigger = 0x8176,
   NiFpga_TRS_DAF_220_ControlU8_triggerEdge = 0x814A,
} NiFpga_TRS_DAF_220_ControlU8;

typedef enum
{
   NiFpga_TRS_DAF_220_ControlU16_cmdByHost = 0x8186,
   NiFpga_TRS_DAF_220_ControlU16_measVoltCompleteDest = 0x8142,
   NiFpga_TRS_DAF_220_ControlU16_scanTriggerTypes = 0x811E,
   NiFpga_TRS_DAF_220_ControlU16_triggerTypes = 0x8152,
} NiFpga_TRS_DAF_220_ControlU16;

typedef enum
{
   NiFpga_TRS_DAF_220_ControlI32_dacStartRegister18Bit = 0x81B0,
   NiFpga_TRS_DAF_220_ControlI32_dacStepSize18Bit = 0x81B4,
   NiFpga_TRS_DAF_220_ControlI32_measVoltPulseLength25ns = 0x8194,
   NiFpga_TRS_DAF_220_ControlI32_measVoltTimeout10ns = 0x8190,
   NiFpga_TRS_DAF_220_ControlI32_nOfBunches = 0x8168,
   NiFpga_TRS_DAF_220_ControlI32_nOfScans = 0x81A8,
   NiFpga_TRS_DAF_220_ControlI32_nOfSteps = 0x81AC,
} NiFpga_TRS_DAF_220_ControlI32;

typedef enum
{
   NiFpga_TRS_DAF_220_ControlU32_dac0VRegister = 0x8144,
   NiFpga_TRS_DAF_220_ControlU32_nOfBins = 0x816C,
   NiFpga_TRS_DAF_220_ControlU32_scanTrigDelay10ns = 0x8110,
   NiFpga_TRS_DAF_220_ControlU32_trigDelay10ns = 0x8170,
   NiFpga_TRS_DAF_220_ControlU32_waitAfterResetus = 0x8198,
   NiFpga_TRS_DAF_220_ControlU32_waitForKepcous = 0x819C,
} NiFpga_TRS_DAF_220_ControlU32;

typedef enum
{
   NiFpga_TRS_DAF_220_TargetToHostFifoU32_transferToHost = 0,
} NiFpga_TRS_DAF_220_TargetToHostFifoU32;

typedef enum
{
   NiFpga_TRS_DAF_220_HostToTargetFifoU32_OutbitsCMD = 1,
} NiFpga_TRS_DAF_220_HostToTargetFifoU32;

#endif
