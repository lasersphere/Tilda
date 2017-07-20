/*
 * Generated with the FPGA Interface C API Generator 14.0.0
 * for NI-RIO 14.0.0 or later.
 */

#ifndef __NiFpga_TRS_DAF_210_7841_h__
#define __NiFpga_TRS_DAF_210_7841_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1400
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_TRS_DAF_210_7841_Bitfile;
 */
#define NiFpga_TRS_DAF_210_7841_Bitfile "NiFpga_TRS_DAF_210_7841.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_TRS_DAF_210_7841_Signature = "126896625D0494B3D643092BB1762951";

typedef enum
{
   NiFpga_TRS_DAF_210_7841_IndicatorBool_DACQuWriteTimeout = 0x8146,
   NiFpga_TRS_DAF_210_7841_IndicatorBool_MCSQuWriteTimeout = 0x814A,
} NiFpga_TRS_DAF_210_7841_IndicatorBool;

typedef enum
{
   NiFpga_TRS_DAF_210_7841_IndicatorI8_MCSerrorcount = 0x8142,
} NiFpga_TRS_DAF_210_7841_IndicatorI8;

typedef enum
{
   NiFpga_TRS_DAF_210_7841_IndicatorU8_postAccOffsetVoltState = 0x813A,
} NiFpga_TRS_DAF_210_7841_IndicatorU8;

typedef enum
{
   NiFpga_TRS_DAF_210_7841_IndicatorU16_MCSstate = 0x8152,
   NiFpga_TRS_DAF_210_7841_IndicatorU16_OutBitsState = 0x811E,
   NiFpga_TRS_DAF_210_7841_IndicatorU16_measVoltState = 0x814E,
   NiFpga_TRS_DAF_210_7841_IndicatorU16_seqState = 0x8176,
} NiFpga_TRS_DAF_210_7841_IndicatorU16;

typedef enum
{
   NiFpga_TRS_DAF_210_7841_IndicatorU32_nOfCmdsOutbit0 = 0x8118,
   NiFpga_TRS_DAF_210_7841_IndicatorU32_nOfCmdsOutbit1 = 0x8114,
   NiFpga_TRS_DAF_210_7841_IndicatorU32_nOfCmdsOutbit2 = 0x8110,
} NiFpga_TRS_DAF_210_7841_IndicatorU32;

typedef enum
{
   NiFpga_TRS_DAF_210_7841_ControlBool_VoltOrScaler = 0x810E,
   NiFpga_TRS_DAF_210_7841_ControlBool_abort = 0x816E,
   NiFpga_TRS_DAF_210_7841_ControlBool_halt = 0x816A,
   NiFpga_TRS_DAF_210_7841_ControlBool_hostConfirmsHzOffsetIsSet = 0x8166,
   NiFpga_TRS_DAF_210_7841_ControlBool_invertScan = 0x8192,
   NiFpga_TRS_DAF_210_7841_ControlBool_pause = 0x8126,
   NiFpga_TRS_DAF_210_7841_ControlBool_softwareTrigger = 0x8122,
   NiFpga_TRS_DAF_210_7841_ControlBool_stopVoltMeas = 0x812A,
   NiFpga_TRS_DAF_210_7841_ControlBool_timedOutWhileHandshake = 0x817A,
} NiFpga_TRS_DAF_210_7841_ControlBool;

typedef enum
{
   NiFpga_TRS_DAF_210_7841_ControlU8_postAccOffsetVoltControl = 0x818E,
   NiFpga_TRS_DAF_210_7841_ControlU8_selectTrigger = 0x8162,
   NiFpga_TRS_DAF_210_7841_ControlU8_triggerEdge = 0x8136,
} NiFpga_TRS_DAF_210_7841_ControlU8;

typedef enum
{
   NiFpga_TRS_DAF_210_7841_ControlU16_cmdByHost = 0x8172,
   NiFpga_TRS_DAF_210_7841_ControlU16_measVoltCompleteDest = 0x812E,
   NiFpga_TRS_DAF_210_7841_ControlU16_triggerTypes = 0x813E,
} NiFpga_TRS_DAF_210_7841_ControlU16;

typedef enum
{
   NiFpga_TRS_DAF_210_7841_ControlI32_dacStartRegister18Bit = 0x819C,
   NiFpga_TRS_DAF_210_7841_ControlI32_dacStepSize18Bit = 0x81A0,
   NiFpga_TRS_DAF_210_7841_ControlI32_measVoltPulseLength25ns = 0x8180,
   NiFpga_TRS_DAF_210_7841_ControlI32_measVoltTimeout10ns = 0x817C,
   NiFpga_TRS_DAF_210_7841_ControlI32_nOfBunches = 0x8154,
   NiFpga_TRS_DAF_210_7841_ControlI32_nOfScans = 0x8194,
   NiFpga_TRS_DAF_210_7841_ControlI32_nOfSteps = 0x8198,
} NiFpga_TRS_DAF_210_7841_ControlI32;

typedef enum
{
   NiFpga_TRS_DAF_210_7841_ControlU32_dac0VRegister = 0x8130,
   NiFpga_TRS_DAF_210_7841_ControlU32_nOfBins = 0x8158,
   NiFpga_TRS_DAF_210_7841_ControlU32_trigDelay10ns = 0x815C,
   NiFpga_TRS_DAF_210_7841_ControlU32_waitAfterResetus = 0x8184,
   NiFpga_TRS_DAF_210_7841_ControlU32_waitForKepcous = 0x8188,
} NiFpga_TRS_DAF_210_7841_ControlU32;

typedef enum
{
   NiFpga_TRS_DAF_210_7841_TargetToHostFifoU32_transferToHost = 0,
} NiFpga_TRS_DAF_210_7841_TargetToHostFifoU32;

typedef enum
{
   NiFpga_TRS_DAF_210_7841_HostToTargetFifoU32_OutbitsCMD = 1,
} NiFpga_TRS_DAF_210_7841_HostToTargetFifoU32;

#endif
