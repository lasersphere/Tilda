/*
 * Generated with the FPGA Interface C API Generator 14.0.0
 * for NI-RIO 14.0.0 or later.
 */

#ifndef __NiFpga_ContSeqV230_h__
#define __NiFpga_ContSeqV230_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1400
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_ContSeqV230_Bitfile;
 */
#define NiFpga_ContSeqV230_Bitfile "NiFpga_ContSeqV230.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_ContSeqV230_Signature = "54009A347C6CF0ED7B420076A8636E0E";

typedef enum
{
   NiFpga_ContSeqV230_IndicatorBool_DACQuWriteTimeout = 0x81C6,
   NiFpga_ContSeqV230_IndicatorBool_SPCtrQuWriteTimeout = 0x8182,
} NiFpga_ContSeqV230_IndicatorBool;

typedef enum
{
   NiFpga_ContSeqV230_IndicatorU8_SPerrorCount = 0x8176,
   NiFpga_ContSeqV230_IndicatorU8_postAccOffsetVoltState = 0x816E,
} NiFpga_ContSeqV230_IndicatorU8;

typedef enum
{
   NiFpga_ContSeqV230_IndicatorU16_OutBitsState = 0x8146,
   NiFpga_ContSeqV230_IndicatorU16_SPstate = 0x817A,
   NiFpga_ContSeqV230_IndicatorU16_measVoltState = 0x81C2,
   NiFpga_ContSeqV230_IndicatorU16_seqState = 0x81AE,
} NiFpga_ContSeqV230_IndicatorU16;

typedef enum
{
   NiFpga_ContSeqV230_IndicatorU32_nOfCmdsOutbit0 = 0x8140,
   NiFpga_ContSeqV230_IndicatorU32_nOfCmdsOutbit1 = 0x813C,
   NiFpga_ContSeqV230_IndicatorU32_nOfCmdsOutbit2 = 0x8138,
} NiFpga_ContSeqV230_IndicatorU32;

typedef enum
{
   NiFpga_ContSeqV230_ControlBool_VoltOrScaler = 0x810E,
   NiFpga_ContSeqV230_ControlBool_abort = 0x81B6,
   NiFpga_ContSeqV230_ControlBool_halt = 0x81BA,
   NiFpga_ContSeqV230_ControlBool_hostConfirmsHzOffsetIsSet = 0x81BE,
   NiFpga_ContSeqV230_ControlBool_invertScan = 0x8196,
   NiFpga_ContSeqV230_ControlBool_pause = 0x814E,
   NiFpga_ContSeqV230_ControlBool_softwareStepTrigger = 0x8116,
   NiFpga_ContSeqV230_ControlBool_softwareTrigger = 0x814A,
   NiFpga_ContSeqV230_ControlBool_softwarescanTrigger = 0x8126,
   NiFpga_ContSeqV230_ControlBool_stopVoltMeas = 0x8152,
   NiFpga_ContSeqV230_ControlBool_timedOutWhileHandshake = 0x81AA,
} NiFpga_ContSeqV230_ControlBool;

typedef enum
{
   NiFpga_ContSeqV230_ControlU8_postAccOffsetVoltControl = 0x8172,
   NiFpga_ContSeqV230_ControlU8_scanTriggerEdge = 0x812A,
   NiFpga_ContSeqV230_ControlU8_selectScanTrigger = 0x8132,
   NiFpga_ContSeqV230_ControlU8_selectStepTrigger = 0x8122,
   NiFpga_ContSeqV230_ControlU8_selectTrigger = 0x8166,
   NiFpga_ContSeqV230_ControlU8_stepTriggerEdge = 0x811A,
   NiFpga_ContSeqV230_ControlU8_triggerEdge = 0x815E,
} NiFpga_ContSeqV230_ControlU8;

typedef enum
{
   NiFpga_ContSeqV230_ControlU16_cmdByHost = 0x81B2,
   NiFpga_ContSeqV230_ControlU16_measVoltCompleteDest = 0x8156,
   NiFpga_ContSeqV230_ControlU16_scanTriggerTypes = 0x8136,
   NiFpga_ContSeqV230_ControlU16_stepTriggerTypes = 0x811E,
   NiFpga_ContSeqV230_ControlU16_triggerTypes = 0x816A,
} NiFpga_ContSeqV230_ControlU16;

typedef enum
{
   NiFpga_ContSeqV230_ControlI32_dacStartRegister18Bit = 0x8188,
   NiFpga_ContSeqV230_ControlI32_dacStepSize18Bit = 0x8184,
   NiFpga_ContSeqV230_ControlI32_measVoltPulseLength25ns = 0x81A0,
   NiFpga_ContSeqV230_ControlI32_measVoltTimeout10ns = 0x81A4,
   NiFpga_ContSeqV230_ControlI32_nOfScans = 0x8190,
   NiFpga_ContSeqV230_ControlI32_nOfSteps = 0x818C,
} NiFpga_ContSeqV230_ControlI32;

typedef enum
{
   NiFpga_ContSeqV230_ControlU32_dac0VRegister = 0x8158,
   NiFpga_ContSeqV230_ControlU32_dwellTime10ns = 0x817C,
   NiFpga_ContSeqV230_ControlU32_scanTrigDelay10ns = 0x812C,
   NiFpga_ContSeqV230_ControlU32_stepTrigDelay10ns = 0x8110,
   NiFpga_ContSeqV230_ControlU32_trigDelay10ns = 0x8160,
   NiFpga_ContSeqV230_ControlU32_waitAfterResetus = 0x819C,
   NiFpga_ContSeqV230_ControlU32_waitForKepcous = 0x8198,
} NiFpga_ContSeqV230_ControlU32;

typedef enum
{
   NiFpga_ContSeqV230_TargetToHostFifoU32_transferToHost = 0,
} NiFpga_ContSeqV230_TargetToHostFifoU32;

typedef enum
{
   NiFpga_ContSeqV230_HostToTargetFifoU32_OutbitsCMD = 1,
} NiFpga_ContSeqV230_HostToTargetFifoU32;

#endif
