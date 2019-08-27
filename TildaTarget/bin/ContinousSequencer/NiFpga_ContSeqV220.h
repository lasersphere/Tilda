/*
 * Generated with the FPGA Interface C API Generator 14.0.0
 * for NI-RIO 14.0.0 or later.
 */

#ifndef __NiFpga_ContSeqV220_h__
#define __NiFpga_ContSeqV220_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1400
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_ContSeqV220_Bitfile;
 */
#define NiFpga_ContSeqV220_Bitfile "NiFpga_ContSeqV220.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_ContSeqV220_Signature = "50181A46831391680B489E44F3DD9313";

typedef enum
{
   NiFpga_ContSeqV220_IndicatorBool_DACQuWriteTimeout = 0x81B2,
   NiFpga_ContSeqV220_IndicatorBool_SPCtrQuWriteTimeout = 0x816E,
} NiFpga_ContSeqV220_IndicatorBool;

typedef enum
{
   NiFpga_ContSeqV220_IndicatorU8_SPerrorCount = 0x8162,
   NiFpga_ContSeqV220_IndicatorU8_postAccOffsetVoltState = 0x815A,
} NiFpga_ContSeqV220_IndicatorU8;

typedef enum
{
   NiFpga_ContSeqV220_IndicatorU16_OutBitsState = 0x8132,
   NiFpga_ContSeqV220_IndicatorU16_SPstate = 0x8166,
   NiFpga_ContSeqV220_IndicatorU16_measVoltState = 0x81AE,
   NiFpga_ContSeqV220_IndicatorU16_seqState = 0x819A,
} NiFpga_ContSeqV220_IndicatorU16;

typedef enum
{
   NiFpga_ContSeqV220_IndicatorU32_nOfCmdsOutbit0 = 0x812C,
   NiFpga_ContSeqV220_IndicatorU32_nOfCmdsOutbit1 = 0x8128,
   NiFpga_ContSeqV220_IndicatorU32_nOfCmdsOutbit2 = 0x8124,
} NiFpga_ContSeqV220_IndicatorU32;

typedef enum
{
   NiFpga_ContSeqV220_ControlBool_VoltOrScaler = 0x810E,
   NiFpga_ContSeqV220_ControlBool_abort = 0x81A2,
   NiFpga_ContSeqV220_ControlBool_halt = 0x81A6,
   NiFpga_ContSeqV220_ControlBool_hostConfirmsHzOffsetIsSet = 0x81AA,
   NiFpga_ContSeqV220_ControlBool_invertScan = 0x8182,
   NiFpga_ContSeqV220_ControlBool_pause = 0x813A,
   NiFpga_ContSeqV220_ControlBool_softwareTrigger = 0x8136,
   NiFpga_ContSeqV220_ControlBool_softwarescanTrigger = 0x8112,
   NiFpga_ContSeqV220_ControlBool_stopVoltMeas = 0x813E,
   NiFpga_ContSeqV220_ControlBool_timedOutWhileHandshake = 0x8196,
} NiFpga_ContSeqV220_ControlBool;

typedef enum
{
   NiFpga_ContSeqV220_ControlU8_postAccOffsetVoltControl = 0x815E,
   NiFpga_ContSeqV220_ControlU8_scanTriggerEdge = 0x8116,
   NiFpga_ContSeqV220_ControlU8_selectScanTrigger = 0x811E,
   NiFpga_ContSeqV220_ControlU8_selectTrigger = 0x8152,
   NiFpga_ContSeqV220_ControlU8_triggerEdge = 0x814A,
} NiFpga_ContSeqV220_ControlU8;

typedef enum
{
   NiFpga_ContSeqV220_ControlU16_cmdByHost = 0x819E,
   NiFpga_ContSeqV220_ControlU16_measVoltCompleteDest = 0x8142,
   NiFpga_ContSeqV220_ControlU16_scanTriggerTypes = 0x8122,
   NiFpga_ContSeqV220_ControlU16_triggerTypes = 0x8156,
} NiFpga_ContSeqV220_ControlU16;

typedef enum
{
   NiFpga_ContSeqV220_ControlI32_dacStartRegister18Bit = 0x8174,
   NiFpga_ContSeqV220_ControlI32_dacStepSize18Bit = 0x8170,
   NiFpga_ContSeqV220_ControlI32_measVoltPulseLength25ns = 0x818C,
   NiFpga_ContSeqV220_ControlI32_measVoltTimeout10ns = 0x8190,
   NiFpga_ContSeqV220_ControlI32_nOfScans = 0x817C,
   NiFpga_ContSeqV220_ControlI32_nOfSteps = 0x8178,
} NiFpga_ContSeqV220_ControlI32;

typedef enum
{
   NiFpga_ContSeqV220_ControlU32_dac0VRegister = 0x8144,
   NiFpga_ContSeqV220_ControlU32_dwellTime10ns = 0x8168,
   NiFpga_ContSeqV220_ControlU32_scanTrigDelay10ns = 0x8118,
   NiFpga_ContSeqV220_ControlU32_trigDelay10ns = 0x814C,
   NiFpga_ContSeqV220_ControlU32_waitAfterResetus = 0x8188,
   NiFpga_ContSeqV220_ControlU32_waitForKepcous = 0x8184,
} NiFpga_ContSeqV220_ControlU32;

typedef enum
{
   NiFpga_ContSeqV220_TargetToHostFifoU32_transferToHost = 0,
} NiFpga_ContSeqV220_TargetToHostFifoU32;

typedef enum
{
   NiFpga_ContSeqV220_HostToTargetFifoU32_OutbitsCMD = 1,
} NiFpga_ContSeqV220_HostToTargetFifoU32;

#endif
