/*
 * Generated with the FPGA Interface C API Generator 14.0.0
 * for NI-RIO 14.0.0 or later.
 */

#ifndef __NiFpga_ContSeqV210_h__
#define __NiFpga_ContSeqV210_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1400
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_ContSeqV210_Bitfile;
 */
#define NiFpga_ContSeqV210_Bitfile "NiFpga_ContSeqV210.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_ContSeqV210_Signature = "33CB04F913691A16344BAD2C297EF27C";

typedef enum
{
   NiFpga_ContSeqV210_IndicatorBool_DACQuWriteTimeout = 0x819E,
   NiFpga_ContSeqV210_IndicatorBool_SPCtrQuWriteTimeout = 0x815A,
} NiFpga_ContSeqV210_IndicatorBool;

typedef enum
{
   NiFpga_ContSeqV210_IndicatorU8_SPerrorCount = 0x814E,
   NiFpga_ContSeqV210_IndicatorU8_postAccOffsetVoltState = 0x8146,
} NiFpga_ContSeqV210_IndicatorU8;

typedef enum
{
   NiFpga_ContSeqV210_IndicatorU16_OutBitsState = 0x811E,
   NiFpga_ContSeqV210_IndicatorU16_SPstate = 0x8152,
   NiFpga_ContSeqV210_IndicatorU16_measVoltState = 0x819A,
   NiFpga_ContSeqV210_IndicatorU16_seqState = 0x8186,
} NiFpga_ContSeqV210_IndicatorU16;

typedef enum
{
   NiFpga_ContSeqV210_IndicatorU32_nOfCmdsOutbit0 = 0x8118,
   NiFpga_ContSeqV210_IndicatorU32_nOfCmdsOutbit02 = 0x8110,
   NiFpga_ContSeqV210_IndicatorU32_nOfCmdsOutbit1 = 0x8114,
} NiFpga_ContSeqV210_IndicatorU32;

typedef enum
{
   NiFpga_ContSeqV210_ControlBool_VoltOrScaler = 0x810E,
   NiFpga_ContSeqV210_ControlBool_abort = 0x818E,
   NiFpga_ContSeqV210_ControlBool_halt = 0x8192,
   NiFpga_ContSeqV210_ControlBool_hostConfirmsHzOffsetIsSet = 0x8196,
   NiFpga_ContSeqV210_ControlBool_invertScan = 0x816E,
   NiFpga_ContSeqV210_ControlBool_pause = 0x8126,
   NiFpga_ContSeqV210_ControlBool_softwareTrigger = 0x8122,
   NiFpga_ContSeqV210_ControlBool_stopVoltMeas = 0x812A,
   NiFpga_ContSeqV210_ControlBool_timedOutWhileHandshake = 0x8182,
} NiFpga_ContSeqV210_ControlBool;

typedef enum
{
   NiFpga_ContSeqV210_ControlU8_postAccOffsetVoltControl = 0x814A,
   NiFpga_ContSeqV210_ControlU8_selectTrigger = 0x813E,
   NiFpga_ContSeqV210_ControlU8_triggerEdge = 0x8136,
} NiFpga_ContSeqV210_ControlU8;

typedef enum
{
   NiFpga_ContSeqV210_ControlU16_cmdByHost = 0x818A,
   NiFpga_ContSeqV210_ControlU16_measVoltCompleteDest = 0x812E,
   NiFpga_ContSeqV210_ControlU16_triggerTypes = 0x8142,
} NiFpga_ContSeqV210_ControlU16;

typedef enum
{
   NiFpga_ContSeqV210_ControlI32_dacStartRegister18Bit = 0x8160,
   NiFpga_ContSeqV210_ControlI32_dacStepSize18Bit = 0x815C,
   NiFpga_ContSeqV210_ControlI32_measVoltPulseLength25ns = 0x8178,
   NiFpga_ContSeqV210_ControlI32_measVoltTimeout10ns = 0x817C,
   NiFpga_ContSeqV210_ControlI32_nOfScans = 0x8168,
   NiFpga_ContSeqV210_ControlI32_nOfSteps = 0x8164,
} NiFpga_ContSeqV210_ControlI32;

typedef enum
{
   NiFpga_ContSeqV210_ControlU32_dac0VRegister = 0x8130,
   NiFpga_ContSeqV210_ControlU32_dwellTime10ns = 0x8154,
   NiFpga_ContSeqV210_ControlU32_trigDelay10ns = 0x8138,
   NiFpga_ContSeqV210_ControlU32_waitAfterResetus = 0x8174,
   NiFpga_ContSeqV210_ControlU32_waitForKepcous = 0x8170,
} NiFpga_ContSeqV210_ControlU32;

typedef enum
{
   NiFpga_ContSeqV210_TargetToHostFifoU32_transferToHost = 0,
} NiFpga_ContSeqV210_TargetToHostFifoU32;

typedef enum
{
   NiFpga_ContSeqV210_HostToTargetFifoU32_OutbitsCMD = 1,
} NiFpga_ContSeqV210_HostToTargetFifoU32;

#endif
