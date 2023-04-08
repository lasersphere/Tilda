/*
 * Generated with the FPGA Interface C API Generator 14.0.0
 * for NI-RIO 14.0.0 or later.
 */

#ifndef __NiFpga_ContSeq_7852_h__
#define __NiFpga_ContSeq_7852_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1400
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_ContSeq_7852_Bitfile;
 */
#define NiFpga_ContSeq_7852_Bitfile "NiFpga_ContSeq_7852.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_ContSeq_7852_Signature = "44BFAC9C65B0FD9FF18184F1EC2BB9E0";

typedef enum
{
   NiFpga_ContSeq_7852_IndicatorBool_DACQuWriteTimeout = 0x81D6,
   NiFpga_ContSeq_7852_IndicatorBool_SPCtrQuWriteTimeout = 0x8192,
   NiFpga_ContSeq_7852_IndicatorBool_internalDacAvailable = 0x8112,
} NiFpga_ContSeq_7852_IndicatorBool;

typedef enum
{
   NiFpga_ContSeq_7852_IndicatorU8_SPerrorCount = 0x8186,
   NiFpga_ContSeq_7852_IndicatorU8_postAccOffsetVoltState = 0x817E,
} NiFpga_ContSeq_7852_IndicatorU8;

typedef enum
{
   NiFpga_ContSeq_7852_IndicatorU16_OutBitsState = 0x8156,
   NiFpga_ContSeq_7852_IndicatorU16_SPstate = 0x818A,
   NiFpga_ContSeq_7852_IndicatorU16_measVoltState = 0x81D2,
   NiFpga_ContSeq_7852_IndicatorU16_seqState = 0x81BE,
} NiFpga_ContSeq_7852_IndicatorU16;

typedef enum
{
   NiFpga_ContSeq_7852_IndicatorU32_nOfCmdsOutbit0 = 0x8150,
   NiFpga_ContSeq_7852_IndicatorU32_nOfCmdsOutbit1 = 0x814C,
   NiFpga_ContSeq_7852_IndicatorU32_nOfCmdsOutbit2 = 0x8148,
} NiFpga_ContSeq_7852_IndicatorU32;

typedef enum
{
   NiFpga_ContSeq_7852_ControlBool_VoltOrScaler = 0x810E,
   NiFpga_ContSeq_7852_ControlBool_abort = 0x81C6,
   NiFpga_ContSeq_7852_ControlBool_halt = 0x81CA,
   NiFpga_ContSeq_7852_ControlBool_hostConfirmsHzOffsetIsSet = 0x81CE,
   NiFpga_ContSeq_7852_ControlBool_invertScan = 0x81A6,
   NiFpga_ContSeq_7852_ControlBool_pause = 0x815E,
   NiFpga_ContSeq_7852_ControlBool_scanDevSet = 0x811A,
   NiFpga_ContSeq_7852_ControlBool_softwareScanTrigger = 0x8136,
   NiFpga_ContSeq_7852_ControlBool_softwareStepTrigger = 0x8126,
   NiFpga_ContSeq_7852_ControlBool_softwareTrigger = 0x815A,
   NiFpga_ContSeq_7852_ControlBool_stopVoltMeas = 0x8162,
   NiFpga_ContSeq_7852_ControlBool_timedOutWhileHandshake = 0x81BA,
} NiFpga_ContSeq_7852_ControlBool;

typedef enum
{
   NiFpga_ContSeq_7852_ControlU8_postAccOffsetVoltControl = 0x8182,
   NiFpga_ContSeq_7852_ControlU8_scanTriggerEdge = 0x813A,
   NiFpga_ContSeq_7852_ControlU8_selectScanTrigger = 0x8142,
   NiFpga_ContSeq_7852_ControlU8_selectStepTrigger = 0x8132,
   NiFpga_ContSeq_7852_ControlU8_selectTrigger = 0x8176,
   NiFpga_ContSeq_7852_ControlU8_stepTriggerEdge = 0x812A,
   NiFpga_ContSeq_7852_ControlU8_triggerEdge = 0x816E,
} NiFpga_ContSeq_7852_ControlU8;

typedef enum
{
   NiFpga_ContSeq_7852_ControlU16_ScanDevice = 0x8116,
   NiFpga_ContSeq_7852_ControlU16_cmdByHost = 0x81C2,
   NiFpga_ContSeq_7852_ControlU16_measVoltCompleteDest = 0x8166,
   NiFpga_ContSeq_7852_ControlU16_scanTriggerTypes = 0x8146,
   NiFpga_ContSeq_7852_ControlU16_stepTriggerTypes = 0x812E,
   NiFpga_ContSeq_7852_ControlU16_triggerTypes = 0x817A,
} NiFpga_ContSeq_7852_ControlU16;

typedef enum
{
   NiFpga_ContSeq_7852_ControlI32_dacStartRegister20Bit = 0x8198,
   NiFpga_ContSeq_7852_ControlI32_dacStepSize20Bit = 0x8194,
   NiFpga_ContSeq_7852_ControlI32_measVoltPulseLength25ns = 0x81B0,
   NiFpga_ContSeq_7852_ControlI32_measVoltTimeout10ns = 0x81B4,
   NiFpga_ContSeq_7852_ControlI32_nOfScans = 0x81A0,
   NiFpga_ContSeq_7852_ControlI32_nOfSteps = 0x819C,
   NiFpga_ContSeq_7852_ControlI32_scanDevTimeout10ns = 0x811C,
} NiFpga_ContSeq_7852_ControlI32;

typedef enum
{
   NiFpga_ContSeq_7852_ControlU32_dac0VRegister = 0x8168,
   NiFpga_ContSeq_7852_ControlU32_dwellTime10ns = 0x818C,
   NiFpga_ContSeq_7852_ControlU32_scanTrigDelay10ns = 0x813C,
   NiFpga_ContSeq_7852_ControlU32_stepTrigDelay10ns = 0x8120,
   NiFpga_ContSeq_7852_ControlU32_trigDelay10ns = 0x8170,
   NiFpga_ContSeq_7852_ControlU32_waitAfterResetus = 0x81AC,
   NiFpga_ContSeq_7852_ControlU32_waitForKepcous = 0x81A8,
} NiFpga_ContSeq_7852_ControlU32;

typedef enum
{
   NiFpga_ContSeq_7852_TargetToHostFifoU32_transferToHost = 0,
} NiFpga_ContSeq_7852_TargetToHostFifoU32;

typedef enum
{
   NiFpga_ContSeq_7852_HostToTargetFifoU32_OutbitsCMD = 1,
} NiFpga_ContSeq_7852_HostToTargetFifoU32;

#endif
