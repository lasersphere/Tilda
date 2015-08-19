/*
 * Generated with the FPGA Interface C API Generator 14.0.0
 * for NI-RIO 14.0.0 or later.
 */

#ifndef __NiFpga_HeinzingerAndKepcoTest_v102_h__
#define __NiFpga_HeinzingerAndKepcoTest_v102_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1400
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_HeinzingerAndKepcoTest_v102_Bitfile;
 */
#define NiFpga_HeinzingerAndKepcoTest_v102_Bitfile "NiFpga_HeinzingerAndKepcoTest_v102.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_HeinzingerAndKepcoTest_v102_Signature = "07925F2E9ECF4BF922BD7AEDD0A076D3";

typedef enum
{
   NiFpga_HeinzingerAndKepcoTest_v102_IndicatorU8_postAccOffsetVoltState = 0x8112,
} NiFpga_HeinzingerAndKepcoTest_v102_IndicatorU8;

typedef enum
{
   NiFpga_HeinzingerAndKepcoTest_v102_IndicatorU16_DacState = 0x811A,
} NiFpga_HeinzingerAndKepcoTest_v102_IndicatorU16;

typedef enum
{
   NiFpga_HeinzingerAndKepcoTest_v102_IndicatorU32_actDACRegister = 0x811C,
} NiFpga_HeinzingerAndKepcoTest_v102_IndicatorU32;

typedef enum
{
   NiFpga_HeinzingerAndKepcoTest_v102_ControlU8_postAccOffsetVoltControl = 0x8116,
} NiFpga_HeinzingerAndKepcoTest_v102_ControlU8;

typedef enum
{
   NiFpga_HeinzingerAndKepcoTest_v102_ControlU16_DacStateCmdByHost = 0x8122,
} NiFpga_HeinzingerAndKepcoTest_v102_ControlU16;

typedef enum
{
   NiFpga_HeinzingerAndKepcoTest_v102_ControlU32_setDACRegister = 0x810C,
} NiFpga_HeinzingerAndKepcoTest_v102_ControlU32;

#endif
