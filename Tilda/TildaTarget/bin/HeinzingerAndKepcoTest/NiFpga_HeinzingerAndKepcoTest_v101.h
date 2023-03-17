/*
 * Generated with the FPGA Interface C API Generator 14.0.0
 * for NI-RIO 14.0.0 or later.
 */

#ifndef __NiFpga_HeinzingerAndKepcoTest_v101_h__
#define __NiFpga_HeinzingerAndKepcoTest_v101_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1400
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_HeinzingerAndKepcoTest_v101_Bitfile;
 */
#define NiFpga_HeinzingerAndKepcoTest_v101_Bitfile "NiFpga_HeinzingerAndKepcoTest_v101.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_HeinzingerAndKepcoTest_v101_Signature = "2702394610028D5D0D33227747280129";

typedef enum
{
   NiFpga_HeinzingerAndKepcoTest_v101_IndicatorU16_DacState = 0x8112,
} NiFpga_HeinzingerAndKepcoTest_v101_IndicatorU16;

typedef enum
{
   NiFpga_HeinzingerAndKepcoTest_v101_IndicatorU32_actDACRegister = 0x8114,
} NiFpga_HeinzingerAndKepcoTest_v101_IndicatorU32;

typedef enum
{
   NiFpga_HeinzingerAndKepcoTest_v101_ControlU8_heinzingerControl = 0x811E,
} NiFpga_HeinzingerAndKepcoTest_v101_ControlU8;

typedef enum
{
   NiFpga_HeinzingerAndKepcoTest_v101_ControlU16_DacStateCmdByHost = 0x811A,
} NiFpga_HeinzingerAndKepcoTest_v101_ControlU16;

typedef enum
{
   NiFpga_HeinzingerAndKepcoTest_v101_ControlU32_setDACRegister = 0x810C,
} NiFpga_HeinzingerAndKepcoTest_v101_ControlU32;

#endif
