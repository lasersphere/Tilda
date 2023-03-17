/*
 * Generated with the FPGA Interface C API Generator 14.0.0
 * for NI-RIO 14.0.0 or later.
 */

#ifndef __NiFpga_HeinzingerAndKepcoTest_h__
#define __NiFpga_HeinzingerAndKepcoTest_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1400
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_HeinzingerAndKepcoTest_Bitfile;
 */
#define NiFpga_HeinzingerAndKepcoTest_Bitfile "NiFpga_HeinzingerAndKepcoTest.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_HeinzingerAndKepcoTest_Signature = "65983986BDA85857BB806DE8DBC8A8B6";

typedef enum
{
   NiFpga_HeinzingerAndKepcoTest_IndicatorU16_DacState = 0x8112,
} NiFpga_HeinzingerAndKepcoTest_IndicatorU16;

typedef enum
{
   NiFpga_HeinzingerAndKepcoTest_IndicatorU32_actDACRegister = 0x8114,
} NiFpga_HeinzingerAndKepcoTest_IndicatorU32;

typedef enum
{
   NiFpga_HeinzingerAndKepcoTest_ControlU8_heinzingerControl = 0x811E,
} NiFpga_HeinzingerAndKepcoTest_ControlU8;

typedef enum
{
   NiFpga_HeinzingerAndKepcoTest_ControlU16_DacStateCmdByHost = 0x811A,
} NiFpga_HeinzingerAndKepcoTest_ControlU16;

typedef enum
{
   NiFpga_HeinzingerAndKepcoTest_ControlU32_setDACRegister = 0x810C,
} NiFpga_HeinzingerAndKepcoTest_ControlU32;

#endif
