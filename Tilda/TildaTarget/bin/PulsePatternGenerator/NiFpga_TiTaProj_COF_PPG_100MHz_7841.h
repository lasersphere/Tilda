/*
 * Generated with the FPGA Interface C API Generator 14.0.0
 * for NI-RIO 14.0.0 or later.
 */

#ifndef __NiFpga_TiTaProj_COF_PPG_100MHz_7841_h__
#define __NiFpga_TiTaProj_COF_PPG_100MHz_7841_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1400
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_TiTaProj_COF_PPG_100MHz_7841_Bitfile;
 */
#define NiFpga_TiTaProj_COF_PPG_100MHz_7841_Bitfile "NiFpga_TiTaProj_COF_PPG_100MHz_7841.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_TiTaProj_COF_PPG_100MHz_7841_Signature = "8FA65434BD550F91DD225CBF732B47DB";

typedef enum
{
   NiFpga_TiTaProj_COF_PPG_100MHz_7841_IndicatorBool_fifo_empty = 0x814E,
   NiFpga_TiTaProj_COF_PPG_100MHz_7841_IndicatorBool_start_sctl = 0x810E,
   NiFpga_TiTaProj_COF_PPG_100MHz_7841_IndicatorBool_stop_sctl = 0x8156,
} NiFpga_TiTaProj_COF_PPG_100MHz_7841_IndicatorBool;

typedef enum
{
   NiFpga_TiTaProj_COF_PPG_100MHz_7841_IndicatorU8_error_code = 0x813A,
} NiFpga_TiTaProj_COF_PPG_100MHz_7841_IndicatorU8;

typedef enum
{
   NiFpga_TiTaProj_COF_PPG_100MHz_7841_IndicatorU16_state = 0x813E,
} NiFpga_TiTaProj_COF_PPG_100MHz_7841_IndicatorU16;

typedef enum
{
   NiFpga_TiTaProj_COF_PPG_100MHz_7841_IndicatorI32_elements_loaded = 0x8120,
   NiFpga_TiTaProj_COF_PPG_100MHz_7841_IndicatorI32_revision = 0x8118,
   NiFpga_TiTaProj_COF_PPG_100MHz_7841_IndicatorI32_ticks_per_us = 0x8114,
} NiFpga_TiTaProj_COF_PPG_100MHz_7841_IndicatorI32;

typedef enum
{
   NiFpga_TiTaProj_COF_PPG_100MHz_7841_IndicatorU32_number_of_cmds = 0x8140,
   NiFpga_TiTaProj_COF_PPG_100MHz_7841_IndicatorU32_stop_addr = 0x8144,
} NiFpga_TiTaProj_COF_PPG_100MHz_7841_IndicatorU32;

typedef enum
{
   NiFpga_TiTaProj_COF_PPG_100MHz_7841_ControlBool_continuous = 0x8152,
   NiFpga_TiTaProj_COF_PPG_100MHz_7841_ControlBool_load = 0x812E,
   NiFpga_TiTaProj_COF_PPG_100MHz_7841_ControlBool_query = 0x811E,
   NiFpga_TiTaProj_COF_PPG_100MHz_7841_ControlBool_replace = 0x8126,
   NiFpga_TiTaProj_COF_PPG_100MHz_7841_ControlBool_reset = 0x8132,
   NiFpga_TiTaProj_COF_PPG_100MHz_7841_ControlBool_run = 0x812A,
   NiFpga_TiTaProj_COF_PPG_100MHz_7841_ControlBool_stop = 0x814A,
   NiFpga_TiTaProj_COF_PPG_100MHz_7841_ControlBool_useJump = 0x8112,
} NiFpga_TiTaProj_COF_PPG_100MHz_7841_ControlBool;

typedef enum
{
   NiFpga_TiTaProj_COF_PPG_100MHz_7841_ControlU32_mem_addr = 0x8134,
} NiFpga_TiTaProj_COF_PPG_100MHz_7841_ControlU32;

typedef enum
{
   NiFpga_TiTaProj_COF_PPG_100MHz_7841_TargetToHostFifoU32_DMA_down = 1,
} NiFpga_TiTaProj_COF_PPG_100MHz_7841_TargetToHostFifoU32;

typedef enum
{
   NiFpga_TiTaProj_COF_PPG_100MHz_7841_HostToTargetFifoU32_DMA_up = 0,
} NiFpga_TiTaProj_COF_PPG_100MHz_7841_HostToTargetFifoU32;

#endif
