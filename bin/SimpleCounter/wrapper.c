/*
 * wrapper.c
 *
 *  Created on: 20.01.2015
 *      Author: noertert
 */


/*
* FPGA Interface C API example for Microsoft Visual C++ 2008 or later for
* computers running Microsoft Windows.
*
* NOTE: In order to run this example, you must compile a LabVIEW FPGA bitfile
*       and generate a C API for it. For more information about using this
*       example, refer to the Examples topic of the FPGA Interface C API Help,
*       located under
*       Start>>All Programs>>National Instruments>>FPGA Interface C API.
*/


#include <stdio.h>
#include <windows.h>
#include <assert.h>
#include "NiFpga_SPMain.h"

NiFpga_Status status;

NiFpga_Status _init()
	{
		// NiFpga_Status status;
		/* must be called before any other calls */
		NiFpga_Status status = NiFpga_Initialize();
		// outp o = {status, 0, 0 };
		return status;
	}


NiFpga_Session _openFPGA()
{
	NiFpga_Session session;
	/* opens a session, downloads the bitstream, and runs the FPGA */
	NiFpga_MergeStatus(&status, NiFpga_Open(NiFpga_SPMain_Bitfile,
			NiFpga_SPMain_Signature,
		"RIO1",
		NiFpga_OpenAttribute_NoRun,
		&session));
	return session;
}
/*

extern "C" { __declspec(dllexport) NiFpga_Status _runFPGA(NiFpga_Session session)
	{
	/* run the FPGA application
	NiFpga_MergeStatus(&status, NiFpga_Run(session, 0));
	// outp o = {status, session, 0 };
	return status;
}
}

extern "C" { __declspec(dllexport) NiFpga_Status _writeDMA(NiFpga_Session session)
{
	NiFpga_MergeStatus(&status, NiFpga_WriteBool(session,
		NiFpga_DMABSP_ControlBool_LoopGo,
		0));
	NiFpga_MergeStatus(&status, NiFpga_WriteBool(session,
		NiFpga_DMABSP_ControlBool_Write,
		1));
	return status;
}
}

extern "C" { __declspec(dllexport) NiFpga_Status _LoopGo(NiFpga_Session session)
{
	NiFpga_MergeStatus(&status, NiFpga_WriteBool(session,
		NiFpga_DMABSP_ControlBool_Write,
		0));
	NiFpga_MergeStatus(&status, NiFpga_WriteBool(session,
		NiFpga_DMABSP_ControlBool_LoopGo,
		1));
	return status;
}
}

extern "C" { __declspec(dllexport) int32_t _measTime(NiFpga_Session session)
{
	int32_t messZeit = 0;
	NiFpga_MergeStatus(&status, NiFpga_ReadI32(session,
		NiFpga_DMABSP_IndicatorI32_measTime,
		&messZeit));
	return messZeit;
}
}


extern "C" { __declspec(dllexport) uint32_t _actCts(NiFpga_Session session)
{
	uint32_t actCts = 0;
	NiFpga_MergeStatus(&status, NiFpga_ReadU32(session,
		NiFpga_DMABSP_IndicatorU32_cts,
		&actCts));
	return actCts;
}
}


extern "C" { __declspec(dllexport) uint32_t _readDMA(NiFpga_Session session, int nofEle)
{
	uint32_t dmaCts = 0;
	NiFpga_MergeStatus(&status, NiFpga_ReadFifoU32(session,
		NiFpga_DMABSP_TargetToHostFifoU32_DMAQueue,
		&dmaCts, nofEle, 10, NULL));
	return dmaCts;
}
}

extern "C" { __declspec(dllexport)  int _rate(NiFpga_Session session, int rat)
{
	/* set the toggle rate to half a second
	NiFpga_MergeStatus(&status, NiFpga_WriteU32(session,
		NiFpga_DMABSP_ControlU32_WaitMilliseconds,
		rat));
	return rat);
}
}

extern "C" { __declspec(dllexport) NiFpga_Status _fpgaexit(NiFpga_Session session)
{
	/* close the session now that we're done
	NiFpga_MergeStatus(&status, NiFpga_Close(session, 0));
	/* must be called after all other calls
	NiFpga_MergeStatus(&status, NiFpga_Finalize());
	return status;
}
}
*/
