/*
 * wrapper.c
 *
 *  Created on: 20.01.2015
 *      Author: kaufmann
 */

#include <stdio.h>
#include <windows.h>
#include <assert.h>
#include "NiFpga_SPMain.h"

NiFpga_Status status;

NiFpga_Status getStatus()
{
	return status;
}

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

NiFpga_Status _runFPGA(NiFpga_Session session)
	{
		/* run the FPGA application */
		NiFpga_MergeStatus(&status, NiFpga_Run(session, 0));
		return status;
	}

size_t DMAnOfEle(NiFpga_Session session)
{
	size_t nOfEle = 0;
	uint32_t dummy;
	NiFpga_MergeStatus(&status, NiFpga_ReadFifoU32(session,
		NiFpga_SPMain_TargetToHostFifoU32_SimpleCounterDMA,
		&dummy, 0, 10, &nOfEle));
	return nOfEle;
}



uint32_t* readDMA(NiFpga_Session session, size_t nOfEle, uint32_t *vals)
{
	NiFpga_MergeStatus(&status, NiFpga_ReadFifoU32(session,
		NiFpga_SPMain_TargetToHostFifoU32_SimpleCounterDMA,
		vals, nOfEle, 10, NULL));
	return vals;
}

NiFpga_Status _stop(NiFpga_Session session)
{
	NiFpga_MergeStatus(&status, NiFpga_WriteBool(session,
			NiFpga_SPMain_ControlBool_stop,
		0));
	return status;
}


NiFpga_Status _fpgaexit(NiFpga_Session session)
{
	/* close the session now that we're done */
	NiFpga_MergeStatus(&status, NiFpga_Close(session, 0));
	/* must be called after all other calls */
	NiFpga_MergeStatus(&status, NiFpga_Finalize());
	return status;
}

