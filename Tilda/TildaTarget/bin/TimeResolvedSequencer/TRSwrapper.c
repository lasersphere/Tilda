/*
 * TRSwrapper.c
 *
 *  Created on: 07.05.2015
 *      Author: simkaufm
 */


#include <stdio.h>
#include <windows.h>
#include <assert.h>
#include "NiFpga_TRS.h"

//status is saved in this c wrapper for now.
NiFpga_Status status;

NiFpga_Status getStatus()
{
	return status;
}

//Necessary FPGA functions
NiFpga_Status init()
	{
		// NiFpga_Status status;
		/* must be called before any other calls */
		NiFpga_Status status = NiFpga_Initialize();
		// outp o = {status, 0, 0 };
		return status;
	}


NiFpga_Session openFPGA()
	{
		NiFpga_Session session;
		/* opens a session, downloads the bitstream, and runs the FPGA */
		NiFpga_MergeStatus(&status, NiFpga_Open(NiFpga_TRS_Bitfile,
			NiFpga_TRS_Signature,
			"Rio1",
			NiFpga_OpenAttribute_NoRun,
			&session));
		return session;
	}

NiFpga_Status runFPGA(NiFpga_Session session)
	{
		/* run the FPGA application */
		NiFpga_MergeStatus(&status, NiFpga_Run(session, 0));
		return status;
	}

NiFpga_Status fpgaexit(NiFpga_Session session)
{
	/* close the session now that we're done */
	NiFpga_MergeStatus(&status, NiFpga_Close(session, 0));
	/* must be called after all other calls */
	NiFpga_MergeStatus(&status, NiFpga_Finalize());
	return status;
}

//User Defined Functions:

/*
 * Function to define the depth of the host sided buffer.
 */
NiFpga_Status confDMA(NiFpga_Session session, size_t nOfEleReq)
{
	uint32_t* dummy;
	size_t dummy2;
	size_t dummy3;
	NiFpga_MergeStatus(&status, NiFpga_AcquireFifoReadElementsU32(session,
			NiFpga_TRS_TargetToHostFifoU32_transferToHost,
			&dummy, nOfEleReq, 10, &dummy2, &dummy3));
	return status;
}

/*
 * Function to read how many elements are in the Host Buffer.
 */
size_t nOfEleInDma(NiFpga_Session session)
{
	size_t nOfEle = 0;
	uint32_t dummy;
	NiFpga_MergeStatus(&status, NiFpga_ReadFifoU32(session,
			NiFpga_TRS_TargetToHostFifoU32_transferToHost,
		&dummy, 0, 10, &nOfEle));
	return nOfEle;
}

/*
 * Function to read a certain number of Elements from the Host side buffer.
 * Writes Result into the position of the pointer passed by vals.
 * session: Handling of Session
 * nOfEle: Number of Elements that will be red
 * *vals: Pointer to the desired location.
 */
NiFpga_Status readDMA(NiFpga_Session session, size_t nOfEle, uint32_t *vals)
{
	NiFpga_MergeStatus(&status, NiFpga_ReadFifoU32(session,
			NiFpga_TRS_TargetToHostFifoU32_transferToHost,
		vals, nOfEle, 10, NULL));
	return status;
}

/*
 * Releases previously acquired FIFO elements.
 *
 * The FPGA target cannot read elements acquired by the host. Therefore, the
 * host must release elements after acquiring them. Always release all acquired
 * elements before closing the session. Do not attempt to access FIFO elements
 * after the elements are released or the session is closed.
 *
 * @param session handle to a currently open session
 * @param fifo FIFO from which to release elements
 * @param elements number of elements to release
 * @return result of the call
 */
NiFpga_Status releaseFifoEle(NiFpga_Session session, size_t nOfEle)
{
	NiFpga_MergeStatus(&status, NiFpga_ReleaseFifoElements(session,
				NiFpga_TRS_TargetToHostFifoU32_transferToHost,
			nOfEle));
	return status;
}

/*
 * Controls:
 *
 *
 */

/*
 * Reading:
 */
NiFpga_Status readBool(NiFpga_Session session, uint32_t indicator)
{
	NiFpga_Bool val;
	NiFpga_MergeStatus(&status, NiFpga_ReadBool(session, indicator, &val));
	return val;
}

NiFpga_Status readI8(NiFpga_Session session, uint32_t indicator)
{
	int8_t val;
	NiFpga_MergeStatus(&status, NiFpga_ReadI8(session, indicator, &val));
	return val;
}

NiFpga_Status readU16(NiFpga_Session session, uint32_t indicator)
{
	uint16_t val;
	NiFpga_MergeStatus(&status, NiFpga_ReadU16(session, indicator, &val));
	return val;
}


/*
 * writing:
 */
NiFpga_Status wBool(NiFpga_Session session, uint32_t control, NiFpga_Bool val)
{
	NiFpga_MergeStatus(&status, NiFpga_WriteBool(session,
			control,
		val));
	return status;
}

NiFpga_Status writeU8(NiFpga_Session session, uint32_t control, uint8_t val)
{
	NiFpga_MergeStatus(&status, NiFpga_WriteU8(session, control, val));
	return status;
}

NiFpga_Status writeU16(NiFpga_Session session, uint32_t control, uint16_t val)
{
	NiFpga_MergeStatus(&status, NiFpga_WriteU16(session, control, val));
	return status;
}

NiFpga_Status writeU32(NiFpga_Session session, uint32_t control, uint32_t val)
{
	NiFpga_MergeStatus(&status, NiFpga_WriteU32(session, control, val));
	return status;
}

NiFpga_Status writeI32(NiFpga_Session session, uint32_t control, int32_t val)
{
	NiFpga_MergeStatus(&status, NiFpga_WriteU32(session, control, val));
	return status;
}

