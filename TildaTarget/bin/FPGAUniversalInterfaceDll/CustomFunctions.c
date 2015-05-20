/*
 * CustomFunctions.c
 *
 *  Created on: 20.05.2015
 *      Author: simkaufm
 */

#include <stddef.h>
#include <stdlib.h>

void freeMemory (size_t *ptr)
{
	free(ptr);
}
