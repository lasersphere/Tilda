#ifndef TESTDLL_GLOBAL_H
#define TESTDLL_GLOBAL_H

#include <QtCore/qglobal.h>

#if defined(TESTDLL_LIBRARY)
#  define TESTDLL_EXPORT Q_DECL_EXPORT
#else
#  define TESTDLL_EXPORT Q_DECL_IMPORT
#endif

#endif // TESTDLL_GLOBAL_H
