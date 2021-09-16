QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets printsupport


CONFIG += c++17
CONFIG  += sanitizer sanitize_address

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    flatbuffer.cpp \
    ipcclient.cpp \
    main.cpp \
    mainwindow.cpp \
    picosettings.cpp \
    processpico.cpp \
    qcustomplot.cpp \
    runpico.cpp

HEADERS += \
    flatbuffer.h \
    ipcclient.h \
    mainwindow.h \
    picosettings.h \
    processpico.h \
    qcustomplot.h \
    runpico.h

FORMS += \
    mainwindow.ui

win32:CONFIG(release, debug|release): LIBS += -L$$PWD -lps6000
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD -lps6000
else:unix: LIBS += -L$$PWD/../Resources/lib/ -lps6000.2

INCLUDEPATH += $$PWD
DEPENDPATH += $$PWD


# Default rules for deployment.
#qnx: target.path = /tmp/$${TARGET}/bin
#else: unix:!android: target.path = /opt/$${TARGET}/bin
#!isEmpty(target.path): INSTALLS += target
