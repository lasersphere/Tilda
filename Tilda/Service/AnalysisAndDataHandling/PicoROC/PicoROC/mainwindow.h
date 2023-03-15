#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "qcustomplot.h"
#include "picosettings.h"
#include "runpico.h"
#include "ipcclient.h"


QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onUpdateSettings();

    void on_actionSave_Settings_triggered();

    void on_actionOpen_Settings_triggered();

    void on_pushButton_GetD_clicked();

    void on_pushButton_Settings_clicked();

    void on_spinBox_CapAt_valueChanged(int arg1);

    void plotData();

private:
    Ui::MainWindow *ui;
    PicoSettings picoSettings;
    PICO_STATUS status_atom;
    RunPico* runAtom;
    void plotRawSignals(int capture);
    void plotTimeSignals(int captures);
    void plotSpectra();
    void deleteThreads();


};
#endif // MAINWINDOW_H
