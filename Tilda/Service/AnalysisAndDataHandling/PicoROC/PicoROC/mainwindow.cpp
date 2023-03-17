#include "mainwindow.h"
#include "ui_mainwindow.h"



MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->spinBox_Delay->setValue(picoSettings.delay);
    ui->spinBox_MeasT->setValue(picoSettings.measTime);
    ui->spinBox_TimeB->setValue(picoSettings.timebase);
    ui->spinBox_Samples->setValue(picoSettings.samples);
    ui->spinBox_TrigA->setValue(picoSettings.trigAtoms);
    ui->spinBox_TrigI->setValue(picoSettings.trigIons);
    ui->doubleSpinBox_CalAtIn->setValue(picoSettings.calAtomIn);
    ui->doubleSpinBox_CalAtOut->setValue(picoSettings.calAtomOut);
    ui->doubleSpinBox_CalIn->setValue(picoSettings.calIonIn);
    ui->doubleSpinBox_CalIOut->setValue(picoSettings.calIonOut);
    ui->spinBox_Tmin->setValue(picoSettings.tmin);
    ui->spinBox_Tmax->setValue(picoSettings.tmax);
    ui->doubleSpinBox_Emin->setValue(picoSettings.eMin);
    ui->doubleSpinBox_Emax->setValue(picoSettings.eMax);
    ui->qcp_At_Raw->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);
    ui->qcp_time_view_At->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);
    ui->qcp_Energy_view_At->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);
    ui->qcp_spectrum_At->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);
    ui->qcp_spectrum_I->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);
    ui->qcp_spectrum->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);
    runAtom = new RunPico(picoSettings,0);
    runAtom->runThread = true;
    runAtom->start(runAtom->TimeCriticalPriority);

    connect(ui->spinBox_Delay,          SIGNAL(valueChanged(int)),      this,   SLOT(onUpdateSettings()));
    connect(ui->spinBox_MeasT,          SIGNAL(valueChanged(int)),      this,   SLOT(onUpdateSettings()));
    connect(ui->spinBox_TimeB,          SIGNAL(valueChanged(int)),      this,   SLOT(onUpdateSettings()));
    connect(ui->spinBox_Samples,        SIGNAL(valueChanged(int)),      this,   SLOT(onUpdateSettings()));
    connect(ui->spinBox_TrigA,          SIGNAL(valueChanged(int)),      this,   SLOT(onUpdateSettings()));
    connect(ui->spinBox_TrigI,          SIGNAL(valueChanged(int)),      this,   SLOT(onUpdateSettings()));
    connect(ui->doubleSpinBox_CalAtIn,  SIGNAL(valueChanged(double)),   this,   SLOT(onUpdateSettings()));
    connect(ui->doubleSpinBox_CalAtOut, SIGNAL(valueChanged(double)),   this,   SLOT(onUpdateSettings()));
    connect(ui->doubleSpinBox_CalIn,    SIGNAL(valueChanged(double)),   this,   SLOT(onUpdateSettings()));
    connect(ui->doubleSpinBox_CalIOut,  SIGNAL(valueChanged(double)),   this,   SLOT(onUpdateSettings()));
    connect(ui->spinBox_Tmin,           SIGNAL(valueChanged(int)),      this,   SLOT(onUpdateSettings()));
    connect(ui->spinBox_Tmax,           SIGNAL(valueChanged(int)),      this,   SLOT(onUpdateSettings()));
    connect(ui->doubleSpinBox_Emin,     SIGNAL(valueChanged(double)),   this,   SLOT(onUpdateSettings()));
    connect(ui->doubleSpinBox_Emax,     SIGNAL(valueChanged(double)),   this,   SLOT(onUpdateSettings()));
    connect (runAtom->processPico, SIGNAL(dataReady()), this, SLOT(plotData()));

    ui->spinBox_CapAt->setValue(status_atom);
}

void MainWindow::deleteThreads()
{
    runAtom->runThread=false;
    status_atom = ps6000CloseUnit (*picoSettings.handle_atom);
    std::cout << "1" << std::endl;

    runAtom->closeProcessandIPCThreads();

    RunPico::mutexAcq.lock();
    RunPico::startAcq.wakeAll();
    RunPico::mutexAcq.unlock();

    runAtom->exit();
    while(!runAtom->isFinished())
    {
        QThread::msleep(10);
    }

    if(runAtom != nullptr)
    {
        delete runAtom;
        runAtom = nullptr;
    }
}

MainWindow::~MainWindow()
{
    deleteThreads();

    delete ui;
}

void MainWindow::onUpdateSettings()
{
    if      (QObject::sender() == ui->spinBox_Delay)            {picoSettings.delay     = ui->spinBox_Delay->value();}
    else if (QObject::sender() == ui->spinBox_MeasT)            {picoSettings.measTime  = ui->spinBox_MeasT->value();}
    else if (QObject::sender() == ui->spinBox_TimeB)            {picoSettings.timebase  = ui->spinBox_TimeB->value();}
    else if (QObject::sender() == ui->spinBox_Samples)          {picoSettings.samples   = ui->spinBox_Samples->value();}
    else if (QObject::sender() == ui->spinBox_TrigA)            {picoSettings.trigAtoms = ui->spinBox_TrigA->value();}
    else if (QObject::sender() == ui->spinBox_TrigI)            {picoSettings.trigIons  = ui->spinBox_TrigI->value();}
    else if (QObject::sender() == ui->doubleSpinBox_CalAtIn)    {picoSettings.calAtomIn = ui->doubleSpinBox_CalAtIn->value();}
    else if (QObject::sender() == ui->doubleSpinBox_CalAtOut)   {picoSettings.calAtomOut= ui->doubleSpinBox_CalAtOut->value();}
    else if (QObject::sender() == ui->doubleSpinBox_CalIn)      {picoSettings.calIonIn  = ui->doubleSpinBox_CalIn->value();}
    else if (QObject::sender() == ui->doubleSpinBox_CalIOut)    {picoSettings.calIonOut = ui->doubleSpinBox_CalIOut->value();}
    else if (QObject::sender() == ui->spinBox_Tmin)             {picoSettings.tmin      = ui->spinBox_Tmin->value();}
    else if (QObject::sender() == ui->spinBox_Tmax)             {picoSettings.tmax      = ui->spinBox_Tmax->value();}
    else if (QObject::sender() == ui->doubleSpinBox_Emin)       {picoSettings.eMin      = ui->doubleSpinBox_Emin->value();}
    else if (QObject::sender() == ui->doubleSpinBox_Emax)       {picoSettings.eMax      = ui->doubleSpinBox_Emax->value();}
}


void MainWindow::on_actionSave_Settings_triggered()
{
    QString fileName = QFileDialog::getSaveFileName(this,
            tr("Save Settings"), "",
            tr("json (*.jsn);;All Files (*)"));
    picoSettings.savePicoSettings(fileName);
}

void MainWindow::on_actionOpen_Settings_triggered()
{
    QString fileName = QFileDialog::getOpenFileName (this,
            tr("Load Settings"), "",
            tr("json (*.jsn);;All Files (*)"));
    picoSettings.loadPicoSettings(fileName);

    ui->spinBox_Delay->setValue(picoSettings.delay);
    ui->spinBox_MeasT->setValue(picoSettings.measTime);
    ui->spinBox_TimeB->setValue(picoSettings.timebase);
    ui->spinBox_Samples->setValue(picoSettings.samples);
    ui->spinBox_TrigA->setValue(picoSettings.trigAtoms);
    ui->spinBox_TrigI->setValue(picoSettings.trigIons);
    ui->doubleSpinBox_CalAtIn->setValue(picoSettings.calAtomIn);
    ui->doubleSpinBox_CalAtOut->setValue(picoSettings.calAtomOut);
    ui->doubleSpinBox_CalIn->setValue(picoSettings.calIonIn);
    ui->doubleSpinBox_CalIOut->setValue(picoSettings.calIonOut);
    ui->spinBox_Tmin->setValue(picoSettings.tmin);
    ui->spinBox_Tmax->setValue(picoSettings.tmax);
    ui->doubleSpinBox_Emin->setValue(picoSettings.eMin);
    ui->doubleSpinBox_Emax->setValue(picoSettings.eMax);
}

void MainWindow::on_pushButton_GetD_clicked()
{


    RunPico::mutexAcq.lock();
    RunPico::startAcq.wakeAll();
    RunPico::mutexAcq.unlock();

    QEventLoop loop;
    connect (runAtom->processPico, SIGNAL(dataReady()), &loop, SLOT(quit()));
    loop.exec();
    int capture=0;
    if(runAtom->processPico->fb->nEvent > 0) {
        capture=runAtom->processPico->fb->nEvent-1;
    }
    ui->spinBox_CapAt->setMaximum(capture);
    ui->spinBox_CapAt->setValue(capture);

    plotTimeSignals(capture);
    plotRawSignals(capture);
    plotSpectra();
}

void MainWindow::plotData()
{
    int capture=0;
    if(runAtom->processPico->fb->nEvent > 0) {
        capture=runAtom->processPico->fb->nEvent-1;
    }

    ui->spinBox_CapAt->setMaximum(capture);
    ui->spinBox_CapAt->setValue(capture);

    plotTimeSignals(capture);
    plotSpectra();
}

void MainWindow::on_pushButton_Settings_clicked()
{
    deleteThreads();
    runAtom = new RunPico(picoSettings,0);
    runAtom->start();
}

void MainWindow::plotRawSignals(int capture)
{
    int samples=picoSettings.samples;
    QVector<double> x(samples), in(samples), out(samples);
    for (int sample=0; sample<picoSettings.samples; ++sample)
    {
        in[sample] = runAtom->processPico->fb->flatbufferP[capture*3*samples+sample]/256;
        out[sample]= runAtom->processPico->fb->flatbufferP[capture*3*samples+samples+sample]/256;
        x[sample] = sample*picoSettings.getSampleTime();
        ui->qcp_At_Raw->addGraph();
        ui->qcp_At_Raw->graph(0)->setData(x, in);
        ui->qcp_At_Raw->addGraph();
        ui->qcp_At_Raw->graph(1)->setPen(QPen(Qt::red));
        ui->qcp_At_Raw->graph(1)->setData(x, out);
        ui->qcp_At_Raw->rescaleAxes();
        ui->qcp_At_Raw->replot();
    }

}

void MainWindow::plotTimeSignals(int captures)
{
    QVector<double> eventsT(256),eventsDE(256),eventsE(256), time(256),energyin(256),energyout(256);
    for (int i=0; i<captures;++i)
    {
        eventsT[runAtom->processPico->fb->vEvents[i].time]++;
        eventsDE[runAtom->processPico->fb->vEvents[i].energyInner]++;
        eventsE[runAtom->processPico->fb->vEvents[i].energyOuter]++;

        //std::cout<<runAtom->processPico->fb->vEvents[i].time<<std::endl;
    }
    for (int i=0; i<256;++i)
    {
        time[i]=i*picoSettings.measTime/256.0+picoSettings.delay;
        energyin[i]=(i-128)*picoSettings.calAtomIn;
        energyout[i]=(i-128)*picoSettings.calAtomOut;
    }
    ui->qcp_time_view_At->addGraph();
    ui->qcp_time_view_At->graph(0)->setData(time, eventsT);
    ui->qcp_time_view_At->rescaleAxes();
    ui->qcp_time_view_At->replot();

    ui->qcp_Energy_view_At->addGraph();
    ui->qcp_Energy_view_At->graph(0)->setData(energyin, eventsDE);
    ui->qcp_Energy_view_At->addGraph();
    ui->qcp_Energy_view_At->graph(1)->setPen(QPen(Qt::red));
    ui->qcp_Energy_view_At->graph(1)->setData(energyout, eventsE);
    ui->qcp_Energy_view_At->rescaleAxes();
    ui->qcp_Energy_view_At->replot();
}

void MainWindow::plotSpectra()
{
    QVector<double> runs;
    std::cout << "runs size " << runs.size() << " " << runAtom->processPico->v_validEvents.size() << std::endl;
    for (int i=0; i<runAtom->processPico->v_validEvents.size();i++)
    {
        runs.push_back(i);
    }

    ui->qcp_spectrum_At->addGraph();
    ui->qcp_spectrum_At->graph(0)->setData(runs, runAtom->processPico->v_validEvents);
    ui->qcp_spectrum_At->rescaleAxes();
    ui->qcp_spectrum_At->replot();
}

void MainWindow::on_spinBox_CapAt_valueChanged(int arg1)
{
   plotRawSignals(arg1);
}
