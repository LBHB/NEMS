#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 09:25:00 2018

@author: svd
"""

#!/usr/bin/env python

# embedding_in_qt5.py --- Simple Qt4 application embedding matplotlib canvases
#
# Copyright (C) 2005 Florent Rougon
#               2006 Darren Dale
# with Updates from @boxcontrol
# <http://www.boxcontrol.net/embedding-matplotlib-plot-on-pyqt5-gui.html>
# <https://github.com/boxcontrol/matplotlibPyQt5>
#
# This file is an example program for matplotlib. It may be used and
# modified with no restriction; raw copies as well as modified versions
# may be distributed without limitation.

import sys
import random
import copy
import types
import traceback
from functools import wraps
import pandas as pd

import matplotlib
#matplotlib.use("Qt5Agg")
import matplotlib.ticker as tkr
import numpy as np

import PyQt5.QtCore as qc
import PyQt5.QtGui as qg
import PyQt5.QtWidgets as qw

from numpy import arange, sin, pi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

#import nems_db.db as nd
#import nems_db.xform_wrappers as nw
from nems0.recording import Recording
import nems0.signal
from nems0.plots.utils import ax_remove_box
#from nems0.gui.editor import ModelEditor

class RecordingPlotWrapper():
    # TODO: Not using this anymore?
    def __init__(self, recording):

        test_signals=['stim','resp','pred']

        self.recording = recording
        self.plot_signals = test_signals
        self.start_time = 0
        self.stop_time = 10

    def get_segment(self, signal):

        fs = self.recording[signal].fs
        start_bin = int(self.start_time * fs)
        stop_bin = int(self.stop_time * fs)

        d = recording[signal].as_continuous()[:,start_bin:stopbin]

        return d


def show_exceptions(*args):
    if len(args) == 0 or isinstance(args[0], types.FunctionType):
        args = []

    @qc.pyqtSlot(*args)
    def slotdecorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                func(*args)
            except:
                print("\n*** Uncaught Exception in PyQt slot ***")
                traceback.print_exc()
        return wrapper

    return slotdecorator


class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        # tweak size of axes in window
        pos1 = self.axes.get_position() # get the original position
        pos2 = [0.075, 0.1, 0.9, 0.8]
        self.axes.set_position(pos2) # set a new position

        # We want the axes cleared every time plot() is called
        #self.axes.hold(False)

        self.compute_initial_figure()
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, qw.QSizePolicy.Expanding,
                                   qw.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass


class MyStaticMplCanvas(MyMplCanvas):
    """Simple canvas with a sine plot."""
    def compute_initial_figure(self):
        t = arange(0.0, 3.0, 0.01)
        s = sin(2*pi*t)
        self.axes.plot(t, s)


class MyDynamicMplCanvas(MyMplCanvas):
    """A canvas that updates itself every second with a new plot."""
    def __init__(self, *args, **kwargs):
        MyMplCanvas.__init__(self, *args, **kwargs)
        timer = qc.QTimer(self)
        timer.timeout.connect(self.update_figure)
        timer.start(1000)

    def compute_initial_figure(self):
        self.axes.plot([0, 1, 2, 3], [1, 2, 0, 4], 'r')

    def update_figure(self):
        # Build a list of 4 random integers between 0 and 10 (both inclusive)
        l = [random.randint(0, 10) for i in range(4)]

        self.axes.plot([0, 1, 2, 3], l, 'r')
        self.draw()


class NemsCanvas(MyMplCanvas):
    """A canvas that updates itself every second with a new plot."""

    def __init__(self, recording=None, signal='stim', parent=None,
                 *args, **kwargs):
        MyMplCanvas.__init__(self, *args, **kwargs)
        if 'mask' in recording.signals:
            self.recording = recording.apply_mask()
        else:
            self.recording = recording
        self.signal = signal
        self.signal_obj = self.recording[self.signal]
        self.fs = self.signal_obj.fs
        self.parent = parent
        print("creating canvas: {}".format(signal))

        sig_array = self.signal_obj.as_continuous()
        # Chop off end of array (where it's all nan'd out after processing)
        # TODO: Make this smarter incase there are intermediate nans?
        self.max_time = sig_array.shape[-1] / self.recording[self.signal].fs

        point = (isinstance(self.recording[self.signal],
                            nems0.signal.PointProcess))
        tiled = (isinstance(self.recording[self.signal],
                            nems0.signal.TiledSignal)
                 or 'stim' in self.recording[self.signal].name
                 or 'contrast' in self.recording[self.signal].name)

        if (not point) and (not tiled):
            self.ymax = np.nanmax(sig_array)*1.25
            self.ymin = min(0, np.nanmin(sig_array)*1.25)

        self.point = point
        self.tiled = tiled

        # skip some channels, get names
        c_count = self.recording[self.signal].shape[0]
        if self.recording[self.signal].chans is None:
            channel_names = [''] * c_count
        else:
            channel_names=self.recording[self.signal].chans[:c_count]
        skip_channels = ['baseline']
        if channel_names is not None:
            keep = np.array([(n not in skip_channels) for n in channel_names])
            channel_names = [channel_names[i] for i in range(c_count) if keep[i]]
        else:
            keep = np.ones(c_count, dtype=bool)
            channel_names = None
        self.keep = keep
        self.channel_names = channel_names

        p = self.parent

        d = sig_array[self.keep, :]

        if self.point:
            self.axes.imshow(d, aspect='auto', cmap='Greys',
                             interpolation='nearest', origin='lower')
            self.axes.get_yaxis().set_visible(False)
        elif self.tiled:
            self.axes.imshow(d, aspect='auto', origin='lower')
        else:
            self.axes.plot(d.T)
            if self.channel_names is not None:
                if len(self.channel_names) > 1:
                    self.axes.legend(self.channel_names, frameon=False)
            self.axes.set_ylim(ymin=self.ymin, ymax=self.ymax)

        self.axes.set_xlim(p.start_time*self.fs, p.stop_time*self.fs)
        self.axes.set_ylabel(self.signal)
        ax_remove_box(self.axes)
        self.draw()

        tick_labels = self.axes.get_xticklabels()
        if self.point or self.tiled:
            new_labels = ['']*len(tick_labels)
            self.axes.set_xticklabels(new_labels)
            self.draw()
        else:
            # TODO: Still not working... Should turn bins to seconds
            fmt = tkr.FuncFormatter(self.seconds_formatter())
            self.axes.yaxis.set_major_formatter(fmt)
            self.draw()

    def compute_initial_figure(self):
        pass

    def seconds_formatter(self):
        def fmt(x, pos):
            s = '{}'.format(x / self.fs)
            return s
        return fmt

    def update_figure(self):
        p = self.parent
        self.axes.set_xlim(p.start_time*self.fs, p.stop_time*self.fs)
        if not (self.point or self.tiled):
            self.axes.set_ylim(ymin=self.ymin, ymax=self.ymax)
        self.draw()


class EpochCanvas(MyMplCanvas):
    """A canvas that updates itself every second with a new plot."""

    def __init__(self, recording=None, signal='stim', parent=None,
                 *args, **kwargs):
        MyMplCanvas.__init__(self, *args, **kwargs)
        self.recording = recording
        self.signal = signal
        self.parent = parent
        print("creating epoch canvas: {}".format(signal))
        self.max_time = 0
        self.epoch_groups = {}

#        self.axes.cla()
##########################################################
#        epochs = self.recording.epochs
#
#        # On each refresh, keep the same keys but reform the lists of indices.
#        self.epoch_groups = {k: [] for k in self.epoch_groups}
#        for i, r in epochs.iterrows():
#            s = r['start']
#            e = r['end']
#            n = r['name']
#
#            prefix = n.split('_')[0]
#            if prefix in ['PreStimSilence', 'PostStimSilence',
#                          'REFERENCE','TARGET']:
#                # skip
#                pass
#            elif prefix in self.epoch_groups:
#                self.epoch_groups[prefix].append(i)
#            else:
#                self.epoch_groups[prefix] = [i]
#
#        colors = ['Red', 'Orange', 'Green', 'LightBlue',
#                  'DarkBlue', 'Purple', 'Pink', 'Black', 'Gray']
#        i = 0
#        for i, g in enumerate(self.epoch_groups):
#            for j in self.epoch_groups[g]:
#                n = epochs['name'][j]
#                s = epochs['start'][j]
#                e = epochs['end'][j]
#
#                try:
#                    n2 = epochs['name'][j+1]
#                    s2 = epochs['start'][j+1]
#                    e2 = epochs['end'][j+1]
#                except KeyError:
#                    # j is already the last epoch in the list
#                    pass
#                    n2 = n
#                    s2 = s
#                    e2 = e
#
#                # If two epochs with the same name overlap,
#                # extend the end of the first to the end of the second
#                # and skip the second epoch.
#                # Same if end goes past next start.
#                if n == n2:
#                    if (s2 < e) or (e > s2):
#                        e = e2
#                        j += 1
#                    else:
#                        pass
#
#                x = np.array([s, e])
#                y = np.array([i, i])
#
#                self.axes.plot(x, y, '-', color=colors[i % len(colors)])
#                self.axes.text(s, i, n, va='bottom', fontsize='small',
#                               color=colors[i % len(colors)])
#
#        self.axes.set_xlim([self.parent.start_time, self.parent.stop_time])
#        self.axes.set_ylim([-0.5, i+0.5])
#        ax_remove_box(self.axes)
#        self.draw()
#
#        xtick_labels = self.axes.get_xticklabels()
#        ytick_labels = self.axes.get_yticklabels()
#        new_xlabels = ['']*len(xtick_labels)
#        new_ylabels = ['']*len(ytick_labels)
#        self.axes.set_xticklabels(new_xlabels)
#        self.axes.set_yticklabels(new_ylabels)
#        self.axes.set_ylabel('epochs')
#        self.draw()

    def compute_initial_figure(self):
        pass

    def update_figure(self):
        self.axes.cla()

        epochs = self.recording.epochs
        p = self.parent
        valid_epochs = epochs[(epochs['start'] >= p.start_time) &
                              (epochs['end'] < p.stop_time)]
        if valid_epochs.size == 0:
            print('no valid epochs')
            # valid_epochs = valid_epochs.append([{'name': 'EXPT', 'start': p.start_time, 'end': p.stop_time}])
            return

        # On each refresh, keep the same keys but reform the lists of indices.
        self.epoch_groups = {k: [] for k in self.epoch_groups}
        for i, r in valid_epochs.iterrows():
            s = r['start']
            e = r['end']
            n = r['name']

            prefix = n.split('_')[0]
            if prefix in ['PreStimSilence', 'PostStimSilence',
                          'REFERENCE','TARGET']:
                # skip
                pass
            elif prefix in self.epoch_groups:
                self.epoch_groups[prefix].append(i)
            else:
                self.epoch_groups[prefix] = [i]

        colors = ['Red', 'Orange', 'Green', 'LightBlue',
                  'DarkBlue', 'Purple', 'Pink', 'Black', 'Gray']
        i = 0
        for i, g in enumerate(self.epoch_groups):
            for j in self.epoch_groups[g]:
                n = valid_epochs['name'][j]
                s = valid_epochs['start'][j]
                e = valid_epochs['end'][j]

                try:
                    n2 = valid_epochs['name'][j+1]
                    s2 = valid_epochs['start'][j+1]
                    e2 = valid_epochs['end'][j+1]
                except KeyError:
                    # j is already the last epoch in the list
                    pass
                    n2 = n
                    s2 = s
                    e2 = e

                # If two epochs with the same name overlap,
                # extend the end of the first to the end of the second
                # and skip the second epoch.
                # Same if end goes past next start.
                if n == n2:
                    if (s2 < e) or (e > s2):
                        e = e2
                        j += 1
                    else:
                        pass

                # Don't plot text boxes outside of plot limits
                if s < p.start_time:
                    s = p.start_time
                elif e > p.stop_time:
                    e = p.stop_time

                x = np.array([s, e])
                y = np.array([i, i])

                self.axes.plot(x, y, '-', color=colors[i % len(colors)])
                self.axes.text(s, i, n, va='bottom', fontsize='small',
                               color=colors[i % len(colors)])

        self.axes.set_xlim([p.start_time, p.stop_time])
        self.axes.set_ylim([-0.5, i+0.5])
        ax_remove_box(self.axes)
        self.draw()

        xtick_labels = self.axes.get_xticklabels()
        ytick_labels = self.axes.get_yticklabels()
        new_xlabels = ['']*len(xtick_labels)
        new_ylabels = ['']*len(ytick_labels)
        self.axes.set_xticklabels(new_xlabels)
        self.axes.set_yticklabels(new_ylabels)
        self.axes.set_ylabel('epochs')
        self.draw()

#        self.axes.set_xlim(self.parent.start_time, self.parent.stop_time)
#        self.draw()


class ApplicationWindow(qw.QMainWindow):

    recording=None
    signals = []
    plot_list = []
    start_time = 0
    display_duration = 10.0
    minimum_duration = 0.001
    stop_time = 10
    time_slider = None

    plot_width = 6
    plot_height = 4
    plot_dpi = 100

    def __init__(self, recording, signals=['stim', 'resp'], cellid=None,
                 modelname=None, ctx=None, xfspec=None):
        qw.QMainWindow.__init__(self)

        self.recording=recording
        self.signals=signals
        self.cellid=cellid
        self.modelname=modelname
        self.ctx = ctx
        self.xfspec = xfspec

        self.setAttribute(qc.Qt.WA_DeleteOnClose)

        self.file_menu = qw.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 qc.Qt.CTRL + qc.Qt.Key_Q)
        self.file_menu.addAction('Screenshot', self.save_screenshot,
                                 qc.Qt.CTRL + qc.Qt.Key_S)
        self.menuBar().addMenu(self.file_menu)

        self.help_menu = qw.QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)

        self.help_menu.addAction('&About', self.about)

        self.main_widget = qw.QWidget(self)
        self.outer_layout = qw.QVBoxLayout()
        header = ''
        if modelname is not None:
            header += 'Model: %s\n' % modelname
        if cellid is not None:
            header += 'Cellid: %s' % cellid
        self.setWindowTitle("application main window")
        self.outer_layout.addWidget(qw.QLabel(header))


        # mpl panels
        signals = [value for value in signals if value in recording.signals]
        self.plot_list = [self._default_plot_instance(s) for s in signals]
        epoch_canvas = EpochCanvas(
                self.recording, 'resp', self, self.main_widget,
                width=self.plot_width, height=self.plot_height,
                dpi=self.plot_dpi
                )
        self.plot_list.insert(0, epoch_canvas)
        self.plot_layout = qw.QVBoxLayout()
        self.plot_layout.setSpacing(25)
        [self.plot_layout.addWidget(p) for p in self.plot_list]

        # Slider for plot view windows
        self._update_max_time()
        self.time_slider = qw.QScrollBar(orientation=1)
        self.time_slider.setRange(0, self.max_time-self.display_duration)
        self.time_slider.setRepeatAction(200, 2)
        self.time_slider.setSingleStep(1)
        self.time_slider.valueChanged.connect(self.scroll_all)
        self.plot_layout.addWidget(self.time_slider)

        self.outer_layout.addLayout(self.plot_layout)

        # TODO not working yet
        tap_right_short = qw.QShortcut(qg.QKeySequence(qc.Qt.Key_Right), self)
        tap_right_short.activated.connect(self.tap_right)
        tap_left_short = qw.QShortcut(qg.QKeySequence(qc.Qt.Key_Left), self)
        tap_left_short.activated.connect(self.tap_left)


        # Set zoom / display range for plot views
        self.display_range = qw.QLineEdit()
        self.display_range.setValidator(
                qg.QDoubleValidator(self.minimum_duration, 10000.0, 4)
                )
        self.display_range.editingFinished.connect(self.set_display_range)
        self.display_range.setText(str(self.display_duration))

        # Increment / Decrement zoom
        plus = qw.QPushButton('Zoom Out')
        plus.clicked.connect(self.increment_display_range)
        minus = qw.QPushButton('Zoom In')
        minus.clicked.connect(self.decrement_display_range)

        range_layout = qw.QHBoxLayout()
        [range_layout.addWidget(w) for w in [self.display_range, plus, minus]]
        self.outer_layout.addLayout(range_layout)

        # control buttons
        qbtn = qw.QPushButton('Quit', self)
        qbtn.clicked.connect(self.close)

        qbtn2 = qw.QPushButton('Test', self)
        qbtn2.clicked.connect(self.print_signals)

        add_sig = qw.QPushButton('Add Signal', self)
        add_sig.clicked.connect(self.add_signal)

        remove_sig = qw.QPushButton('Remove Signal', self)
        remove_sig.clicked.connect(self.remove_signal)

        #edit_model = qw.QPushButton('Edit Model', self)
        #edit_model.clicked.connect(self.open_model_editor)

        control_layout = qw.QHBoxLayout()
        control_layout.addWidget(qbtn)
        control_layout.addWidget(qbtn2)
        control_layout.addWidget(add_sig)
        control_layout.addWidget(remove_sig)
        #control_layout.addWidget(edit_model)
        self.outer_layout.addLayout(control_layout)

        self.main_widget.setLayout(self.outer_layout)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        self.statusBar().showMessage("Welcome to the NEMS recording browser", 2000)
        self._update_range()

    # Plot Window adjusters
    #@show_exceptions('bool')
    def scroll_all(self):
        self.start_time = self.time_slider.value()
        self.stop_time = self.start_time + self.display_duration

        # don't go past the latest time of the biggest plot
        # (should all have the same max most of the time)
        self._update_max_time()
        if self.stop_time >= self.max_time:
            self.stop_time = self.max_time
            self.start_time = max(0, self.max_time - self.display_duration)

        [p.update_figure() for p in self.plot_list]

    def _update_max_time(self):
        self.max_time = max([p.max_time for p in self.plot_list])

    def tap_right(self):
        self.time_slider.set_value(
                self.time_slider.value + self.time_slider.singleStep
                )

    def tap_left(self):
        self.time_slider.set_value(
                self.time_slider.value - self.time_slider.singleStep
                )

    #@show_exceptions('bool')
    def set_display_range(self):
        duration = float(self.display_range.text())
        if not duration:
            print("Duration not set to a valid value. Please enter a"
                  "a number > 0")
            return
        self.display_duration = duration
        self._update_range()

    #@show_exceptions('bool')
    def increment_display_range(self):
        self.display_duration += 1
        self.display_range.setText(str(self.display_duration))
        self._update_range()

    #@show_exceptions('bool')
    def decrement_display_range(self):
        self.display_duration -= 1
        self.display_range.setText(str(self.display_duration))
        self._update_range()

    def _update_range(self):
        self.time_slider.setRange(0, self.max_time-self.display_duration)
        self.time_slider.setSingleStep(int(np.ceil(self.display_duration/10)))
        self.time_slider.setPageStep(int(self.display_duration))
        self.scroll_all()

    # Add / Remove plots

    #@show_exceptions('bool')
    def print_signals(self):
        '''
        For testing/development, verify plot_list, layout and signals
        match up.
        '''
        for s in self.signals:
            print("signal: %s" % s)
        for p in self.plot_list:
            print("plot: %s" % p.signal)

    #@show_exceptions('bool')
    def add_signal(self):
        n = len(self.plot_list)
        valid_signals = [s for s in self.recording.signals
                         if s not in self.signals]
        idx, ok = qw.QInputDialog.getInt(self, "Which plot position?", "Index",
                                         n, 0, n, 1)
        if not ok:
            return
        s, ok = qw.QInputDialog.getItem(self, "Name of the signal to plot?",
                                        "Signals:", valid_signals, 0, False)
        if not ok:
            return

        self.signals.insert(idx, s)
        self.plot_list.insert(idx, self._default_plot_instance(s))
        if idx in [-1, len(self.plot_list)]:
            layout_idx = idx-1
        else:
            layout_idx = idx
        self.plot_layout.insertWidget(layout_idx, self.plot_list[idx])
        self.main_widget.update()
        self.scroll_all()

    #@show_exceptions('bool')
    def remove_signal(self):
        s, ok = qw.QInputDialog.getItem(self, "Name of the signal to remove?",
                                        "Signals:", self.signals, 0, False)
        if not ok:
            return

        idx = self.signals.index(s)
        self.signals.pop(idx)
        self.plot_list.pop(idx)
        self.plot_layout.itemAt(idx).widget().deleteLater()
        self.main_widget.update()
        self.scroll_all()

    #@show_exceptions('bool')
    def _default_plot_instance(self, s):
        return NemsCanvas(self.recording, s, self, self.main_widget,
                          width=self.plot_width, height=self.plot_height,
                          dpi=self.plot_dpi)

    #def open_model_editor(self):
    #    self.editor = ModelEditor(ctx=self.ctx, xfspec=self.xfspec)

    def close_model_editor(self):
        self.editor = None

    def resp_as_spikes(self):
        pass

    def resp_as_psth(self):
        pass

    @show_exceptions('bool')
    def save_screenshot(self):
        raise ValueError("Screenshots not working quite yet...")
        file, ok = qw.QFileDialog.getSaveFileName(
                self, "Filepath to save screenshot?",
                filter="PNG(*.png);; JPEG(*.jpg)"
                )
        if not ok:
            return

        screenshot = qw.QApplication.primaryScreen().grabWindow(0)
        if file[-3:] == 'png':
            screenshot.save(file, 'png')
        elif file[-3:] == 'jpg':
            screenshot.save(file, 'jpg')

    def closeEvent(self):
        print('closing recording browser')

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        qw.QMessageBox.about(self, "About",
  """NEMS recording browser

  Implemented based on embedding_in_qt5.py example (Copyright 2015 BoxControL)
  and
  http://matplotlib.org/examples/user_interfaces/embedding_in_qt4.html
  """
  )


class PandasModel(qc.QAbstractTableModel):
    def __init__(self, df = pd.DataFrame(), parent=None):
        qc.QAbstractTableModel.__init__(self, parent=parent)
        self._df = df

    def headerData(self, section, orientation, role=qc.Qt.DisplayRole):
        if role != qc.Qt.DisplayRole:
            return qc.QVariant()

        if orientation == qc.Qt.Horizontal:
            try:
                return self._df.columns.tolist()[section]
            except (IndexError, ):
                return qc.QVariant()
        elif orientation == qc.Qt.Vertical:
            try:
                # return self.df.index.tolist()
                return self._df.index.tolist()[section]
            except (IndexError, ):
                return qc.QVariant()

    def data(self, index, role=qc.Qt.DisplayRole):
        if role != qc.Qt.DisplayRole:
            return qc.QVariant()

        if not index.isValid():
            return qc.QVariant()

        return qc.QVariant(str(self._df.ix[index.row(), index.column()]))

    def setData(self, index, value, role):
        row = self._df.index[index.row()]
        col = self._df.columns[index.column()]
        if hasattr(value, 'toPyObject'):
            # PyQt4 gets a QVariant
            value = value.toPyObject()
        else:
            # PySide gets an unicode
            dtype = self._df[col].dtype
            if dtype != object:
                value = None if value == '' else dtype.type(value)
        self._df.set_value(row, col, value)
        return True

    def rowCount(self, parent=qc.QModelIndex()):
        return len(self._df.index)

    def columnCount(self, parent=qc.QModelIndex()):
        return len(self._df.columns)

    def sort(self, column, order):
        colname = self._df.columns.tolist()[column]
        self.layoutAboutToBeChanged.emit()
        self._df.sort_values(colname, ascending= order == qc.Qt.AscendingOrder, inplace=True)
        self._df.reset_index(inplace=True, drop=True)
        self.layoutChanged.emit()


def pandas_table_test():

    data = {'a': [1, 2, 3], 'b': ['dog','cat','ferret']}
    df = pd.DataFrame.from_dict(data)
    w = qw.QWidget()

    def loadFile(self):
        fileName, _ = qw.QFileDialog.getOpenFileName(w, "Open File", "", "CSV Files (*.csv)");
        pathLE.setText(fileName)
        df = pd.read_csv(fileName)
        model = PandasModel(df)
        pandasTv.setModel(model)


    hLayout = qw.QHBoxLayout()
    pathLE = qw.QLineEdit(w)
    hLayout.addWidget(pathLE)
    loadBtn = qw.QPushButton("Select File", w)
    hLayout.addWidget(loadBtn)
    loadBtn.clicked.connect(loadFile)

    vLayout = qw.QVBoxLayout(w)
    vLayout.addLayout(hLayout)

    pandasTv = qw.QTableView()
    model = PandasModel(df)
    pandasTv.setModel(model)
    vLayout.addWidget(pandasTv)

    w.show()
    w.raise_()
    return w


def browse_recording(rec, signals=['stim', 'resp', 'psth', 'pred'], cellid=None,
                     modelname=None, ctx=None, xfspec=None):
    aw = ApplicationWindow(recording=rec, signals=signals,
                           cellid=cellid, modelname=modelname, ctx=ctx,
                           xfspec=xfspec)
    _window_startup(aw)

    return aw


def browse_context(ctx, rec='val', signals=['stim', 'resp'], rec_idx=0,
                   xfspec=None):
    rec = ctx[rec]
    if isinstance(rec, list):
        rec = rec[rec_idx]
    meta = ctx['modelspec'].meta
    cellid = meta.get('cellid', None)
    modelname = meta.get('modelname', None)

    aw = browse_recording(rec, signals, cellid, modelname,
                          ctx=ctx, xfspec=xfspec)

    return aw


def _window_startup(aw):
    aw.setWindowTitle("NEMS data browser")
    aw.show()
    aw.scroll_all()

    aw.raise_()

    return aw

#import nems_db.xform_wrappers as nw
#batch = 289
#modelname = "ozgf.fs100.ch18-ld-sev_dlog-wc.18x2.g-fir.2x15-lvl.1-dexp.1_init-basic"
#cellid = 'TAR010c-21-4'
#xf, ctx = nw.load_model_baphy_xform(cellid, batch, modelname, eval_model=True,
#                                    only=0)

#browse_context(ctx, rec='val', signals=['stim', 'resp'], rec_idx=0)
