import matplotlib.ticker as tkr
import numpy as np
import PyQt5.QtWidgets as qw
import PyQt5.QtGui as qg
import PyQt5.QtCore as qc

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import random

from nems0.plots.utils import ax_remove_box

class MplWindow(qw.QWidget):
    def __init__(self, parent=None, fig=None):
        super(MplWindow, self).__init__(parent)

        # a figure instance to plot on
        if fig is None:
            self.figure = plt.figure()
        else:
            self.figure = fig

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Just some button connected to `plot` method
        self.button = qw.QPushButton('Plot')
        self.button.clicked.connect(self.plot)

        # set the layout
        layout = qw.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def plot(self):
        ''' plot some random stuff '''
        # random data
        data = [random.random() for i in range(10)]

        # create an axis
        ax = self.figure.add_subplot(111)

        # discards the old graph
        ax.clear()

        # plot data
        ax.plot(data, '*-')

        # refresh canvas
        self.canvas.draw()


class NemsCanvas(FigureCanvas):

    def __init__(self, parent=None, width=8, height=2, dpi=72, hide_axes=True):
        '''QWidget for displaying a matplotlib axes.'''
        plt.ioff()
        fig = plt.figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.figure = fig
        if hide_axes:
            self.axes.get_yaxis().set_visible(False)
            self.axes.get_xaxis().set_visible(False)

        super(FigureCanvas, self).__init__(fig)
        self._bbox_queue = []
        self.setParent(parent)
        self.setMinimumHeight(50)
        FigureCanvas.setSizePolicy(self, qw.QSizePolicy.Expanding,
                                   qw.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        if hide_axes:
            self.setContentsMargins(0, 0, 0, 0)
            self.figure.subplots_adjust(left=0, bottom=0, right=1,
                                        top=1, wspace=0, hspace=0)
        if hide_axes:
            self.axes.spines['right'].set_visible(False)
            self.axes.spines['top'].set_visible(False)
            self.axes.spines['bottom'].set_visible(False)
            self.axes.spines['left'].set_visible(False)


class ComparisonCanvas(NemsCanvas):
    def __init__(self, signals, names, parent=None, *args, **kwargs):
        NemsCanvas.__init__(self, *args, **kwargs)
        plt.sca(self.axes)
        for s in signals:
            plt.plot(s)
        plt.legend(names)


class EpochCanvas(NemsCanvas):

    def __init__(self, recording=None, signal='stim', parent=None,
                 *args, **kwargs):
        '''QWidget for displaying epochs timestamps in a matplotlib axes.'''
        NemsCanvas.__init__(self, *args, **kwargs)
        self.recording = recording
        self.signal = signal
        self.parent = parent
        self.max_time = 0
        self.epoch_groups = {}
        self.update_figure()

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

            prefix = n.split('_')[0].split(',')[0].strip(' ').lower()
            if len(n) < 5:
                prefix = 'X'
            if prefix in ['prestimsilence', 'poststimsilence',
                          'reference', 'target']:
                # skip
                pass
            elif prefix in self.epoch_groups:
                self.epoch_groups[prefix].append(i)
            else:
                self.epoch_groups[prefix] = [i]

        colors = ['Red', 'Orange', 'Green', 'LightBlue',
                  'DarkBlue', 'Purple', 'Pink', 'Black', 'Gray']
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
                #if len(self.epoch_groups[g]) < 5:
                self.axes.text(s, i, n, va='bottom', fontsize='small',
                               color=colors[i % len(colors)])

        self.axes.set_xlim([p.start_time, p.stop_time])
        self.axes.set_ylim([-0.5, i+0.5])
        ax_remove_box(self.axes)
        self.draw()


class PrettyWidget(qw.QWidget):

    def __init__(self, parent=None, imagepath=None):
        qw.QWidget.__init__(self, parent=parent)
        self.imagepath = imagepath
        self.resize(400, 300)

        self.center()
        self.setWindowTitle('Browser')
        self.config_group = 'PrettyWidget'

        self.lb = qw.QLabel(self)
        self.lb.resize(self.width(), self.height())
        self.pixmap = None

        self.update_imagepath(imagepath)
        #self.show()

    def resizeEvent(self, event):
        self.lb.resize(self.width(), self.height())
        self.lb.setPixmap(self.pixmap.scaled(self.size(), qc.Qt.KeepAspectRatio, qc.Qt.SmoothTransformation))
        qw.QWidget.resizeEvent(self, event)

    def update_imagepath(self, imagepath):
        self.imagepath=imagepath
        self.pixmap = qg.QPixmap(self.imagepath)
        self.resize(self.pixmap.width(), self.pixmap.height())
        self.lb.resize(self.width(), self.height())
        self.lb.setPixmap(self.pixmap.scaled(self.size(), qc.Qt.KeepAspectRatio, qc.Qt.SmoothTransformation))

        #self.lb.setPixmap(self.pixmap.scaled(self.size(), qc.Qt.KeepAspectRatio, qc.Qt.SmoothTransformation))
        #self.pixmap.repaint()

    def center(self):
        qr = self.frameGeometry()
        cp = qw.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

