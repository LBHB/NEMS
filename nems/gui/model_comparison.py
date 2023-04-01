import numpy as np
import PyQt5.QtWidgets as qw
import PyQt5.QtCore as qc
import PyQt5.QtGui as qg
import matplotlib.pyplot as plt

from nems0 import xforms
from nems0.gui.models import ArrayModel
from nems0.gui.canvas import NemsCanvas
from nems0.modelspec import _lookup_fn_at
import nems0.db as nd
from nems0.registry import KeywordRegistry
from nems0.plugins import (default_keywords, default_loaders,
                          default_initializers, default_fitters)
from nems0 import get_setting

class ComparisonWindow(qw.QMainWindow):

    def __init__(self, batch, cellids, modelnames):
        '''
        Main Window wrapper for NEMS model editor GUI.
        Allows browsing and editing of fitted model parameters,
        xforms spec options (TODO), plotting data on a per-module
        basis, and manual initialization & fitting with adjustable
        iteration counts.

        Parameters
        ----------
        modelspec : ModelSpec
            A NEMS ModelSpec containing at least one module.
        xfspec : nested list
            A NEMS xforms spec (see nems0.xforms) containing at least one step.
        rec : Recording
            A NEMS Recording, generally expected to contain 'stim', 'resp',
            and 'pred' signals.
        ctx : dict
            A NEMS context dictionary (see nems0.xforms)
        rec_name : str
            Key used to set rec from ctx instead of passing rec directly,
            e.x. 'val' or 'est'.


        '''
        super(qw.QMainWindow, self).__init__()
        self.browser = ComparisonWidget(batch, cellids, modelnames, self)
        self.setCentralWidget(self.browser)
        self.show()


class ComparisonWidget(qw.QWidget):

    def __init__(self, batch, cellids, modelnames, parent=None):
        '''
        contexts should be a nested dictionary with the format:
            contexts = {
                    cellid1: {'model1': ctx_a, 'model2': ctx_b},
                    cellid2: {'model1': ctx_c, 'model2': ctx_d},
                    ...
                    }
        '''
        super(qw.QWidget, self).__init__()
        d = nd.get_results_file(batch=batch, cellids=cellids,
                                modelnames=modelnames)
        contexts = {}
        for c in cellids:
            cell_contexts = {}
            for m in modelnames:
                try:
                    filepath = d[d.cellid == c][d.modelname == m]['modelpath'].values[0] + '/'
                    xfspec, ctx = xforms.load_analysis(filepath, eval_model=True)
                    cell_contexts[m] = ctx
                except IndexError:
                    print("Coudln't find modelpath for cell: %s model: %s"
                          % (c, m))
                    pass
            contexts[c] = cell_contexts
        self.contexts = contexts
        self.batch = batch
        self.cellids = cellids
        self.modelnames = modelnames

        self.time_scroller = TimeScroller(self)

        self.layout = qw.QVBoxLayout()
        self.tabs = qw.QTabWidget()
        self.comparison_tabs = []
        for k, v in self.contexts.items():
            names = list(v.keys())
            names.insert(0, 'Response')
            signals = []
            for i, m in enumerate(v):
                if i == 0:
                    resp = resp = v[list(v.keys())[0]]['val']['resp']
                    times = np.linspace(0, resp.shape[-1] / resp.fs, resp.shape[-1])
                    signals.append(resp.as_continuous().T)
                signals.append(v[m]['val']['pred'].as_continuous().T)
            if signals:
                tab = ComparisonFrame(signals, names, times, self)
                self.comparison_tabs.append(tab)
                self.tabs.addTab(tab, k)
            else:
                pass

        self.time_scroller._update_max_time()

        self.layout.addWidget(self.tabs)
        self.layout.addWidget(self.time_scroller)
        self.setLayout(self.layout)


class ComparisonFrame(qw.QFrame):

    def __init__(self, signals, names, times, parent):
        super(qw.QFrame, self).__init__()
        self.parent = parent
        self.setFrameStyle(qw.QFrame.Panel | qw.QFrame.Sunken)
        self.highlight_obj = None
        self.times = times

        # TODO: toggle which signals are active
        self.signals = signals
        self.active_signals = signals
        self.names = names
        self.active_names = names

        self.layout = qw.QHBoxLayout()
        self.canvas = qw.QWidget()  # temporary canvas
        self.layout.addWidget(self.canvas)
        self.layout.setAlignment(qc.Qt.AlignTop)
        self.setLayout(self.layout)
        self.new_plot()

    def new_plot(self):
        '''Remove plot from layout and replace it with a new one.'''
        self.layout.removeWidget(self.canvas)
        self.highlight_obj = None
        self.canvas.close()
        self.canvas = NemsCanvas(parent=self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.plot_on_axes()
        self.layout.addWidget(self.canvas)

        self.update_plot()

    def plot_on_axes(self):
        '''Draw plot on current canvas axes.'''
        ax = self.canvas.figure.add_subplot(111)
        plt.sca(ax)
        # always include response as backdrop
        plt.plot(self.times, self.signals[0], color='gray', alpha=0.4)
        for s in self.active_signals[1:]:
            plt.plot(self.times, s)
        plt.legend(self.active_names)

        self.canvas.draw()

    def update_plot(self):
        '''Shift xlimits of current plot if it's scrollable.'''
        ts = self.parent.time_scroller
        self.canvas.axes.set_xlim(ts.start_time, ts.stop_time)
        self.canvas.draw()


class TimeScroller(qw.QFrame):

    start_time = 0
    display_duration = 10.0
    minimum_duration = 0.001
    stop_time = 10
    slider_scaling = 6
    max_time = 5000  # Larger value -> smoother scrolling

    def __init__(self, parent):
        '''QWidget for controlling cell_index, fit_index, and plot xlims.'''
        super(qw.QFrame, self).__init__()
        self.parent = parent
        self.collapsed = False
        self.setFrameStyle(qw.QFrame.Panel | qw.QFrame.Raised)

        # Slider for plot view windows
        self.time_slider = qw.QScrollBar(orientation=1)
        policy = qw.QSizePolicy()
        policy.setHorizontalPolicy(qw.QSizePolicy.Expanding)
        self.time_slider.setSizePolicy(policy)
        #self._update_max_time()
        self.time_slider.setRange(0, self.max_time)
        self.time_slider.setRepeatAction(200, 2)
        self.time_slider.setSingleStep(1)
        self.time_slider.valueChanged.connect(self.scroll_all)

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
        self.range_layout = qw.QHBoxLayout()
        self.range_layout.setAlignment(qc.Qt.AlignTop)
        [self.range_layout.addWidget(w) for w in [self.display_range, plus, minus]]

        self.buttons_layout = qw.QHBoxLayout()
        self.buttons_layout.setAlignment(qc.Qt.AlignTop)

        layout = qw.QVBoxLayout()
        layout.setAlignment(qc.Qt.AlignTop)
        layout.addWidget(self.time_slider)
        layout.addLayout(self.range_layout)
        layout.addLayout(self.buttons_layout)
        self.setLayout(layout)

        #self._update_range()
        self.time_slider.setRange(0, self.max_time-self.display_duration)
        self.time_slider.setSingleStep(int(np.ceil(self.display_duration/10)))
        self.time_slider.setPageStep(int(self.display_duration))

    def scroll_all(self):
        '''Update xlims for all plots based on slider value.'''
        self.start_time = self.time_slider.value()
        self.stop_time = self.start_time + self.display_duration
        [c.update_plot() for c in self.parent.comparison_tabs]

    def _update_max_time(self):
        contexts = self.parent.contexts
        c = self.parent.cellids[0]
        m = self.parent.modelnames[0]
        resp = contexts[c][m]['val'].apply_mask()['resp']
        self.max_time = resp.as_continuous().shape[-1] / resp.fs
        self.max_signal_time = self.max_time
        self._update_range()

    def tap_right(self):
        self.time_slider.set_value(
                self.time_slider.value + self.time_slider.singleStep
                )

    def tap_left(self):
        self.time_slider.set_value(
                self.time_slider.value - self.time_slider.singleStep
                )

    def set_display_range(self):
        duration = float(self.display_range.text())
        if not duration:
            print("Duration not set to a valid value. Please enter a"
                  "a number > 0")
            return
        self.display_duration = duration
        self._update_range()

    def increment_display_range(self):
        self.display_duration += 1
        self.display_range.setText(str(self.display_duration))
        self._update_range()

    def decrement_display_range(self):
        self.display_duration -= 1
        self.display_range.setText(str(self.display_duration))
        self._update_range()

    def _update_range(self):
        self.time_slider.setRange(0, self.max_time-self.display_duration)
        self.time_slider.setSingleStep(int(np.ceil(self.display_duration/10)))
        self.time_slider.setPageStep(int(self.display_duration))
        self.scroll_all()

























