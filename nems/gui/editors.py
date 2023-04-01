'''
Contains relevant classes for contsructing the NEMS model editor.

The different parts of the editor are layed out as follows:

------------------------------EditorWindow--------------------------------------
: :------------------------------ EditorWidget--------------------------------::
: : :-------------------ModelspecEditor-----------------: :---XfspecEditor---:::
: : : EpochsCollapser |                  | EpochCanvas  : :                  :::
: : :                                                   : :                  :::
: : : SignalCollapser | -SignalControls- | SignalCanvas : :                  :::
: : :                                                   : :                  :::
: : : ModuleCollapser | -ModuleControls- | ModuleCanvas : :   XfstepEditor   :::
: : :      [+/-]        :  PhiEditor   :      [Plot]    : :                  :::
: : :                   :..............:                : :                  :::
: : :          (1 of each per module in modelspec)      : :  (1 per step in  :::
: : :                                                   : :     the xfspec)  :::
: : :                         ...                       : :        ...       :::
: : :                                                   : :                  :::
: : :...................................................: :..................:::
: :                                                                           ::
: : ---------GlobalControls-----------------------------: :------FitEditor---:::
: : :  <----------------Plot Scrollbar--------------->  : :[Init Fn][Do Init]:::
: : : [Plot Display Duration]    [Zoom Out]   [Zoom In] : :[Fit Fn] [#] [Fit]:::
: : : [Reset Modelspec] [Set Fit index] [Set Cell Index]: :                  :::
: : :...................................................: :..................:::
: :...........................................................................::
:..............................................................................:

There are also buttons along the left, right, and bottom edges for toggling
the visibility of the ModelspecEditor, XfspecEditor, and the GlobalControls
& FitEditor, respectively.

Most modules will only need to refer to their immediate parent. However,
ModuleCollapsers reference the ModuleController and ModuleCanvas within
the same row in order to toggle their visibility. Additionally, GlobalControls
needs to reference many distantly related modules.

Classes
-------
EditorWindow(QMainWindow): Unpacks ctx if necessary and contains EditorWidget.
EditorWidget(QWidget): Keeps a reference to all data and lays out the
                       ModelspecEditor, XfspecEditor, GlobalControls, and
                       FitEditor.
ModelspecEditor(QWidget): Handles evaluation of modelspec and lays out a
                          ModuleCollapser, ModuleController, and ModuleCanvas
                          for each module in the modelspec.
ModuleCollapser(QWidget): Toggles visibility of one module.
ModuleController(QWidget): Controls for changing plot type, editing phi, etc.
                           for one module.
ModuleCanvas(QWidget): Plot display for one module.
EpochsCollapser(QWidget): Toggles visibilty of the epochs display.
EpochsWrapper(QWidget): Contains the EpochCanvas. Simple layout to maintain
                        consistency of appearance with module canvases.
XfspecEditor(QWidget): Handles evaluation of xfspec and lays out an XfstepEditor
                       for each step in the xfspec. Hidden by default.
                       (more features to be added later).
GlobalControls(QWidget): Provides user controls for adjusting plot limits and
                         changing fit and cell indices for modelspec
                         (more features to be added later)
FitEditor(QWidget): Provides user controls for re-initializing the modelspec
                    as well as incremental fitting
                    (more features to be added later).

'''

import sys
import copy
import json
import logging
from os.path import expanduser, join, dirname, exists

import numpy as np
import matplotlib
import PyQt5.QtWidgets as qw
import PyQt5.QtCore as qc
import PyQt5.QtGui as qg

app = qw.QApplication.instance()
if app is None:
    # if it does not exist then a QApplication is created
    app = qw.QApplication(sys.argv)

from nems0 import xforms
from nems0.gui.models import ArrayModel
from nems0.gui.canvas import NemsCanvas, EpochCanvas, MplWindow
from nems0.modelspec import _lookup_fn_at
import nems0.db as nd
from nems0.registry import keyword_lib
from nems0 import get_setting
import matplotlib.pyplot as plt
log = logging.getLogger(__name__)

from configparser import ConfigParser
import nems

configfile = join(nems0.get_setting('SAVED_SETTINGS_PATH') + '/gui.ini')

# These are used as click-once operations
# TODO: separate initialization from prefitting
_INIT_FNS = [
        'nems0.initializers.from_keywords',
        'nems0.initializers.prefit_LN'
        ]

# These can be repeated as needed in small steps
_FIT_FNS = [
        'nems0.analysis.fit_basic.fit_basic',
        'nems0.analysis.fit_iteratively.fit_iteratively'
        ]

# TODO: add backwards compatibility shim to add plot_fns, plot_fn_idx etc to
#       old modelspecs if none of the modules have those specified.

# TODO: Switch modelspec, xfspec etc. references to all just point to
#       EditorWidget copy instead of making separate copies.
#       Then all updates can use the most convenient
#       pointer instead of needing to call parent.parent.parent.modelspec



class EditorWindow(qw.QMainWindow):

    def __init__(self, modelspec=None, xfspec=None, rec=None, ctx=None,
                 rec_name='val', control_widget=None):
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
        if (modelspec is None) and (ctx is not None):
            modelspec = ctx.get('modelspec', None)
        if (rec is None) and (ctx is not None):
            rec = ctx.get(rec_name, None)
        if ctx is None:
            ctx={'modelspec':modelspec, 'rec': rec}
        self.editor = EditorWidget(modelspec, xfspec, rec, ctx, self, control_widget=control_widget)
        self.title = 'NEMS Model Browser: {}: {}'.format(
                ctx['meta']['cellid'],
                ctx['meta']['modelname'])
        self.setCentralWidget(self.editor)
        self.setWindowTitle(self.title)
        self.show()

    def closeEvent(self, QCloseEvent):
        self.editor.global_controls.save_view(on_close=True)


class EditorWidget(qw.QWidget):

    def __init__(self, modelspec=None, xfspec=None, rec=None, ctx=None,
                 parent=None, control_widget=None):
        '''
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
            A NEMS context dictionary (see nems0.xforms).
        parent : QWidget*
            Expected to be an instance of EditorWindow.

        '''
        super(qw.QWidget, self).__init__()
        self.xfspec = xfspec
        self.modelspec = modelspec
        self.rec = rec
        self.rec=self.rec.apply_mask(reset_epochs=True)
        self.modelspec.recording=self.rec
        if ctx is not None:
            meta = ctx['modelspec'].meta
            modelname = meta.get('modelname', 'n/a')
            batch = meta.get('batch', 0)
            cellid = meta.get('cellid', 'CELL')
            self.title = "%s  ||  %s  ||  %s  " % (modelname, cellid, batch)

        if ctx is None:
            self.ctx = {}
        else:
            self.ctx = ctx

        # By default, start with xfspec steps hidden but all other
        # controls showing.
        self.modules_collapsed = False
        self.xfsteps_collapsed = True
        self.bottom_collapsed = False

        # Layout is split up into:
        # First row: ModelspecEditor (left), XfspecEditor (right)
        # Second row: GlobalControls (left), FitEditor (right)
        # Third row: Bottom collapser
        outer_layout = qw.QVBoxLayout()
        row_one_layout = qw.QHBoxLayout()
        row_two_layout = qw.QHBoxLayout()
        row_three_layout = qw.QHBoxLayout()

        self.modelspec_editor = ModelspecEditor(modelspec, self.rec, self)
        if self.xfspec is None:
            self.xfspec = []
        self.xfspec_editor = XfspecEditor(self.xfspec, self)
        if control_widget is None:
            self.global_controls = GlobalControls(self)
        else:
            self.global_controls = control_widget
            self.global_controls.link_editor(self)

        self.fit_editor = FitEditor(self)

        self.modelspec_editor.setup_layout()
        # Have to set up these plots afterward to get
        # canvases to fill the layout properly.
        self.modelspec_editor.refresh_plots()
        self.modelspec_editor.epochs.setup_figure()

        self.setup_module_collapser()
        self.setup_xfstep_collapser()
        self.hide_xfstep_controls()
        self.setup_bottom_collapser()

        row_one_layout.addLayout(self.module_collapser_layout)
        row_one_layout.addWidget(self.modelspec_editor)
        row_one_layout.addWidget(self.xfspec_editor)
        row_one_layout.addLayout(self.xfstep_collapser_layout)
        if control_widget is None:
            row_two_layout.addWidget(self.global_controls)
        row_two_layout.addWidget(self.fit_editor)
        row_two_layout.setContentsMargins(10, 10, 10, 2)
        outer_layout.addLayout(row_one_layout)
        outer_layout.addLayout(row_two_layout)
        outer_layout.addLayout(self.bottom_collapser_layout)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(outer_layout)

        self.setWindowTitle(self.title)
        self.set_view_params()
        self.show()

    def set_view_params(self,view_file=None,config_group=None):
        if view_file is None:
            view_file = configfile
        if config_group is None:
            config_group = 'Model_Browser_' + self.ctx['meta']['modelname']

        config = ConfigParser()
        if not exists(view_file):
            log.warning(f"Config file {view_file} does not exist. Can't load view parameters")
            return
        config.read(view_file)
        if config_group not in config.sections():
            if view_file is not None:
                log.warning(f"Config file {view_file} does not contain a {config_group} section. Can't load view parameters")
            return
        else:
            log.info(f"Loading view params from {view_file}, {config_group} section.")
        try:
            time_range = eval(config.get(config_group, 'time_range'))
            plot_fn_idxs = eval(config.get(config_group, 'plot_fn_idxs'))
            plot_channels = eval(config.get(config_group, 'plot_channels'))
            collapsed = eval(config.get(config_group, 'collapsed'))
            signal_idxs = eval(config.get(config_group, 'signal_idxs'))
            signal_fn_idxs = eval(config.get(config_group, 'signal_fn_idxs'))
            signal_channels = eval(config.get(config_group, 'signal_channels'))
            signal_collapsed = eval(config.get(config_group, 'signal_collapsed'))

            self.global_controls.display_start.setText(str(time_range[0]))
            self.global_controls.display_range.setText(str(time_range[1] - time_range[0]))
            self.global_controls.set_display_range()

            for mod, pfi, pc in zip(self.modelspec_editor.modules, plot_fn_idxs, plot_channels):
                mod.plot_fn_idx = pfi
                mod.plot_channel = pc

            for cont, pfi, pc in zip(self.modelspec_editor.controllers, plot_fn_idxs, plot_channels):
                cont.plot_functions_menu.setCurrentIndex(pfi)
                cont.channel_entry.setText(str(pc))

            for cont, si, pfi, pc in zip(self.modelspec_editor.signal_controllers, signal_idxs, signal_fn_idxs, signal_channels):
                cont.signal_menu.setCurrentIndex(si)
                cont.plot_functions_menu.setCurrentIndex(pfi)
                cont.channel_entry.setText(str(pc))

            for coll, collapsed_ in zip(self.modelspec_editor.collapsers, collapsed):
                coll.collapsed = not collapsed_
                coll.toggle_collapsed()

            for coll, collapsed_ in zip(self.modelspec_editor.signal_collapsers, signal_collapsed):
                coll.collapsed = not collapsed_
                coll.toggle_collapsed()

        except Exception as inst:
            print('Error setting view parameters:')
            print(inst)

    def set_new_modelspec(self, new):
        '''Update modelspec references to new and evaluate it.'''
        self.modelspec = new
        self.ctx['modelspec'] = new
        self.modelspec.recording = self.rec
        self.modelspec_editor.modelspec = new
        self.modelspec_editor.modelspec.recording = self.rec
        self.modelspec_editor.evaluate_model()

    def setup_module_collapser(self):
        '''Add button to left of ModelspecEditor for hiding module controls.'''
        self.module_collapser_layout = qw.QVBoxLayout()

        self.module_collapser = qw.QToolButton(self)
        self.module_collapser.setMaximumWidth(15)
        self.module_collapser.clicked.connect(self.toggle_module_controls)
        self.module_collapser.setArrowType(qc.Qt.LeftArrow)
        policy = qw.QSizePolicy()
        policy.setVerticalPolicy(qw.QSizePolicy.Expanding)
        self.module_collapser.setSizePolicy(policy)

        self.module_collapser_layout.addWidget(self.module_collapser)

    def setup_xfstep_collapser(self):
        '''Add button to right of XfspecEditor for hiding xfstep controls.'''
        self.xfstep_collapser_layout = qw.QVBoxLayout()

        self.xfstep_collapser = qw.QToolButton(self)
        self.xfstep_collapser.setMaximumWidth(15)
        self.xfstep_collapser.clicked.connect(self.toggle_xfstep_controls)
        self.xfstep_collapser.setArrowType(qc.Qt.RightArrow)
        policy = qw.QSizePolicy()
        policy.setVerticalPolicy(qw.QSizePolicy.Expanding)
        self.xfstep_collapser.setSizePolicy(policy)

        self.xfstep_collapser_layout.addWidget(self.xfstep_collapser)

    def setup_bottom_collapser(self):
        '''Add button to bottom of window for hiding Global and Fit controls.'''
        self.bottom_collapser_layout = qw.QHBoxLayout()

        self.bottom_collapser = qw.QToolButton(self)
        self.bottom_collapser.setMaximumHeight(15)
        self.bottom_collapser.clicked.connect(self.toggle_bottom_controls)
        self.bottom_collapser.setArrowType(qc.Qt.DownArrow)
        policy = qw.QSizePolicy()
        policy.setHorizontalPolicy(qw.QSizePolicy.Expanding)
        self.bottom_collapser.setSizePolicy(policy)

        self.bottom_collapser_layout.addWidget(self.bottom_collapser)

    def toggle_module_controls(self):
        '''Hide module controls if visible, or show them if hidden.'''
        if self.modules_collapsed:
            self.show_module_controls()
        else:
            self.hide_module_controls()
        self.modules_collapsed = not self.modules_collapsed

    def hide_module_controls(self):
        '''Make module controls invisible.'''
        collapsers = self.modelspec_editor.collapsers
        controllers = self.modelspec_editor.controllers
        for col, con in zip(collapsers, controllers):
            con.hide()
        self.module_collapser.setArrowType(qc.Qt.RightArrow)

    def show_module_controls(self):
        '''Make module controls visible.'''
        collapsers = self.modelspec_editor.collapsers
        controllers = self.modelspec_editor.controllers
        for col, con in zip(collapsers, controllers):
            if not col.collapsed:
                con.show()
        self.module_collapser.setArrowType(qc.Qt.LeftArrow)

    def toggle_xfstep_controls(self):
        '''Hide xfstep controls if visible, or show them if hidden.'''
        if self.xfsteps_collapsed:
            self.show_xfstep_controls()
        else:
            self.hide_xfstep_controls()
        self.xfsteps_collapsed = not self.xfsteps_collapsed

    def hide_xfstep_controls(self):
        '''Make xfstep controls invisible.'''
        for s in self.xfspec_editor.steps:
            s.hide()
        self.xfstep_collapser.setArrowType(qc.Qt.LeftArrow)

    def show_xfstep_controls(self):
        '''Make xfstep controls visible.'''
        for s in self.xfspec_editor.steps:
            s.show()
        self.xfstep_collapser.setArrowType(qc.Qt.RightArrow)

    def toggle_bottom_controls(self):
        '''Hide Global and Fit controls if visible, or show them if hidden.'''
        if self.bottom_collapsed:
            self.fit_editor.show()
            self.global_controls.toggle_controls()
            self.bottom_collapser.setArrowType(qc.Qt.DownArrow)
        else:
            self.fit_editor.hide()
            self.global_controls.toggle_controls()
            self.bottom_collapser.setArrowType(qc.Qt.UpArrow)
        self.bottom_collapsed = not self.bottom_collapsed


class ModelspecEditor(qw.QWidget):
    def __init__(self, modelspec, rec, parent=None):
        '''
        QWidget for displaying per-module plots and editing model parameters.

        Parameters:
        -----------
        modelspec : ModelSpec
            A NEMS ModelSpec containing at least one module.
        rec : Recording
            A NEMS Recording, generally expected to contain 'stim', 'resp',
            and 'pred' signals.
        parent : QWidget*
            Expected to be an instance of EditorWidget.

        Methods:
        --------
        refresh_plots: Re-generate all module plots.
        evaluate_model: Use modelspec.evaluate() to update modelspec's recording,
                        then re-generate module plots and update EditorWidget's
                        modelspec recording as well.
        reset_model: Assign copy of original modelspec (whatever was present
                     when the application was launched), then refresh
                     module plots and update EditorWidget's modelspec.

        '''
        super(qw.QWidget, self).__init__()
        self.modelspec = modelspec
        self.original_modelspec = copy.deepcopy(modelspec)
        self.rec = rec
        self.parent = parent
        self.modules = []
        self.signal_displays = []

    def setup_layout(self):
        '''
        Create a new layout with collapsers, controllers and plots

        Start a grid layout with an EpochsCollapser and EpochsWrapper
        on the first row.
        Then, for each module in self.modelspec, create a row with:
        [ModuleCollapser, ModuleController, ModuleCanvas],
        and add it to the grid layout.

        '''
        self.layout = qw.QGridLayout()
        self.modules = [ModuleCanvas(i, m, self)
                        for i, m in enumerate(self.modelspec.modules)]
        self.controllers = [ModuleControls(m, self) for m in self.modules]
        self.collapsers = [ModuleCollapser(m, self) for m in self.modules]

        signal = list(self.rec.signals.keys())[0]
        self.signal_displays = [SignalCanvas(self.rec, signal, self)]
        self.signal_controllers = [SignalControls(s, self) for s in self.signal_displays]
        self.signal_collapsers = [SignalCollapser(s, c, self)
                                  for s, c in zip(self.signal_displays, self.signal_controllers)]

        # epochs at top
        self.epochs = EpochsWrapper(
            recording=self.rec,
            parent=self.parent.global_controls
        )
        self.epochs_collapser = EpochsCollapser(self.epochs, self)
        self.layout.addWidget(self.epochs_collapser, 0, 0)
        self.layout.addWidget(self.epochs, 0, 2)
        j = 1

        # then signals
        swidgets = zip(self.signal_collapsers, self.signal_controllers, self.signal_displays)
        for col, cnt, m in swidgets:
            self.layout.addWidget(col, j, 0)
            self.layout.addWidget(cnt, j, 1)
            self.layout.addWidget(m, j, 2)
            j += 1

        # then all the modules
        widgets = zip(self.collapsers, self.controllers, self.modules)
        for col, cnt, m in widgets:
            self.layout.addWidget(col, j, 0)
            self.layout.addWidget(cnt, j, 1)
            self.layout.addWidget(m, j, 2)
            j += 1

        self.layout.setAlignment(qc.Qt.AlignTop)
        self.layout.setColumnStretch(2, 4)
        self.setLayout(self.layout)

    def refresh_plots(self):
        '''Regenerate plot for each module and signal.'''
        for m in self.modules:
            if m.parent.modelspec[m.mod_index]['plot_fns']\
                    [m.parent.modelspec[m.mod_index]['plot_fn_idx']] != 'nems0.plots.api.null':
                m.new_plot()

        for m in self.signal_displays:
            m.new_plot()

    def evaluate_model(self):
        '''Evaluate modelspec and regenerate plots using updated recording.'''
        new_rec = self.parent.modelspec.evaluate()
        self.parent.modelspec.recording = new_rec
        self.modelspec.recording = new_rec
        self.refresh_plots()

    def reset_model(self):
        '''Reassign modelspec to original copy and regenerate layout.'''
        self.modelspec = copy.deepcopy(self.original_modelspec)
        self.parent.modelspec = self.modelspec
        self.clear_layout()
        self.setup_layout()
        self.refresh_plots()
        self.epochs.setup_figure()

    def clear_layout(self):
        temp = qw.QWidget()
        temp.setLayout(self.layout)


class ModuleCanvas(qw.QFrame):

    def __init__(self, mod_index, data, parent):
        '''
        QWidget for dispalying a module plot

        Parameters
        ----------
        mod_index : int
            Index of this canvas's module within parent editor's ModelSpec
        data : dict
            A single module from a NEMS ModelSpec.
        parent : QWidget*
            Expected to be an instance of ModelspecEditor.

        Attributes
        ----------
        plot_fn_idx : int
            Index for plot_fns list in data to use for choosing a plot type.
        fit_index : int
            fit_index from parent editor's modelspec.
        sig_name : str
            Name of signal to use for plotting.
        scrollable : boolean
            Determines whether or not plot can be scrolled in time by
            the GlobalControls slider.

        '''
        super(qw.QFrame, self).__init__()
        self.mod_index = mod_index
        self.parent = parent
        self.setFrameStyle(qw.QFrame.Panel | qw.QFrame.Sunken)
        self.highlight_obj = None

        # define plot_list here rather than in ModuleControls
        self.plot_list = self.parent.modelspec[self.mod_index].get('plot_fns', [])
        mod_id = self.parent.modelspec[self.mod_index].get('id', [])
        template = keyword_lib[mod_id]
        registry_plot_fns = template.get('plot_fns', [])
        for a in registry_plot_fns:
            if a not in self.plot_list:
                self.plot_list.append(a)
        self.parent.modelspec[self.mod_index]['plot_fns'] = self.plot_list

        # Default plot options - set them up here then change w/ controller
        self.plot_fn_idx = data.get('plot_fn_idx', 0)
        self.plot_channel = parent.modelspec.plot_channel
        self.fit_index = parent.modelspec.fit_index
        # TODO: Need to do something smarter for signal name
        self.sig_name = self.parent.modelspec[self.mod_index]['fn_kwargs'].get('o','pred')
        self.scrollable = self.check_scrollable()

        self.layout = qw.QHBoxLayout()
        self.canvas = qw.QWidget()
        self.layout.addWidget(self.canvas)
        self.layout.setAlignment(qc.Qt.AlignTop)
        self.setLayout(self.layout)
        # self.axes = None

    def new_plot(self):
        '''Remove plot from layout and replace it with a new one.'''
        self.layout.removeWidget(self.canvas)
        self.highlight_obj = None
        self.canvas.close()
        if hasattr(self.canvas,'figure'):
            plt.close(self.canvas.figure)
        self.canvas = NemsCanvas(parent=self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.plot_on_axes()
        self.layout.addWidget(self.canvas)
        self.scrollable = self.check_scrollable()
        self.update_plot()

    def onclick(self, event):
        #print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #      ('double' if event.dblclick else 'single', event.button,
        #       event.x, event.y, event.xdata, event.ydata))
        t = round(event.xdata,3)
        #self.parent.parent.global_controls.cursor_time = t
        #self.parent.parent.global_controls.cursor_time_line.setText(str(t))
        self.parent.parent.global_controls.update_cursor_time(t)

    def plot_on_axes(self):
        '''Draw plot on current canvas axes.'''
        gc = self.parent.parent.global_controls
        #if self.axes is None:
        self.canvas.axes = self.canvas.figure.add_subplot(111, label='module'+str(self.mod_index))
        ax = self.canvas.axes
        #ax.clear()
        rec = self.parent.modelspec.recording
        self.parent.modelspec[self.mod_index]['plot_fn_idx']=self.plot_fn_idx
        self.parent.modelspec.plot(mod_index=self.mod_index, rec=rec, ax=ax,
                                   plot_fn_idx=self.plot_fn_idx, fit_index=self.fit_index,
                                   sig_name=self.sig_name, no_legend=True,
                                   channels=self.plot_channel,
                                   cursor_time=gc.cursor_time,
                                   linewidth=1, fontsize=8)
        if self.scrollable:
            cid = self.canvas.mpl_connect('button_press_event', self.onclick)

        self.canvas.draw()

    def check_scrollable(self):
        '''Set self.scrollable based on if current plot type is scrollable.'''
        #plots = self.parent.modelspec[self.mod_index].get(
        #        'plot_fns', ['nems0.plots.api.mod_output']
        #        )
        fn_ref = _lookup_fn_at(self.plot_list[self.plot_fn_idx])
        if ('scrollable' in dir(fn_ref)) and fn_ref.scrollable:
            scrollable = True
        else:
            scrollable = False
        return scrollable

    def update_plot(self):
        '''Shift xlimits of current plot if it's scrollable.'''

        if self.scrollable:

            gc = self.parent.parent.global_controls

            self.canvas.axes.set_xlim(gc.start_time, gc.stop_time)
            self.canvas.draw()
        else:
            pass

    def update_cursor(self):
        '''plot/move cursor bar on scrollable plots, adjust other plots that depend on cursor value.'''
        gc = self.parent.parent.global_controls
        fn_ref = _lookup_fn_at(self.plot_list[self.plot_fn_idx])
        if self.scrollable:
            self.canvas.axes.set_xlim(gc.start_time, gc.stop_time)

            h_times = np.ones(2) * gc.cursor_time
            if self.highlight_obj is None:
                ylim = self.canvas.axes.get_ylim()
                self.highlight_obj = self.canvas.axes.plot(h_times, ylim, 'r-')[0]
            else:
                self.highlight_obj.set_xdata(h_times)

            self.canvas.draw()
        elif ('cursor' in dir(fn_ref)) and fn_ref.cursor:
            print('update plot {}: {} cursor time={:.3f}'.format(
                self.mod_index, self.plot_list[self.plot_fn_idx], gc.cursor_time))
            self.new_plot()
        else:
            pass

    def change_channel(self, value):
        self.plot_channel = int(value)
        self.new_plot()


class SignalCanvas(qw.QFrame):

    def __init__(self, rec, sig_name, parent):
        '''
        QWidget for displaying a signal

        Parameters
        ----------
        rec : Recording object (typically inherited from parent)
            NEMS recording object
        sig_name : str
            Name of signal to plot
        parent : QWidget*
            Expected to be an instance of ModelspecEditor.

        Attributes
        ----------
        plot_fn_idx : int
            Index for plot_fns list in data to use for choosing a plot type.
        scrollable : boolean
            Determines whether or not plot can be scrolled in time by
            the GlobalControls slider.

        '''
        super(qw.QFrame, self).__init__()
        self.rec = rec
        self.parent = parent
        self.sig_name = sig_name
        self.setFrameStyle(qw.QFrame.Panel | qw.QFrame.Sunken)
        self.highlight_obj = None

        # define plot_list here rather than in ModuleControls
        self.plot_list = ['nems0.plots.api.spectrogram',
                          'nems0.plots.api.timeseries_from_signals']

        # Default plot options - set them up here then change w/ controller
        self.plot_fn_idx = 0
        self.plot_channel = parent.modelspec.plot_channel
        self.scrollable = self.check_scrollable()

        self.layout = qw.QHBoxLayout()
        self.canvas = qw.QWidget()
        self.layout.addWidget(self.canvas)
        self.layout.setAlignment(qc.Qt.AlignTop)
        self.setLayout(self.layout)

    def new_plot(self):
        '''Remove plot from layout and replace it with a new one.'''
        self.layout.removeWidget(self.canvas)
        self.highlight_obj = None
        self.canvas.close()
        if hasattr(self.canvas, 'figure'):
            plt.close(self.canvas.figure)
        self.canvas = NemsCanvas(parent=self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.plot_on_axes()
        self.layout.addWidget(self.canvas)
        self.scrollable = self.check_scrollable()

        self.update_plot()

    def onclick(self, event):
        #print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #      ('double' if event.dblclick else 'single', event.button,
        #       event.x, event.y, event.xdata, event.ydata))
        t = round(event.xdata,3)
        #self.parent.parent.global_controls.cursor_time = t
        #self.parent.parent.global_controls.cursor_time_line.setText(str(t))
        self.parent.parent.global_controls.update_cursor_time(t)

    def plot_on_axes(self):
        '''Draw plot on current canvas axes.'''
        gc = self.parent.parent.global_controls
        self.canvas.axes = self.canvas.figure.add_subplot(111)
        ax = self.canvas.axes
        rec = self.rec
        from nems0.modelspec import _lookup_fn_at
        plot_fn = _lookup_fn_at(self.plot_list[self.plot_fn_idx])
        plot_fn(rec=rec, modelspec=self.parent.modelspec, sig_name=self.sig_name,
                channels=self.plot_channel, no_legend=True,
                ax=ax, cursor_time=gc.cursor_time)

        if self.scrollable:
            cid = self.canvas.mpl_connect('button_press_event', self.onclick)

        self.canvas.draw()

    def check_scrollable(self):
        '''Set self.scrollable based on if current plot type is scrollable.'''
        fn_ref = _lookup_fn_at(self.plot_list[self.plot_fn_idx])
        if ('scrollable' in dir(fn_ref)) & fn_ref.scrollable:
            scrollable = True
        else:
            scrollable = False
        return scrollable

    def update_plot(self):
        '''Shift xlimits of current plot if it's scrollable.'''

        if self.scrollable:
            gc = self.parent.parent.global_controls
            self.canvas.axes.set_xlim(gc.start_time, gc.stop_time)
            self.canvas.draw()
        else:
            pass

    def update_cursor(self):
        '''plot/move cursor bar on scrollable plots, adjust other plots that depend on cursor value.'''
        gc = self.parent.parent.global_controls
        fn_ref = _lookup_fn_at(self.plot_list[self.plot_fn_idx])
        if self.scrollable:
            self.canvas.axes.set_xlim(gc.start_time, gc.stop_time)

            h_times = np.ones(2) * gc.cursor_time
            if self.highlight_obj is None:
                ylim = self.canvas.axes.get_ylim()
                self.highlight_obj = self.canvas.axes.plot(h_times, ylim, 'r-')[0]
            else:
                self.highlight_obj.set_xdata(h_times)

            self.canvas.draw()
        elif ('cursor' in dir(fn_ref)) and fn_ref.cursor:
            print('update plot {}: {} cursor time={:.3f}'.format(
                self.sig_name, self.plot_list[self.plot_fn_idx], gc.cursor_time))
            self.new_plot()
        else:
            pass

    def change_channel(self, value):
        self.plot_channel = int(value)
        self.new_plot()



class EpochsWrapper(qw.QFrame):

    def __init__(self, recording=None, parent=None):
        '''Adds a layout around an EpochCanvas to match ModuleCanvas.'''
        super(qw.QFrame, self).__init__()
        self.recording = recording
        self.epoch_parent = parent

    def setup_figure(self):
        '''Create a layout and add an EpochCanvas.'''
        self.layout = qw.QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        print('creating EpochCanvas')
        self.epochs = EpochCanvas(recording=self.recording,
                                  parent=self.epoch_parent)
        self.layout.addWidget(self.epochs)
        self.setLayout(self.layout)
        self.setFrameStyle(qw.QFrame.Panel | qw.QFrame.Sunken)

    def update_figure(self):
        '''Shifts xlim of epoch plot according to GlobalControls slider.'''
        self.epochs.update_figure()


class ModuleCollapser(qw.QWidget):

    def __init__(self, module, parent):
        '''Button for controlling visibility of associated module in editor.'''
        super(qw.QWidget, self).__init__()
        self.module = module
        self.controller = parent.controllers[module.mod_index]
        self.parent = parent
        self.collapsed = False

        layout = qw.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.toggle = qw.QPushButton('-', self)
        self.toggle.setFixedSize(12, 12)
        self.toggle.clicked.connect(self.toggle_collapsed)
        layout.addWidget(self.toggle)
        layout.setAlignment(qc.Qt.AlignTop)
        self.setLayout(layout)

    def toggle_collapsed(self):
        '''Toggle visibility of ModuleCanvas and ModuleController.'''
        if self.collapsed:
            self.module.show()
            if not self.parent.parent.modules_collapsed:
                self.controller.show()
            self.toggle.setText('-')
        else:
            self.module.hide()
            self.controller.hide()
            self.toggle.setText('+')
        self.collapsed = not self.collapsed


class EpochsCollapser(qw.QWidget):

    def __init__(self, epochs, parent):
        '''Button for controlling visibility of EpochsWrapper.'''
        super(qw.QWidget, self).__init__()
        self.epochs = epochs
        self.parent = parent
        self.collapsed = False

        layout = qw.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.toggle = qw.QPushButton('-', self)
        self.toggle.setFixedSize(12, 12)
        self.toggle.clicked.connect(self.toggle_collapsed)
        layout.addWidget(self.toggle)
        layout.setAlignment(qc.Qt.AlignTop)
        self.setLayout(layout)

    def toggle_collapsed(self):
        '''Toggle visibility of EpochsWrapper.'''
        if self.collapsed:
            self.epochs.show()
            self.toggle.setText('-')
        else:
            self.epochs.hide()
            self.toggle.setText('+')
        self.collapsed = not self.collapsed


class SignalCollapser(qw.QWidget):

    def __init__(self, signal_display, signal_controller, parent=None):
        '''Button for controlling visibility of a signal .'''
        super(qw.QWidget, self).__init__()
        self.rec = parent.rec
        self.signal_display = signal_display
        self.controller = signal_controller
        self.parent = parent
        self.collapsed = False

        layout = qw.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.toggle = qw.QPushButton('-', self)
        self.toggle.setFixedSize(12, 12)
        self.toggle.clicked.connect(self.toggle_collapsed)
        layout.addWidget(self.toggle)
        layout.setAlignment(qc.Qt.AlignTop)
        self.setLayout(layout)

    def toggle_collapsed(self):
        '''Toggle visibility of ModuleCanvas and ModuleController.'''
        if self.collapsed:
            self.signal_display.show()
            if not self.parent.parent.modules_collapsed:
                self.controller.show()
            self.toggle.setText('-')
        else:
            self.signal_display.hide()
            self.controller.hide()
            self.toggle.setText('+')
        self.collapsed = not self.collapsed


class ModuleControls(qw.QFrame):

    def __init__(self, module, parent=None):
        '''QWidget for choosing module plot type and editing parameters.'''
        super(qw.QFrame, self).__init__()
        self.module = module
        self.parent = parent
        self.mod_index = self.module.mod_index
        self.module_data = copy.deepcopy(
                self.module.parent.modelspec[self.mod_index]
                )
        self.setFrameStyle(qw.QFrame.Panel | qw.QFrame.Raised)

        self.layout = qw.QVBoxLayout()

        name = self.module_data['fn']
        self.label = qw.QLabel(name)
        self.label.setFixedSize(330, 20)
        self.label.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.layout.addWidget(self.label)

        plot_layout = qw.QHBoxLayout()

        self.plot_functions_menu = qw.QComboBox()
        self.plot_functions_menu.addItems(self.module.plot_list)
        initial_index = self.module.plot_fn_idx
        if initial_index is None:
            initial_index = 0
        self.plot_functions_menu.setCurrentIndex(initial_index)
        self.plot_functions_menu.setFixedSize(250, 24)

        plot_channel_layout = qw.QHBoxLayout()
        self.decrease_channel_btn = qw.QPushButton('-')
        self.decrease_channel_btn.clicked.connect(self.decrease_channel)
        self.decrease_channel_btn.setFixedSize(15, 15)
        self.channel_entry = qw.QLineEdit(str(self.module.plot_channel))
        self.channel_entry.textChanged.connect(self.change_channel)
        self.channel_entry.setFixedSize(30, 15)
        self.increase_channel_btn = qw.QPushButton('+')
        self.increase_channel_btn.clicked.connect(self.increase_channel)
        self.increase_channel_btn.setFixedSize(15, 15)
        plot_channel_layout.addWidget(self.decrease_channel_btn)
        plot_channel_layout.addWidget(self.channel_entry)
        plot_channel_layout.addWidget(self.increase_channel_btn)

        plot_layout.addWidget(self.plot_functions_menu)
        plot_layout.addLayout(plot_channel_layout)
        self.layout.addLayout(plot_layout)
        self.plot_functions_menu.currentIndexChanged.connect(self.change_plot)

        button_layout = qw.QHBoxLayout()
        self.edit_phi_btn = qw.QPushButton('Edit Phi')
        self.edit_phi_btn.clicked.connect(self.edit_phi)
        self.reset_phi_btn = qw.QPushButton('Reset Phi')
        self.reset_phi_btn.clicked.connect(self.reset_phi)
        button_layout.addWidget(self.edit_phi_btn)
        button_layout.addWidget(self.reset_phi_btn)
        self.layout.addLayout(button_layout)

        self.phi_editor = PhiEditor(self.module_data.get('phi', {}), self)
        self.phi_editor.hide()
        self.save_phi_btn = qw.QPushButton('Save Phi')
        self.save_phi_btn.hide()
        self.save_phi_btn.clicked.connect(self.save_phi)
        self.layout.addWidget(self.phi_editor)
        self.layout.addWidget(self.save_phi_btn)

        self.layout.setAlignment(qc.Qt.AlignTop)
        self.layout.setSpacing(2)
        self.setLayout(self.layout)

    def change_plot(self, index):
        '''Change plot type according to plot_fn_idx and regenerate plot.'''
        self.module.plot_fn_idx = int(index)
        self.module.new_plot()

    def edit_phi(self):
        '''Show PhiEditor and 'Save Phi' button.'''
        self.phi_editor.show()
        self.save_phi_btn.show()

    def save_phi(self):
        '''Save phi values in PhiEditor and hide 'Save Phi' button.'''
        new_phi = self.phi_editor.export_phi()
        if not self.phi_equal(new_phi):
            need_evaluate = True
        else:
            need_evaluate = False

        self.parent.modelspec[self.mod_index]['phi'] = copy.deepcopy(new_phi)
        self.phi_editor.hide()
        self.save_phi_btn.hide()

        # Only re-evaluate if phi has changed from previous version.
        if need_evaluate:
            self.parent.evaluate_model()
            self.module_data['phi'] = copy.deepcopy(new_phi)

    def reset_phi(self):
        '''Set all phi for this module to original values.'''
        self.phi_editor.reset_phi()
        self.save_phi()

    def phi_equal(self, phi2):
        '''Test equality of module phi and phi2.'''
        equal = True
        phi1 = self.parent.modelspec[self.mod_index]['phi']
        for v1, v2 in zip(phi1.values(), phi2.values()):
            if not np.array_equal(v1, v2):
                equal = False
                break
        return equal

    def change_channel(self):
        old_channel = self.module.plot_channel
        new_channel = self.channel_entry.text()
        try:
            self.module.change_channel(int(new_channel))
        except:
            # Leaving this bare b/c not clear what exception type it will be.
            # But reason is if plot channel is not valid for whichever
            # plot function the module is using.
            log.warning("Invalid plot channel: %s for module: %s"
                        % (new_channel, self.mod_index))
            self.channel_entry.setText(str(old_channel))

    def decrease_channel(self):
        old_channel = self.module.plot_channel
        self.channel_entry.setText(str(max(0, old_channel-1)))

    def increase_channel(self):
        # TODO: Would be nice to know a valid channel range but
        #       I'm not sure how to extract that information from the modules.
        old_channel = self.module.plot_channel
        self.channel_entry.setText(str((old_channel+1)))


class PhiEditor(qw.QWidget):

    def __init__(self, phi, parent):
        '''QWidget for editing module parameters.'''
        super(qw.QWidget, self).__init__(parent)
        self.phi = phi
        self.original_phi = copy.deepcopy(phi)
        self.parent = parent

        self.setup_layout()

    def setup_layout(self):
        '''Add a layout with an ArrayModel for each phi entry.'''
        self.layout = qw.QFormLayout()
        self.arrays = {}
        for k, v in self.phi.items():
            label = qw.QLabel(k)
            array = ArrayModel(self, v)
            self.arrays[k] = array
            self.layout.addRow(label, array)

        self.layout.setAlignment(qc.Qt.AlignTop)
        self.setLayout(self.layout)

    def reset_phi(self):
        '''Reset all phi entries to their original values.'''
        self.phi = copy.deepcopy(self.original_phi)
        # remove old layout
        temp = qw.QWidget()
        temp.setLayout(self.layout)
        # recreate layout with new phi
        self.setup_layout()

    def export_phi(self):
        '''Return all phi entries and their current values as a dict.'''
        return {k: v.export_array() for k, v in self.arrays.items()}


class SignalControls(qw.QFrame):

    def __init__(self, signal_display, parent=None):
        '''QWidget for choosing signal to plot and plot type.'''
        super(qw.QFrame, self).__init__()
        self.signal_display = signal_display
        self.parent = parent
        self.signal_list = list(parent.rec.signals.keys())
        self.sig_name = signal_display.sig_name
        self.setFrameStyle(qw.QFrame.Panel | qw.QFrame.Raised)

        self.layout = qw.QVBoxLayout()

        sig_name = self.sig_name
        self.label = qw.QLabel(sig_name)
        self.label.setFixedSize(330, 20)
        self.label.setStyleSheet("background-color: rgb(255, 255, 255);")
        #self.layout.addWidget(self.label)

        plot_layout = qw.QHBoxLayout()

        self.signal_menu = qw.QComboBox()
        self.signal_menu.addItems(self.signal_list)
        initial_index = self.signal_list.index(sig_name)
        if initial_index is None:
            initial_index = 0
        self.signal_menu.setCurrentIndex(initial_index)
        self.signal_menu.setFixedSize(250, 24)
        self.signal_menu.currentIndexChanged.connect(self.change_signal)

        self.plot_functions_menu = qw.QComboBox()
        self.plot_functions_menu.addItems(self.signal_display.plot_list)
        initial_index = self.signal_display.plot_fn_idx
        if initial_index is None:
            initial_index = 0
        self.plot_functions_menu.setCurrentIndex(initial_index)
        self.plot_functions_menu.setFixedSize(250, 24)
        self.plot_functions_menu.currentIndexChanged.connect(self.change_plot)

        plot_channel_layout = qw.QHBoxLayout()
        self.decrease_channel_btn = qw.QPushButton('-')
        self.decrease_channel_btn.clicked.connect(self.decrease_channel)
        self.decrease_channel_btn.setFixedSize(15, 15)
        self.channel_entry = qw.QLineEdit(str(self.signal_display.plot_channel))
        self.channel_entry.textChanged.connect(self.change_channel)
        self.channel_entry.setFixedSize(30, 15)
        self.increase_channel_btn = qw.QPushButton('+')
        self.increase_channel_btn.clicked.connect(self.increase_channel)
        self.increase_channel_btn.setFixedSize(15, 15)
        plot_channel_layout.addWidget(self.decrease_channel_btn)
        plot_channel_layout.addWidget(self.channel_entry)
        plot_channel_layout.addWidget(self.increase_channel_btn)

        self.layout.addWidget(self.signal_menu)
        plot_layout.addWidget(self.plot_functions_menu)
        plot_layout.addLayout(plot_channel_layout)
        self.layout.addLayout(plot_layout)

        self.layout.setAlignment(qc.Qt.AlignTop)
        self.setLayout(self.layout)

    def change_signal(self, index):
        '''Change change plot to new signal.'''
        self.sig_name = self.signal_list[index]

        # avoid crashing b/c new signal doesn't have enough channels
        if self.signal_display.rec[self.sig_name].shape[0]<self.signal_display.plot_channel:
            self.channel_entry.setText(str(0))
            self.signal_display.plot_channel = 0
            self.signal_display.change_channel(0)

        self.signal_display.sig_name = self.sig_name
        self.signal_display.new_plot()

        self.label.setText(self.sig_name)

    def change_plot(self, index):
        '''Change plot type according to plot_fn_idx and regenerate plot.'''
        self.signal_display.plot_fn_idx = int(index)
        self.signal_display.new_plot()

    def change_channel(self):
        old_channel = self.signal_display.plot_channel
        new_channel = self.channel_entry.text()
        try:
            self.signal_display.change_channel(int(new_channel))
        except:
            # Leaving this bare b/c not clear what exception type it will be.
            # But reason is if plot channel is not valid for whichever
            # plot function the module is using.
            log.warning("Invalid plot channel: %s for signal: %s"
                        % (new_channel, self.sig_name))
            self.channel_entry.setText(str(old_channel))

    def decrease_channel(self):
        old_channel = self.signal_display.plot_channel
        self.channel_entry.setText(str(max(0, old_channel-1)))

    def increase_channel(self):
        # TODO: Would be nice to know a valid channel range but
        #       I'm not sure how to extract that information from the modules.
        old_channel = self.signal_display.plot_channel
        self.channel_entry.setText(str((old_channel+1)))


class XfspecEditor(qw.QWidget):

    def __init__(self, xfspec, parent=None):
        '''(TODO)QWidget for editing xfspec options and toggling steps.'''
        super(qw.QWidget, self).__init__()
        self.xfspec = xfspec
        self.original_xfspec = copy.deepcopy(xfspec)
        self.parent = parent

        self.outer_layout = qw.QHBoxLayout()

        self.steps = [XfStepEditor(i, s, self)
                      for i, s in enumerate(self.xfspec)]
        self.step_layout = qw.QVBoxLayout()
        [self.step_layout.addWidget(s) for s in self.steps]

        self.outer_layout.addLayout(self.step_layout)
        self.setLayout(self.outer_layout)

    def filtered_xfspec(self):
        '''Return a nested list containing only xfsteps that are checked.'''
        checks = [s.checked for s in self.steps]
        x = [s for s, c in zip(self.xfspec, checks) if c]
        return x


class XfStepEditor(qw.QFrame):

    def __init__(self, index, step, parent):
        '''(TODO)QWidget for editing the parameters of one xforms step.'''
        super(qw.QFrame, self).__init__()
        self.index = index
        self.step = step
        self.parent = parent
        self.checked = True
        self.setFrameStyle(qw.QFrame.Panel | qw.QFrame.Raised)

        layout = qw.QVBoxLayout()
        self.checkbox = qw.QCheckBox(self.step[0], self)
        self.checkbox.setCheckState(qc.Qt.Checked)
        self.checkbox.stateChanged.connect(self.toggle)
        layout.addWidget(self.checkbox)
        self.setLayout(layout)

    def toggle(self):
        self.checked = not self.checked


class GlobalControls(qw.QFrame):

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
        self.editors = [parent]
        self.collapsed = False
        self.setFrameStyle(qw.QFrame.Panel | qw.QFrame.Raised)

        # Slider for plot view windows
        self.time_slider = qw.QScrollBar(orientation=1)
        policy = qw.QSizePolicy()
        policy.setHorizontalPolicy(qw.QSizePolicy.Expanding)

        # Set start for plot views
        self.display_start = qw.QLineEdit()
        self.display_start.setValidator(
                qg.QDoubleValidator(0, 10000.0, 4)
                )
        self.display_start.editingFinished.connect(self.set_display_range)
        self.display_start.setText(str(self.start_time))

        self.time_slider.setSizePolicy(policy)
        self._update_max_time()
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
        [self.range_layout.addWidget(w) for w in [qw.QLabel('Start(s)'), self.display_start, qw.QLabel('Dur(s)'), self.display_range, plus, minus]]

        self.buttons_layout = qw.QHBoxLayout()
        self.buttons_layout.setAlignment(qc.Qt.AlignTop)
        self.reset_model_btn = qw.QPushButton('Reset Model')
        self.reset_model_btn.clicked.connect(self.reset_model)

        self.fit_index_label = qw.QLabel('Fit Index')
        self.fit_index_line = qw.QLineEdit()
        self.fit_index_line.editingFinished.connect(self.update_fit_index)
        self.fit_index_line.setText(str(self.parent.modelspec_editor.modelspec.fit_index))

        self.cell_index_label = qw.QLabel('Cell')
        self.cell_index_line = qw.QLineEdit()
        self.cell_index_line.editingFinished.connect(self.update_cell_index)
        self.cell_index_line.setText(str(self.parent.modelspec.plot_channel))
        #self.cell_index_line.setText(str(self.parent.modelspec_editor.modelspec.cell_index))

        self.cursor_time = 0
        self.cursor_time_label = qw.QLabel('Cursor')
        self.cursor_time_line = qw.QLineEdit()
        self.cursor_time_line.editingFinished.connect(self.update_cursor_time)
        self.cursor_time_line.setText(str(self.cursor_time))

        self.quickplot_btn = qw.QPushButton('Save Quickplot')
        self.quickplot_btn.clicked.connect(self.export_plot)

        self.save_view_btn = qw.QPushButton('Save View')
        self.save_view_btn.clicked.connect(self.save_view)

        self.buttons_layout.addWidget(self.reset_model_btn)
        self.buttons_layout.addWidget(self.fit_index_label)
        self.buttons_layout.addWidget(self.fit_index_line)
        self.buttons_layout.addWidget(self.cell_index_label)
        self.buttons_layout.addWidget(self.cell_index_line)
        self.buttons_layout.addWidget(self.cursor_time_label)
        self.buttons_layout.addWidget(self.cursor_time_line)
        self.buttons_layout.addWidget(self.quickplot_btn)
        self.buttons_layout.addWidget(self.save_view_btn)

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

        self.export_path = expanduser("~")

    def scroll_all(self):
        '''Update xlims for all plots based on slider value.'''
        #self.start_time = self.time_slider.value()/self.slider_scaling
        self.start_time = self.time_slider.value()
        self.stop_time = self.start_time + self.display_duration
        self.display_start.setText(str(self.start_time))

        # don't go past the latest time of the biggest plot
        # (should all have the same max most of the time)
        #self._update_max_time()
        #if self.stop_time >= self.max_signal_time:
        #    self.stop_time = self.max_signal_time
        #    self.start_time = max(0, self.max_signal_time - self.display_duration)
        #    #self.time_slider.setValue(0)

        for ed in self.editors:
            [m.update_plot() for m in ed.modelspec_editor.modules]
            [s.update_plot() for s in ed.modelspec_editor.signal_displays]
            if len(ed.modelspec_editor.modules):
                ed.modelspec_editor.epochs.update_figure()

    def _update_max_time(self):
        resp = self.parent.rec.apply_mask()['resp']
        self.max_time = resp.as_continuous().shape[-1] / resp.fs
        self.max_signal_time = self.max_time
        self._update_range()
        #self.time_slider.setTickInterval(self.max_time/100)
        #self.time_slider.setRange(0, self.max_time-self.display_duration)
        #self.slider_scaling = self.max_time/(self.max_signal_time - self.display_duration)

    # def tap_right(self):
    #     self.time_slider.setValue(
    #             self.time_slider.value() + self.time_slider.singleStep
    #             )
    #
    # def tap_left(self):
    #     self.time_slider.setValue(
    #             self.time_slider.value() - self.time_slider.singleStep
    #             )

    def set_display_range(self):
        try:
            start = float(self.display_start.text())
        except:
            start = 0
        try:
            duration = float(self.display_range.text())
        except:
            duration = 10
        self.time_slider.setValue(start)
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

    def reset_model(self):
        self.parent.modelspec_editor.reset_model()

    def export_plot(self):
        time_range = (self.start_time, self.stop_time)
        fig = self.parent.modelspec.quickplot(time_range=time_range, modidx_set=None)
        w = MplWindow(fig=fig)
        w.show()

        path = join(self.export_path,self.parent.modelspec.meta['cellid']+
                    "_"+self.parent.modelspec.modelspecname+'.pdf')

        fname, mask = qw.QFileDialog.getSaveFileName(self, 'Save file', path, '*.pdf')
        log.info('saving quickplot to %s', fname)
        fig.savefig(fname)
        # keep path in memory for next save
        self.export_path = dirname(fname)

    def save_view(self, on_close=False):
        time_range = (self.start_time, self.stop_time)
        plot_fn_idxs = [mod.plot_fn_idx for mod in self.parent.modelspec_editor.modules]
        plot_channels = [mod.plot_channel for mod in self.parent.modelspec_editor.modules]
        collapsed = [sc.collapsed for sc in self.parent.modelspec_editor.collapsers]
        signal_idxs = [sc.signal_menu.currentIndex() for sc in self.parent.modelspec_editor.signal_controllers]
        signal_fn_idxs = [sc.plot_functions_menu.currentIndex() for sc in self.parent.modelspec_editor.signal_controllers]
        signal_channels = [int(sc.channel_entry.text()) for sc in self.parent.modelspec_editor.signal_controllers]
        signal_collapsed = [sc.collapsed for sc in self.parent.modelspec_editor.signal_collapsers]

        path = configfile
        config_group = 'Model_Browser_'+self.parent.ctx['meta']['modelname']

        if not on_close:
            dlg = view_config_location_dlg(path, config_group, parent=self)
            if dlg.exec():
                path, config_group = dlg.getInputs()
                print((path, config_group))
            else:
                print('Canceled')
                return

        config = ConfigParser()
        if not exists(path):
            log.warning(f"Config file {path} does not exist, creating")
        else:
            config.read(path)

        try:
            # Create non-existent section
            config.add_section(config_group)
        except:
            pass

        config.set(config_group, 'time_range', str(time_range))
        config.set(config_group, 'plot_fn_idxs', str(plot_fn_idxs))
        config.set(config_group, 'plot_channels', str(plot_channels))
        config.set(config_group, 'collapsed', str(collapsed))
        config.set(config_group, 'signal_idxs', str(signal_idxs))
        config.set(config_group, 'signal_fn_idxs', str(signal_fn_idxs))
        config.set(config_group, 'signal_channels', str(signal_channels))
        config.set(config_group, 'signal_collapsed', str(signal_collapsed))

        log.info(f'saving view parameters to {configfile}, section {config_group}')
        with open(configfile, 'w') as f:
            config.write(f)


    def update_fit_index(self):
        i = int(self.fit_index_line.text())

        for ed in self.editors:
            j = ed.modelspec_editor.modelspec.fit_index

            if i == j:
                return

            if i > len(ed.modelspec_editor.modelspec.raw):
                # TODO: Flash red or something to indicate error
                self.fit_index_line.setText(str(j))
                return

            ed.modelspec_editor.modelspec.fit_index = i
            ed.modelspec_editor.evaluate_model()

    def update_cell_index(self):
        try:
            i = int(self.cell_index_line.text())
            for ed in self.editors:
                for j, mc in enumerate(ed.modelspec_editor.controllers):
                    _plot_fn = mc.module.plot_list[mc.module.plot_fn_idx]
                    if _plot_fn.split(".")[-1] in ['pred_resp', 'strf_local_lin',
                                                   'perf_per_cell']:
                        log.info(f'{j}: {_plot_fn} updating')
                        mc.channel_entry.setText(str(i))
                    else:
                        log.info(f'{j}: {_plot_fn} not updating')
        except:
            log.info("Invalid cell index entry")
            """
            j = ed.modelspec_editor.modelspec.cell_index

            if i == j:
               return

            if i > len(ed.modelspec_editor.modelspec.phis):
               # TODO: Flash red or something to indicate error
               self.cell_index_line.setText(str(j))
               return

            ed.modelspec_editor.modelspec.cell_index = i
            ed.modelspec_editor.evaluate_model()
            """

    def update_cursor_time(self, t=None):
        if t is None:
            t = float(self.cursor_time_line.text())
        else:
            self.cursor_time_line.setText(str(t))

        # TODO : check for valid time

        if t == self.cursor_time:
            return

        self.cursor_time = t
        for ed in self.editors:
            [m.update_cursor() for m in ed.modelspec_editor.modules]

    def toggle_controls(self):
        if self.collapsed:
            show_layout(self.buttons_layout)
            show_layout(self.range_layout)
        else:
            hide_layout(self.buttons_layout)
            hide_layout(self.range_layout)
        self.collapsed = not self.collapsed

    def link_editor(self, editor):
        """
        link controls to an additional editor window so that scrolling and scaling are yoked
        """
        self.editors.append(editor)


class FitEditor(qw.QFrame):

    def __init__(self, parent):
        '''QWidget for manually initializing and fitting modelspec.'''
        super(qw.QFrame, self).__init__()
        self.parent = parent
        self.setFrameStyle(qw.QFrame.Panel | qw.QFrame.Raised)

        self.outer_layout = qw.QVBoxLayout()
        self.outer_layout.setAlignment(qc.Qt.AlignTop)

        self.init_layout = qw.QHBoxLayout()
        self.init_layout.setAlignment(qc.Qt.AlignTop)

        self.init_fn_menu = qw.QComboBox()
        self.init_fn_menu.addItems(_INIT_FNS)
        self.init_layout.addWidget(self.init_fn_menu)

        self.init_btn = qw.QPushButton('Initialize')
        self.init_btn.clicked.connect(self.initialize)
        self.init_layout.addWidget(self.init_btn)

        self.fit_layout = qw.QHBoxLayout()
        self.fit_layout.setAlignment(qc.Qt.AlignTop)

        self.fit_fn_menu = qw.QComboBox()
        self.fit_fn_menu.addItems(_FIT_FNS)
        self.fit_layout.addWidget(self.fit_fn_menu)

        self.fit_btn = qw.QPushButton('Fit')
        self.fit_btn.clicked.connect(self.fit)
        self.fit_layout.addWidget(self.fit_btn)

        self.iter_label = qw.QLabel('# iters')
        self.iter_edit = qw.QLineEdit('50')
        self.fit_layout.addWidget(self.iter_label)
        self.fit_layout.addWidget(self.iter_edit)

        self.outer_layout.addLayout(self.init_layout)
        self.outer_layout.addLayout(self.fit_layout)
        self.setLayout(self.outer_layout)

    def initialize(self):
        name = self.init_fn_menu.currentText()

        if 'from_keywords' in name:
            self.reset_from_keywords()
        else:
            fn = _lookup_fn_at(name)
            rec = self.parent.rec
            modelspec = self.parent.modelspec
            new_modelspec = fn(rec, modelspec)
            self.parent.set_new_modelspec(new_modelspec)

    def reset_from_keywords(self):
        fn = _lookup_fn_at('nems0.initializers.from_keywords')
        s = self.parent.modelspec.modelspecname
        registry = self.parent.ctx.get('registry', None)
        rec = self.parent.rec
        meta = self.parent.modelspec.meta
        new_modelspec = fn(s, registry=registry, rec=rec, meta=meta)
        self.parent.set_new_modelspec(new_modelspec)

    def fit(self):
        name = self.fit_fn_menu.currentText()
        fn = _lookup_fn_at(name)
        n_iters = int(self.iter_edit.text())
        fit_kwargs = {'max_iter': n_iters}
        rec = self.parent.ctx['est']
        modelspec = self.parent.modelspec

        new_modelspec = fn(rec, modelspec, fit_kwargs=fit_kwargs)
        self.parent.set_new_modelspec(new_modelspec)


def hide_layout(layout):
    widgets = (layout.itemAt(i).widget() for i in
               range(layout.count()))
    for w in widgets:
        w.hide()

def show_layout(layout):
    widgets = (layout.itemAt(i).widget() for i in
               range(layout.count()))
    for w in widgets:
        w.show()


# Just for testing - typically will be opened by recording_browser.py
# ctx and xfspec should be loaded into current console environment elsewhere
def run(modelspec, xfspec, rec, ctx):
    app = qw.QApplication(sys.argv)
    ex = EditorWindow(modelspec=modelspec, xfspec=xfspec, rec=rec, ctx=ctx)
    sys.exit(app.exec_())

def browse_xform_fit(ctx, xfspec, recname='val', control_widget=None):
    app = qw.QApplication.instance()
    modelspec = ctx['modelspec']
    rec = ctx[recname].apply_mask()
    ex = EditorWindow(modelspec=modelspec, xfspec=xfspec, rec=rec,
                      ctx=ctx, control_widget=control_widget)
    return ex

_DEBUG = False
if __name__ == '__main__':
    if _DEBUG:
        sys._excepthook = sys.excepthook
        def exception_hook(exctype, value, traceback):
            print(exctype, value, traceback)
            sys._excepthook(exctype, value, traceback)
            sys.exit(1)
        sys.excepthook = exception_hook

    if 'load' in sys.argv:
        batch = 271
        cellid = "TAR010c-18-1"
        modelname = 'dlog-wc.18x1.g-fir.1x15-lvl.1-dexp.1'
        d = nd.get_results_file(batch, modelnames=[modelname], cellids=[cellid])
        filename = d.loc[0,'modelpath']
        xfspec, ctx = xforms.load_analysis(filename)
        modelspec = ctx['modelspec']
        rec = ctx['val']
    else:
        # If modelspec or rec aren't defined, set to None and let
        # EditorWindow try to set them based on ctx
        try:
            temp1 = modelspec
        except NameError:
            modelspec = None
        try:
            temp2 = rec
        except NameError:
            rec = None

    run(modelspec, xfspec, rec, ctx)

class view_config_location_dlg(qw.QDialog):
    def __init__(self, path='', config_group='', parent = None):
        qw.QWidget.__init__(self, parent=parent)
        # super(view_config_location_dlg, self).__init__(parent)

        formLayout = qw.QFormLayout(self)
        pathlabel = qw.QLabel(self)
        pathlabel.setText('Config file:')
        pathlabel.setMaximumWidth(200)
        self.pathLE = qw.QLineEdit(self)
        self.pathLE.setText(str(path))
        self.pathLE.setMaximumWidth(1200)
        formLayout.addRow(pathlabel, self.pathLE)

        config_grouplabel = qw.QLabel(self)
        config_grouplabel.setText('config group:')
        config_grouplabel.setMaximumWidth(200)
        self.config_groupLE = qw.QLineEdit(self)
        self.config_groupLE.setText(str(config_group))
        self.config_groupLE.setMaximumWidth(1200)
        formLayout.addRow(config_grouplabel, self.config_groupLE)

        buttonBox = qw.QDialogButtonBox(qw.QDialogButtonBox.Ok | qw.QDialogButtonBox.Cancel, self)
        formLayout.addWidget(buttonBox)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        self.resize(1200, 100)
        self.show()
    def getInputs(self):
        path = self.pathLE.text()
        config_group = self.config_groupLE.text()
        return path, config_group