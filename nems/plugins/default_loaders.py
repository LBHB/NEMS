from nems0.registry import xform, xmodule

@xform()
def ld(loadkey, recording_uri=None, recording_uri_list=None, cellid=None):
    '''
    Default loader xfspec. Loads the recording and (optionally) specifies
    input_name and output_name context variables that tell, respectively, the
    name of the input signal fed into the model and output signal to be
    predicted by the model.

    options:
    n : normalize input stim
    revrec : set input_name = 'resp', output_name = 'stim' (ie, reversed from
             default input_name = 'stim, output_name = 'resp')
    popXXX : all cells that don't match cellid should be saved to signal XXX
             default XXX='population'
    p : set input_name = 'psth'

    '''

    options = loadkey.split('.')[1:]

    d = {}

    # TODO: really should only have one argument or the other, but currently some code uses the first
    # version while other code uses the second version. So until that gets resolved, need to accept
    # both versions as kwargs, otherwise the information will not be passed from the registry.
    if recording_uri is not None:
        d['recording_uri_list'] = [recording_uri]
    elif recording_uri_list is not None:
        d['recording_uri_list'] = recording_uri_list

    d['normalize'] = False
    #if cellid is not None:
    #    d['cellid'] = cellid

    for op in options:
        if op.startswith('pop'):
            if len(op) == 3:
                d['save_other_cells_to_state'] = 'population'
            else:
                d['save_other_cells_to_state'] = op[3:]
        elif op == 'revrec':
            d['input_name'] = 'resp'
            d['output_name'] = 'stim'
        elif op == 'p':
            d['input_name'] = 'psth'
        elif op == 'k':
            d['save_other_cells_to_state'] = 'keep'
        elif op == 'n':
            d['normalize'] = True

    xfspec = [['nems0.xforms.load_recordings', d]]

    return xfspec

@xform()
def none(loadkey):
    '''
    Does nothing, but xforms expects at least one loader/preprocessor.
    '''
    return [['nems0.xforms.none', {}]]
