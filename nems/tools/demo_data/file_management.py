import requests
import pathlib

import numpy as np

from nems.tools.arrays import apply_to_dict


# E.g. "/path/to/NEMS/nems/tools/demo_data/saved_data/"
basepath = pathlib.Path(__file__).parent / 'saved_data'
demo_files = ['TAR010c-18-2.npz', "TAR010c_data.npz"]

def download_data(url, filepath):
    r = requests.get(url, stream=True)
    with open(filepath, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)


# TODO: Other download options, or just the one file?
def download_demo(overwrite=False):
    
    prefix = 'https://s3-us-west-2.amazonaws.com/nemspublic/sample_data/'
    for filebase in demo_files:
        url = prefix + filebase
        filepath = basepath / filebase

        if filepath.is_file() and not overwrite:
            print(
                "Demo data already exists at filepath:\n"
                f"{filepath}\n"
                "To overwrite data, use `download_demo(overwrite=True)`."
            )
            
        else:
            print(f"Downloading NEMS demo data from:\n{url}\n...\n")
            download_data(url, filepath)
            print(
                "NEMS demo data has been successfully downloaded to:\n"
                f"{filepath}"
            )

    print(
        "\n\n"
        "Data can be loaded using:\n"
        "`training_dict, test_dict = nems.load_demo()`\n"
        "and indexed with:\n"
        "`training_dict['spectrogram']  # 100hz spectrogram\n"
        "`training_dict['response']     # 100hz PSTH / firing rate"
    )


# TODO: other options?
# TODO: batched version?
def load_demo(filebase=None, tutorial_subset=False):
    """
    
    :param filebase: {str, None}
        'TAR010c-18-2.npz': load data for a single A1 neuron presented with natural sound stimuli
        'TAR010c_data.npz': load data for 55 simultaneously recorded A1 neurons
        None defaults to 'TAR010c-18-2.npz'
    :param tutorial_subset: {bool}
        if True, truncates the data in time to speed up example model fitting
    :return training_dict, test_dict each a dictionary with spectrogram and response matrices
    """
    
    if filebase is None:
        filebase = demo_files[0]
        
    filepath = basepath / filebase
    if not filepath.is_file():
        raise ValueError(
            "NEMS demo data does not exist at path:\n"
            f"{filepath}\n"
            "First use `nems.download_demo()`, then try `load_demo()` again."
        )
    else:
        # TODO: just save the file this way instead?
        #       possibly rename it as well, to something like:
        #       "a1_100hz_linear.npz"
        with np.load(filepath) as data_dict:        
            training_dict = {
                'spectrogram': data_dict['X_est'],
                'response': data_dict['Y_est'],
                'cellid': data_dict.get('cellid','TAR010c-18-2')
            }
            test_dict = {
                'spectrogram': data_dict['X_val'],
                'response': data_dict['Y_val'],
                'cellid': data_dict.get('cellid', 'TAR010c-18-2')
            }

    if tutorial_subset:
        # TODO: 2900 is hard-coded for TAR010c-18-2, maybe make this its own
        #       file instead?
        select_subset = lambda a: a[:2900]  # first 5 stimuli
        training_dict = apply_to_dict(select_subset, training_dict)
        test_dict = apply_to_dict(select_subset, training_dict)

    # TODO: how to report meta information? maybe just print here?

    return training_dict, test_dict
