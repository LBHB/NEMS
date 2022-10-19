"""Shows how to use the fit debugger to get a detailed report of fit steps."""

import matplotlib.pyplot as plt

from nems import Model, load_demo
from nems.tools.debug import debug_fitter, animated_debug
from nems.visualization.fitter import parameter_pca_table

# NOTE: Requires first downloading demo data using `nems.download_demo`.
#       Alternatively, modify the first few lines to use a different
#       input and target.

# 1. Load data.
#    The NEMS demo data contains a single concatenated natural sound spectrogram
#    and firing rate response for one neuron from ferret primary auditory cortex,
#    both represented at 100 Hz.
train, test = load_demo(tutorial_subset=True)  # First 5 sounds only
input = train['spectrogram']  
target = train['response']

# 2. Specify model.
#    Here we'll use a rank-3 LN model and start from random initial conditions.
model = Model.from_keywords('wc.18x3-fir.15x3-dexp.1').sample_from_priors()


# 3a. Static debug plot.
#    Fit the model for 10 iterations using the scipy backend, and report error,
#    gradient, and parameter information for each iteration, along with the
#    first 10 seconds of the spectrogram input and final model prediction.
#
#    Parameter PCs determined by treating each entry in the length-N
#    vector returned by `model.get_parameter_vector()` as a feature, and the
#    vector associated with each of the M models in `models` as one
#    observation, then computing principal components for the resulting
#    M x N matrix.
fig, models = debug_fitter(
    model, input, target, iterations=10, sampling_rate=100,
    xlim=(0,10),  # specify in seconds since `sampling_rate` is given
    ## Uncomment and provide a save location if you don't want to re-fit the
    ## models in step 4.
    # save_path="/path/to/directory"
    )

# 3b. PCA table.
#    Show information for the first 5 parameter PCs, including which Parameters
#    are associated with the largest weights from each PC.
fig, ax = plt.subplots(figsize=(8,3))
parameter_pca_table(models, n=5, fontsize=16)


# 4. Animated debug plot.
#    As 3a, but update the plot one iteration at a time. Most information is
#    already present in the static plot, but the animation also shows the
#    intermedite model prediction associated with each fit iteration.
#    NOTE: matplotlib animations are very dependent on backend, IDE, OS, etc.
#          For example, they tend not to work with jupyter notebooks. If the
#          animation doesn't play, try saving it as a movie instead by
#          specifying `animation_save_path`.
animation, models = animated_debug(
    model, input, target, iterations=10, sampling_rate=100,
    xlim=(0,10),
    ## Uncomment and provide the save location from step 3 if applicable.
    # load_path="/path/to/directory",
    ## Uncomment and provide a filepath where the animation should be saved,
    ## if desired.
    ## NOTE: You may need to install the FFmpeg encoder, for example using
    ##       `conda install -c conda-forge ffmpeg`.
    # animation_save_path="/path/to/movie.mp4",
    )
