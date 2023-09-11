"""Shows how to use the fit debugger to get a detailed report of fit steps."""

import matplotlib.pyplot as plt

from nems import Model, load_demo
from nems.tools.debug import debug_fitter, animated_debug
from nems.visualization.fitter import parameter_pca_table


train, test = load_demo(tutorial_subset=True)  #First 5 sounds only
input = train['spectrogram']  
target = train['response']


# Creating a basic model from keywords and sample_from_priors
model = Model.from_keywords('wc.18x3-fir.15x3-dexp.1').sample_from_priors()


###########################
# Static Debug Plotting
#   iterations: Set the amount of iterations for fitting our model
#   sampling_rate: Labeling our time-axis to real units
#   xlim: Positional arg for ax.set_xlim
#   !See debug_fitter documentation for many other parameters 
###########################
fig, models = debug_fitter(
    model, input, target, iterations=10, sampling_rate=100,
    xlim=(0,10),  # specify in seconds since `sampling_rate` is given
    ## Uncomment and provide a save location if you don't want to re-fit the
    ## models in step 4.
    # save_path="/path/to/directory"
    )


###########################
# PCA's and PCA table
# parameter_pc_table: Generates table for first 5 weights of n parameter PC's
#   n: The first n parameter PC's to show
#   fontsize: Fontsize for the figure and graph being plotted
#
#    Parameter PCs determined by treating each entry in the length-N
#    vector returned by `model.get_parameter_vector()` as a feature, and the
#    vector associated with each of the M models in `models` as one
#    observation, then computing principal components for the resulting
#    M x N matrix.
###########################
fig, ax = plt.subplots(figsize=(8,3))
parameter_pca_table(models, n=5, fontsize=16)


###########################
# Animated debug plots
# animated_debug:
#   animation_save_path: Path to save animated debug
#   frame_interval: Delay between frames in milliseconds 
#   Otherwise same parameters available as debug_fitter
#
# Updates the plot at each iteration via animations
#
# NOTE: matplotlib aimations are dependent on backend, IDE, OS, etc...
# If animation is not playing, try saving a filepath for the animation
###########################
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
