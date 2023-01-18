Quick Start to Modeling with NEMS
=================================

Build a standard linear-nonlinear spectrotemporal receptive field (LN-STRF)
model.
::

    from nems import Model
    from nems.layers import FiniteImpulseResponse, DoubleExponential

    model = Model()
    model.add_layers(
        FiniteImpulseResponse(shape=(15, 18)),  # 15 taps, 18 spectral channels
        DoubleExponential(shape=(1,))           # static nonlinearity, 1 output
    )


Or use the customizable keyword system for faster scripting and prototyping.
::

    from nems import Model

    same_model = Model.from_keywords('fir.15x18-dexp.1')


Fit the model to (fake) evoked neural activity (in this case, in response to a
sound represented by a spectrogram).
::

    import numpy as np

    spectrogram = np.random.rand(1000, 18)  # 1000 time bins, 18 channels
    response = np.random.rand(1000, 1)      # 1 neural response

    fitted_model = model.fit(spectrogram, response)


Predict the response to a different stimulus.
::

    test_spectrogram = np.random.rand(1000, 18)
    prediction = fitted_model.predict(test_spectrogram)


Score the prediction
::

    from nems.metrics import correlation
    print(correlation(prediction, response))
    # OR
    print(model.score(test_spectrogram, response, metric='correlation'))


Try the above examples with real data:
::

    import nems

    nems.download_demo()
    training_dict, test_dict = nems.load_demo()

    # Each dictionary contains a 100 hz natural sound spectrogram and
    # the PSTH / firing rate of the recorded spiking response.
    spectrogram = training_dict['spectrogram']
    response = training_dict['response']

