# TODO -- break off into a separate state-dependent model example that actually works

def my_data_loader2(file_path=None):
    # Dummy function to demonstrate the data format.
    print(f'Loading data from {file_path}, but not really...')
    spectrogram = np.random.random(size=(1000, 18))
    response = spectrogram[:,[1]] - spectrogram[:,[7]]*0.5 + np.random.randn(1000, 1)*0.1 + 0.5

    state = np.ones((len(response), 2))
    state[:500,:] = 0
    response = response * (1+state[:,1])

    return spectrogram, response, state


spectrogram, response = my_data_loader('path/to_data.csv')
print(f'Our original dataset size is {spectrogram.shape}')
model = Model()
model.add_layers(
    WeightChannels(shape=(18, 1)),  # Input size of 18, Output size of 1
    LevelShift(shape=(1,)) ,# WeightChannels will provide 1 input to shift
)
fitter_options = {'options': {'maxiter': 100, 'ftol': 1e-5}}
fitter_options = {'options': {'maxiter': 1000, 'tolerance': 1e-5}}

model_fit = model.fit(spectrogram, target=response,
                      fitter_options=fitter_options)




spectrogram, response, state = my_data_loader2('path/to_data.csv')
from nems.layers import LevelShift, WeightChannels, StateGain
print(f'Our original dataset size is {spectrogram.shape}')
model = Model()
model.add_layers(
    WeightChannels(shape=(18, 1)),  # Input size of 18, Output size of 1
    StateGain(shape=(2,1))
)
fitter_options = {'options': {'maxiter': 100, 'ftol': 1e-5}}
fitter_options = {'options': {'maxiter': 1000, 'tolerance': 1e-5}}

model_fit = model.fit(spectrogram, target=response, state=state,
                      fitter_options=fitter_options)
