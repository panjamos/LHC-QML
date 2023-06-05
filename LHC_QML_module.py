import numpy as np
import uproot
import qiskit
import sklearn
import os
import sys

def load_data(signal_datapath, background_datapath):
    #file with HToZZ in name is signal
    signal = uproot.open(signal_datapath)
    background = uproot.open(background_datapath)
    return signal, background



def format_data(signal, background, feature_keys):
    #changes the format of the data to be more like the old code

    number_signal_events = len(signal['f_Z1mass'].array(library="np"))
    number_background_events = len(background['f_Z1mass'].array(library="np"))

    print('# of signal events: ' + str(number_signal_events))
    print('# of background events: ' + str(number_background_events))

    events_and_labels = [np.append(np.ones(number_signal_events),np.zeros(number_background_events))] 
    #creates data array with n_signal 1s and n_background 0s
    #will be appended with event data

    #puts each feature array in the data array along the first axis
    #all signal events followed by all background events
    for key in feature_keys:
        event_data = np.append(signal[key].array(library='np'), background[key].array(library='np'))
        events_and_labels = np.append(events_and_labels, [event_data], axis=0)
    #NOTE: rewrite later --> It's uneccessary to put everything in one array just to split it back up before we return

    #transpose array to match format of previous code
    events_and_labels = np.transpose(events_and_labels)

    #create the input and targets arrays in format for training model
    #still in order of signal events followed by background events
    features = events_and_labels[:,1:]
    labels = events_and_labels[:,0]
    
    return features, labels



def preprocess_data(features, labels):
    #currently the only preprocessing is scaling each feature independently to be between 0 and 1

    scaler = sklearn.preprocessing.MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    #return scaler object for unscaling later if necessary
    return scaled_features, labels, scaler



def save_model(fit_result, save_folder):
    #improve to also save file with vqc parameters, i.e. steps, points used, seed, split number, feature map, optimizer, ansatz, etc
    
    fit_number = 0
    fit_filepath_default = save_folder + "/fit_result"
    fit_filepath = fit_filepath_default

    while os.path.exists(fit_filepath + '.npy'):
        fit_filepath = fit_filepath_default + str(fit_number)
        fit_number += 1
        if fit_number == 100:
            sys.exit("filepath likely incorrect") 
    
    np.save(fit_filepath, fit_result.x)
    print('file saved to ' + fit_filepath + '.npy')



def score_model(vqc, train_features, test_features, train_labels, test_labels, feature_indices):
    train_score_loaded = vqc.score(train_features[:500,feature_indices], train_labels[:500])
    test_score_loaded = vqc.score(test_features[:500,feature_indices], test_labels[:500])

    print(f"Quantum VQC on the training dataset: {train_score_loaded:.5f}")
    print(f"Quantum VQC on the test dataset:     {test_score_loaded:.5f}")