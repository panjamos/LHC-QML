import numpy as np
import uproot
import qiskit
import sklearn
import os
import sys
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc

def load_data(signal_datapath, background_datapath, keys):
    #file with HToZZ in name is signal
    signal_tree = uproot.open(signal_datapath)
    background_tree = uproot.open(background_datapath)
    signal_dict = {}
    background_dict = {}

    for key in keys:
        signal_dict[key] = signal_tree[key].array(library="np")
        background_dict[key] = background_tree[key].array(library="np")
    print('data loaded')
    return signal_dict, background_dict



def format_data(signal, background, feature_keys):
    #changes the format of the data to be more like the old code

    number_signal_events = len(signal['f_Z1mass'])
    number_background_events = len(background['f_Z1mass'])

    print('# of signal events: ' + str(number_signal_events))
    print('# of background events: ' + str(number_background_events))

    events_and_labels = [np.append(np.ones(number_signal_events),np.zeros(number_background_events))] 
    #creates data array with n_signal 1s and n_background 0s
    #will be appended with event data

    #puts each feature array in the data array along the first axis
    #all signal events followed by all background events
    for key in feature_keys:
        event_data = np.concatenate((signal[key], background[key]))
        events_and_labels = np.append(events_and_labels, [event_data], axis=0)
    #NOTE: rewrite later --> It's uneccessary to put everything in one array just to split it back up before we return

    #transpose array to match format of previous code
    events_and_labels = np.transpose(events_and_labels)

    #create the input and targets arrays in format for training model
    #still in order of signal events followed by background events
    features = events_and_labels[:,1:]
    labels = events_and_labels[:,0]
    print('data formatted')
    return features, labels



def preprocess_data(features, labels):
    #currently the only preprocessing is scaling each feature independently to be between 0 and 1

    scaler = sklearn.preprocessing.MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    #return scaler object for unscaling later if necessary
    print('data preprocessed')
    return scaled_features, labels, scaler



def save_model(vqc, save_folder):
    #improve to also save file with vqc parameters, i.e. steps, points used, seed, split number, feature map, optimizer, ansatz, etc
    
    fit_number = 0
    fit_filepath_default = save_folder + "/fit_result"
    fit_filepath = fit_filepath_default

    while os.path.exists(fit_filepath):
        fit_filepath = fit_filepath_default + str(fit_number)
        fit_number += 1
        if fit_number == 500:
            sys.exit("filepath likely incorrect") 
    
    vqc.save(fit_filepath)
    print('file saved to ' + fit_filepath)



def score_model(vqc, train_features, test_features, train_labels, test_labels, feature_indices):
    train_score_loaded = vqc.score(train_features[:500,feature_indices], train_labels[:500])
    test_score_loaded = vqc.score(test_features[:500,feature_indices], test_labels[:500])

    print("Warning: only scoring on 500 pts")
    print(f"Quantum VQC on the training dataset: {train_score_loaded:.5f}")
    print(f"Quantum VQC on the test dataset:     {test_score_loaded:.5f}")



def plot_pairwise(compare_keys, signal, background):
    #expects data in dicts with keys being the features and value being 1d array of data for events
    #NOTE: something might be wrong with this code, plots look different from old version
    feature_dict = {}
    for key in compare_keys:
        feature_dict[key] = np.concatenate((background[key],signal[key]))

    df = pd.DataFrame(feature_dict)

    event_labels = np.concatenate((np.full(len(background[key]),"background"),np.full(len(signal[key]),"signal")))

    df["Event Type"] = pd.Series(event_labels)

    plot = sns.pairplot(df, hue="Event Type", corner=True, palette = {'signal' : 'r', 'background' : 'b'},markers=["X", "."], diag_kws = dict(common_norm=False), plot_kws = dict(linewidth=0.2,alpha=0.75))
    plot.fig.suptitle("Feature Comparison Plots")
    plt.show()

def plot_loss(losses):
    plt.title("Loss During training")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.plot(range(1,len(losses)+1), losses)
    plt.show()

def plot_discriminator(prediction, target):
    labels = ['Background', 'Signal']
    colors = ['b', 'r']
    plt.figure(1) 
    #I split up signal and background using masks instead of loops
    #masking test values where prediction is indicated signal/background
    signal = np.bool_(target.flat)

    n, bins, patches = plt.hist(prediction[signal],50, histtype='step', color=colors[1], label=labels[1])#,density=True) 
    n, bins, patches = plt.hist(prediction[~signal],50, histtype='step', color=colors[0], label=labels[0])#,density=True) 
    plt.title('Discriminator')
    plt.ylabel("Counts")
    plt.legend()
    plt.xlabel("Output Value")
    plt.legend()
    plt.show()

def plot_roc(prediction, labels):
    fpr, tpr, _ = roc_curve(labels, prediction)

    auc_roc = auc(fpr, tpr)

    plt.figure(2,figsize=(6,6))
    plt.plot(tpr, 1.0-fpr, lw=3, alpha=0.8,
            label="(AUC={:.3f})".format(auc_roc))
    plt.xlabel("Signal efficiency")
    plt.ylabel("Background rejection")
    plt.legend(loc=3)
    plt.xlim((0.0, 1.0))
    plt.ylim((0.0, 1.0))
    plt.show()
    plt.close()