import numpy as np
import uproot
from qiskit_machine_learning.algorithms.classifiers import VQC
import sklearn
import os
import sys
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc


def load_data(signal_path, background_path, keys, folder_paths=True):
    #loads all root files in folders given if folder_paths == True
    #loads all root files with paths given in signal_path list and background_path list if folder_paths == False
    if folder_paths:
        print("\nsignal data from:")
        signal_filepaths = print_get_root_filepaths(signal_path)
        print("\nbackground data from:")
        background_filepaths = print_get_root_filepaths(background_path)
        files_used = [background_filepaths, signal_filepaths]
    else:
        if not isinstance(signal_path,list):
            signal_path = [signal_path]
        if not isinstance(background_path,list):
            background_path = [background_path]
        
        signal_filepaths = signal_path
        background_filepaths = background_path

        print("\nsignal data from:")
        for path in signal_filepaths: print(path)
        print("\nbackground data from:")
        for path in background_filepaths: print(path)
        files_used = [background_filepaths, signal_filepaths]

    signal_datapaths = []
    background_datapaths = []

    for filepath in signal_filepaths:
        signal_datapaths.append(filepath + ":HZZ4LeptonsAnalysisReduced")
    for filepath in background_filepaths:
        background_datapaths.append(filepath + ":HZZ4LeptonsAnalysisReduced")
    
    signal_dict = uproot.concatenate(signal_datapaths, keys, library='np')
    background_dict = uproot.concatenate(background_datapaths, keys, library='np')

    print('\ndata loaded')
    return signal_dict, background_dict, files_used



def format_data(signal, background):
    #changes the format of the data to be more like the old code
    #signal, background dicts --> feature and label arrays
    keys = list(signal.keys())

    number_signal_events = len(signal[keys[0]])
    number_background_events = len(background[keys[0]])

    print('\n# of signal events: ' + str(number_signal_events))
    print('# of background events: ' + str(number_background_events) + '\n')

    labels = np.append(np.ones(number_signal_events),np.zeros(number_background_events))
    #creates data array with n_signal 1s and n_background 0s

    features = [np.concatenate((signal[keys[0]], background[keys[0]]))]
    #initialize feature array with first feature
    
    if len(keys) != 1:
        for key in keys[1:]:
            #put each following feature in array with the first index corresponding to feature
            one_feature = np.concatenate((signal[key], background[key]))
            features = np.append(features, [one_feature], axis=0)

    #transpose array for input to vqc
    features = np.transpose(features)

    #create the input and targets arrays in format for training model
    #still in order of signal events followed by background events
    print('data formatted')
    return features, labels



def preprocess_data(features, labels):
    #currently the only preprocessing is scaling each feature independently to be between 0 and 1

    scaler = sklearn.preprocessing.MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    #return scaler object for unscaling later if necessary
    print('data preprocessed\n')
    return scaled_features, labels, scaler



def save_model(vqc: VQC, save_folder, seed='not specified', n_training_points='not specified', training_feature_keys='not specified', files_used='not specified'):
    #improve to also save file with vqc parameters, i.e. steps, points used, seed, split number, feature map, optimizer, ansatz, etc
    
    fit_number = 0
    fit_filepath_default = save_folder + "/trained_vqc"
    fit_filepath = fit_filepath_default

    while os.path.exists(fit_filepath):
        fit_filepath = fit_filepath_default + str(fit_number)
        fit_number += 1
        if fit_number == 500:
            sys.exit("filepath likely incorrect") 
    
    vqc.save(fit_filepath)

    info_file = open(fit_filepath + '.txt', 'a', encoding="utf-8")
    info_file.write('feature map = \n' + str(vqc.feature_map) +
                     '\nansatz = \n' + str(vqc.ansatz) +
                     '\nloss function = ' + str(vqc.loss) +
                     '\noptimizer = ' + str(vqc.optimizer) +
                     '\niterations = ' + str(vqc.optimizer._options['maxiter']) +
                     '\nseed = ' + str(seed) +
                     '\nnumber of training points = ' + str(n_training_points) +
                     '\nfeatures used in training = ' + str(training_feature_keys))
    
    info_file.write('\n\nsignal files used\n')
    for filepath in files_used[1]:
        info_file.write(filepath + '\n')
    info_file.write('\n\nbackground files used\n')
    for filepath in files_used[0]:
        info_file.write(filepath + '\n')
    info_file.close()

    print('\nvqc file saved to ' + fit_filepath)
    print('\ninfo file saved to ' + fit_filepath + '.txt')



def score_model(vqc: VQC, train_features, test_features, train_labels, test_labels):
    train_score_loaded = vqc.score(train_features[:500,:], train_labels[:500])
    test_score_loaded = vqc.score(test_features[:500,:], test_labels[:500])

    print("Warning: only scoring on 500 pts")
    print(f"Quantum VQC on the training dataset: {train_score_loaded:.5f}")
    print(f"Quantum VQC on the test dataset:     {test_score_loaded:.5f}")



def plot_pairwise(signal, background):
    #expects data in dicts with keys being the features and value being 1d array of data for events
    feature_dict = {}
    for key in signal.keys():
        feature_dict[key] = np.concatenate((background[key],signal[key]))

    df = pd.DataFrame(feature_dict)

    event_labels = np.concatenate((np.full(len(background[key]),"background"),np.full(len(signal[key]),"signal")))

    df["Event Type"] = pd.Series(event_labels)

    plot = sns.pairplot(df, hue="Event Type", corner=True, palette = {'signal' : 'r', 'background' : 'b'},
                        markers=["X", "."], diag_kws = dict(common_norm=False), plot_kws = dict(linewidth=0.2,alpha=0.75))
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

def print_get_root_filepaths(directory_path):
    directory = os.fsencode(directory_path)
    filepaths = []

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith('.root'):
            filepath = os.fsdecode(os.path.normpath(os.path.join(directory, file)))
            print(filepath)
            filepaths.append(filepath)
    return filepaths