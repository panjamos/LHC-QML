import IPython as ipy
import LHC_QML_module as lqm
from sklearn.model_selection import train_test_split 
import time
import matplotlib.pyplot as plt
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.circuit.library import RealAmplitudes, EfficientSU2
from qiskit.algorithms.optimizers import COBYLA, SLSQP, SPSA

signal_datapath = "D:\LHC-QML\data\output_VBF_HToZZTo4L_M-125_8TeV-powheg-pythia6.root:HZZ4LeptonsAnalysisReduced"
background_datapath = "D:\LHC-QML\data\output_GluGluToZZTo4L_8TeV-gg2zz-pythia6.root:HZZ4LeptonsAnalysisReduced"

load_path = "D:/LHC-QML/models/fit_result"

choice_feature_keys = [
 'f_lept1_pt', 'f_lept1_eta', 'f_lept1_phi', 'f_lept1_pfx', 'f_lept2_pt',
 'f_lept2_eta', 'f_lept2_phi', 'f_lept2_pfx', 'f_lept3_pt', 'f_lept3_eta',
 'f_lept3_phi', 'f_lept4_pt', 'f_lept4_eta', 'f_lept4_phi', 'f_Z1mass',
 'f_Z2mass', 'f_angle_costhetastar', 'f_angle_costheta1', 'f_angle_costheta2', 'f_angle_phi',
 'f_angle_phistar1', 'f_pt4l', 'f_eta4l', 'f_mass4l', 'f_deltajj',
 'f_massjj', 'f_jet1_pt', 'f_jet1_eta', 'f_jet1_phi', 'f_jet1_e',
 'f_jet2_pt', 'f_jet2_eta', 'f_jet2_phi', 'f_jet2_e']

seed = 123
n_training_points = 500
#decides if callback makes a graph at each step

training_feature_indices = [15,21,22,23] #corresponds to f_Z2mass, f_pt4l, f_eta4l, f_mass4l
#training_key_indices = range(len(choice_feature_keys))
num_features = len(training_feature_indices)

#loads data from files
signal, background = lqm.load_data(signal_datapath, background_datapath, choice_feature_keys)

#formats data for input into vqc
features, labels = lqm.format_data(signal, background, choice_feature_keys)

#scales features to be between 0 and 1
features, labels, scaler = lqm.preprocess_data(features, labels)

n_signal_events = (labels == 1).sum()

#splits data into testing and training sets
#data is first cut to inlcude equal number of signal and background events
#TODO: maybe split signal and backgrounds seperately to ensure equal number of signal/background in each test/training set and then combine and randomize order
train_features, test_features, train_labels, test_labels = train_test_split(
    features[:2*n_signal_events], labels[:2*n_signal_events], train_size=0.8, random_state=seed)


#compare_keys = ['f_Z1mass','f_pt4l', 'f_eta4l', 'f_mass4l']
#lqm.plot_pairwise(compare_keys, signal, background)

vqc = VQC.load(load_path)
lqm.score_model(vqc, train_features, test_features, train_labels, test_labels, training_feature_indices)

prediction = vqc.predict(test_features[:,training_feature_indices])
prob = vqc._neural_network.forward(test_features[:,training_feature_indices],vqc._fit_result.x)

lqm.plot_discriminator(prediction, test_labels)
lqm.plot_roc(prediction, test_labels)

lqm.plot_discriminator(prob[:,1], test_labels)
lqm.plot_roc(prob[:,1], test_labels)