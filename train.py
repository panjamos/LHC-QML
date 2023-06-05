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
save_folder = "D:/LHC-QML/fit results"

choice_feature_keys = [
 'f_lept1_pt', 'f_lept1_eta', 'f_lept1_phi', 'f_lept1_pfx', 'f_lept2_pt',
 'f_lept2_eta', 'f_lept2_phi', 'f_lept2_pfx', 'f_lept3_pt', 'f_lept3_eta',
 'f_lept3_phi', 'f_lept4_pt', 'f_lept4_eta', 'f_lept4_phi', 'f_Z1mass',
 'f_Z2mass', 'f_angle_costhetastar', 'f_angle_costheta1', 'f_angle_costheta2', 'f_angle_phi',
 'f_angle_phistar1', 'f_pt4l', 'f_eta4l', 'f_mass4l', 'f_deltajj',
 'f_massjj', 'f_jet1_pt', 'f_jet1_eta', 'f_jet1_phi', 'f_jet1_e',
 'f_jet2_pt', 'f_jet2_eta', 'f_jet2_phi', 'f_jet2_e']

seed = 123
n_training_points = 100
#graph = False
#decides if callback makes a graph at each step

training_feature_indices = [15,21,22,23] #corresponds to f_Z2mass, f_pt4l, f_eta4l, f_mass4l
#training_key_indices = range(len(choice_feature_keys))
num_features = len(training_feature_indices)

feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
sampler = Sampler()
ansatz = EfficientSU2(num_qubits=num_features, reps=3)
optimizer = COBYLA(maxiter=20)



#loads data from files
signal_dict, background_dict = lqm.load_data(signal_datapath, background_datapath, choice_feature_keys)

#formats data for input into vqc
features, labels = lqm.format_data(signal_dict, background_dict, choice_feature_keys)

#scales features to be between 0 and 1
features, labels, scaler = lqm.preprocess_data(features, labels)

n_signal_events = (labels == 1).sum()

#splits data into testing and training sets
#data is first cut to inlcude equal number of signal and background events
#TODO: maybe split signal and backgrounds seperately to ensure equal number of signal/background in each test/training set and then combine and randomize order
train_features, test_features, train_labels, test_labels = train_test_split(
    features[:2*n_signal_events], labels[:2*n_signal_events], train_size=0.8, random_state=seed)



losses = []
times = []

def callback(weights, step_loss):
#function called at each step of vqc training
#expected to only have these two inputs and return None
    losses.append(step_loss)
    times.append(time.time())
    print(f"Iteration {len(losses)} Training time: {round(times[-1]-times[-2],2)} seconds")
    print(f"Loss: {step_loss:.4f}")
    #if graph:
    #    ipy.display.clear_output(wait=True)
    #    plt.title("Objective function value against iteration")
    #    plt.xlabel("Iteration")
    #    plt.ylabel("Objective function value")
    #    plt.plot(range(1,len(losses)+1), losses)
    #    plt.show()

vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback,
)

start = time.time()
times.append(start)
vqc.fit(train_features[:n_training_points,training_feature_indices], train_labels[:n_training_points]) #training the vqc
elapsed = time.time() - start

print(f"Training time: {round(elapsed)} seconds")

fit_result = vqc._fit_result

lqm.save_model(fit_result, save_folder)
lqm.score_model(vqc, train_features, test_features, train_labels, test_labels, training_feature_indices)

lqm.plot_loss(losses)