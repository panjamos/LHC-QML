import LHC_QML_module as lqm
from sklearn.model_selection import train_test_split 
import time
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.circuit.library import RealAmplitudes, EfficientSU2
from qiskit.algorithms.optimizers import COBYLA, SLSQP, SPSA

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

signals_folder = "./data/signal"
backgrounds_folder = "./data/background"
save_folder = "./models"

#choice_feature_keys = [
# 'f_lept1_pt', 'f_lept1_eta', 'f_lept1_phi', 'f_lept1_pfx', 'f_lept2_pt',
# 'f_lept2_eta', 'f_lept2_phi', 'f_lept2_pfx', 'f_lept3_pt', 'f_lept3_eta',
# 'f_lept3_phi', 'f_lept4_pt', 'f_lept4_eta', 'f_lept4_phi', 'f_Z1mass',
# 'f_Z2mass', 'f_angle_costhetastar', 'f_angle_costheta1', 'f_angle_costheta2', 'f_angle_phi',
# 'f_angle_phistar1', 'f_pt4l', 'f_eta4l', 'f_mass4l', 'f_deltajj',
# 'f_massjj', 'f_jet1_pt', 'f_jet1_eta', 'f_jet1_phi', 'f_jet1_e',
# 'f_jet2_pt', 'f_jet2_eta', 'f_jet2_phi', 'f_jet2_e']

seed = 123
n_training_points = 10

training_feature_keys = ['f_Z2mass','f_pt4l', 'f_eta4l', 'f_mass4l']
num_features = len(training_feature_keys)

feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
sampler = Sampler()
ansatz = EfficientSU2(num_qubits=num_features, reps=3)
optimizer = COBYLA(maxiter=10)



#loads data from files
signal_dict, background_dict, files_used = lqm.load_data(signals_folder, backgrounds_folder, training_feature_keys)

#formats data for input into vqc
features, labels = lqm.format_data(signal_dict, background_dict)

#scales features to be between 0 and 1
features, labels, scaler = lqm.preprocess_data(features, labels)

n_signal_events = (labels == 1).sum()

#splits data into testing and training sets
#data is first cut to inlcude equal number of signal and background events
#TODO: maybe split signal and backgrounds seperately to ensure equal number of signal/background in each test/training set and then combine and randomize order
train_features, test_features, train_labels, test_labels = train_test_split(
    features[:2*n_signal_events,:], labels[:2*n_signal_events], train_size=0.8, random_state=seed)


losses = []
times = []

def callback(weights, step_loss):
#function called at each step of vqc training
#expected to only have these two inputs and return None
    losses.append(step_loss)
    times.append(time.time())
    print(f"Iteration {len(losses)} Training time: {round(times[-1]-times[-2],2)} seconds")
    print(f"Loss: {step_loss:.4f}")

vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback
)

start = time.time()
times.append(start)
vqc.fit(train_features[:n_training_points,:], train_labels[:n_training_points]) #training the vqc
elapsed = time.time() - start

print(f"Training time: {round(elapsed)} seconds\n")

lqm.save_model(vqc, save_folder, seed=seed, n_training_points=n_training_points, training_feature_keys=training_feature_keys, files_used=files_used)
lqm.score_model(vqc, train_features, test_features, train_labels, test_labels)

lqm.plot_loss(losses)