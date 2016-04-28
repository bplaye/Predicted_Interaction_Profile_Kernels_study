import os
import numpy as np
import pickle

from sklearn.cross_validation import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.metrics import zero_one_loss
from sklearn.metrics.pairwise import rbf_kernel

from scripts.load_dataset import load_dataset


def get_files_paths(base_path):

	FileName_PositiveInstancesDictionnary = os.path.join(
		base_path, "dictionnaries_and_lists/SmallMolMWFilter_UniprotHumanProt_DrugBank_Dictionary.csv"
	)

	FileName_ListProt = os.path.join(base_path, "dictionnaries_and_lists/list_MWFilter_UniprotHumanProt.txt")
	FileName_ListMol = os.path.join(base_path, "dictionnaries_and_lists/list_MWFilter_mol.txt")

	FileName_MolKernel = os.path.join(base_path, "kernels/kernels.data/Tanimoto_d=8_DrugBankSmallMolMWFilterHuman.data")
	FileName_DicoMolKernel_indice2instance = os.path.join(base_path, "kernels/dict/dico_indice2mol_InMolKernel.data")
	FileName_DicoMolKernel_instance2indice = os.path.join(base_path, "kernels/dict/dico_mol2indice_InMolKernel.data")

	return FileName_PositiveInstancesDictionnary, FileName_ListProt, FileName_ListMol, FileName_MolKernel, \
	       FileName_DicoMolKernel_indice2instance, FileName_DicoMolKernel_instance2indice


def calculate_disimilarity(x, y):
	N, P = x.shape
	_, T = y.shape

	train, test = train_test_split(np.arange(N), test_size=0.1, )
	models = [SVC(kernel='precomputed') for task in range(T)]
	x_tr = x[train, :][:,train]
	x_te = x[test, :][:,train]
	y_tr = y[train, :]
	y_te = y[test,:]
	#train one model for each task
	for t in range(T):
		models[t].fit(x_tr, y_tr[:, t])

	disimilarities = np.zeros((T, T))
	for t_1 in range(T):
		for t_2 in range(t_1, T):
			loss_differences = []
			for m_1 in range(T):
				prediction = models[m_1].predict(x_te).flatten()
				per1 = zero_one_loss(prediction, y_te[:,t_1].flatten())
				per2 = zero_one_loss(prediction, y_te[:,t_2].flatten())
				loss_differences.append(np.abs(per1-per2))

			disimilarities[t_1,t_2] = np.max(loss_differences)
			disimilarities[t_2,t_1] = disimilarities[t_1,t_2]

	return disimilarities

def make_prediction(x_tr, y_tr, x_te, disimilarities):
	#Calculate task kernel
	task_kernel = rbf_kernel(disimilarities)
	#Train the model
	multitask_kernel_tr = np.kron(x_tr, task_kernel)
	model = SVC(kernel='precomputed', probability=True)
	model.fit(multitask_kernel_tr, y_tr.flatten())
	#Predict
	multitask_kernel_te = np.kron(x_te, task_kernel)
	prediction =  model.predict(multitask_kernel_te)
	#Reshape the matrix as (n_te, T) array.
	return prediction.reshape((x_te.shape[0], y_tr.shape[1])) , model

if __name__ == '__main__':
	base_path = '.'
	files_paths = get_files_paths(base_path)
	K_mol, DicoMolKernel_ind2mol, DicoMolKernel_mol2ind, interaction_matrix = load_dataset(files_paths)
	predictions = []
	final_models = []
	folds = []
	for tr_idx, te_idx in KFold(KFold.shape[0], n_folds=10):
		folds.append((tr_idx,te_idx))

		x_tr = K_mol[tr_idx, :][:, tr_idx]
		x_te = K_mol[te_idx, :][:, tr_idx]
		y_tr = interaction_matrix[tr_idx, :]
		y_te = interaction_matrix[te_idx, :]
		disimilarities = calculate_disimilarity(x_tr, y_tr)

		prediction, model = make_prediction(x_tr, y_tr, x_te, disimilarities )

		predictions.append((prediction,y_te))
		final_models.append(model)


	results = {
		'predictions': predictions,
		'models': final_models,
		'folds': folds
	}
	with open('outputfile.pickle', 'w') as f:
		pickle.dump(results, f)