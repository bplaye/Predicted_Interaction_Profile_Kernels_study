import os
import numpy as np
import pickle
import argparse

from sklearn.cross_validation import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.metrics import zero_one_loss
from sklearn.metrics.pairwise import rbf_kernel

from scripts.load_dataset import load_dataset


def get_files_paths(base_path):
	filename_positive_instances_dictionnary = os.path.join(
		base_path, "dictionnaries_and_lists/SmallMolMWFilter_UniprotHumanProt_DrugBank_Dictionary.csv"
	)

	filename_list_prot = os.path.join(base_path, "dictionnaries_and_lists/list_MWFilter_UniprotHumanProt.txt")
	filename_list_mol = os.path.join(base_path, "dictionnaries_and_lists/list_MWFilter_mol.txt")

	filename_mol_kernel = os.path.join(
		base_path,
		"kernels/kernels.data/Tanimoto_d=8_DrugBankSmallMolMWFilterHuman.data"
	)
	filename_dicomolkernel_indice2instance = os.path.join(base_path, "kernels/dict/dico_indice2mol_InMolKernel.data")
	filename_dicomolkernel_instance2indice = os.path.join(base_path, "kernels/dict/dico_mol2indice_InMolKernel.data")

	return filename_positive_instances_dictionnary, filename_list_prot, filename_list_mol, filename_mol_kernel, \
		filename_dicomolkernel_indice2instance, filename_dicomolkernel_instance2indice


def calculate_disimilarity(x, y):
	n, p = x.shape
	_, n_tasks = y.shape

	train, test = train_test_split(np.arange(n), test_size=0.1, )
	models = [SVC(kernel='precomputed') for _ in range(n_tasks)]
	x_tr = x[train, :][:, train]
	x_te = x[test, :][:, train]
	y_tr = y[train, :]
	y_te = y[test, :]
	# train one model for each task
	for t in range(n_tasks):
		models[t].fit(x_tr, y_tr[:, t])

	disimilarity_matrix = np.zeros((n_tasks, n_tasks))
	for t_1 in range(n_tasks):
		for t_2 in range(t_1, n_tasks):
			loss_differences = []
			for m_1 in range(n_tasks):
				pred = models[m_1].predict(x_te).flatten()
				per1 = zero_one_loss(pred, y_te[:, t_1].flatten())
				per2 = zero_one_loss(pred, y_te[:, t_2].flatten())
				loss_differences.append(np.abs(per1 - per2))

			disimilarity_matrix[t_1, t_2] = np.max(loss_differences)
			disimilarity_matrix[t_2, t_1] = disimilarity_matrix[t_1, t_2]

	return disimilarity_matrix


def make_prediction(x_tr, y_tr, x_te, disimilarity_matrix):
	# Calculate task kernel
	task_kernel = rbf_kernel(disimilarity_matrix)
	# Train the model
	multitask_kernel_tr = np.kron(x_tr, task_kernel)
	model = SVC(kernel='precomputed', probability=True)
	model.fit(multitask_kernel_tr, y_tr.flatten())
	# Predict
	multitask_kernel_te = np.kron(x_te, task_kernel)
	y_pred = model.predict(multitask_kernel_te)
	# Reshape the matrix as (n_te, T) array.
	return y_pred.reshape((x_te.shape[0], y_tr.shape[1])), model


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('-p', '--path', nargs='?', default='.', type=str, dest='base_data_path')
	parser.add_argument('-o', '--out', nargs='?', default='results.pickle', type=str, dest='output_filename')

	args = parser.parse_args()
	base_data_path = args.base_data_path
	output_filename = args.output_filename

	files_paths = get_files_paths(base_data_path)
	K_mol, DicoMolKernel_ind2mol, DicoMolKernel_mol2ind, interaction_matrix = load_dataset(files_paths)
	predictions = []
	final_models = []
	folds = []
	for tr_idx, te_idx in KFold(KFold.shape[0], n_folds=10):
		folds.append((tr_idx, te_idx))

		x_training = K_mol[tr_idx, :][:, tr_idx]
		x_testing = K_mol[te_idx, :][:, tr_idx]
		y_training = interaction_matrix[tr_idx, :]
		y_testing = interaction_matrix[te_idx, :]
		disimilarities = calculate_disimilarity(x_training, y_training)

		prediction, final_model = make_prediction(x_training, y_training, x_testing, disimilarities)

		predictions.append((prediction, y_testing))
		final_models.append(final_model)

	results = {
		'predictions': predictions,
		'models': final_models,
		'folds': folds
	}

	with open(output_filename, 'w') as f:
		pickle.dump(results, f)
