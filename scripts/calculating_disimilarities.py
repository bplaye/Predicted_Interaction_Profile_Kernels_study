import os
import numpy as np
import pickle
import argparse
import sys

from sklearn.cross_validation import train_test_split, KFold, LeaveOneOut
from sklearn.svm import SVC
from sklearn.metrics import zero_one_loss
from sklearn.metrics.pairwise import linear_kernel

from load_dataset import load_dataset


def get_files_paths(base_path):
	'''
	Recieve the base directory path and returns de paht to all the data files.
	:param base_path: base path to the data root folder
	:return: Returns 6 files paht to the different files. The matrix of positives instances, a list of proteins, a list
		of molecules, a kernel of molecules similarities, a dictionary for molecules to indices in the kernel,
		and the corresponding dictionary for indices to molecules.
	'''
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

	print(filename_mol_kernel)
	return filename_positive_instances_dictionnary, filename_list_prot, filename_list_mol, filename_mol_kernel, \
	       filename_dicomolkernel_indice2instance, filename_dicomolkernel_instance2indice


def calculate_disimilarity(x, y, printed, test_size=0.1, ):
	'''
	Calculate a disimilarity matrix between the different tasks.
	:param x: An arrray containing the kernel for the instances.
	:param y: a matrix containing the target value for each x in the rows, and the task in the columns.
	:param test_size: A float containing the proportion of test dataset to be used.
	:return: A n_task by n_task matrix containing.
	'''
	n, _ = x.shape
	_, n_tasks = y.shape

	train, test = train_test_split(np.arange(n), test_size=test_size)
	models = []
	preds = []
	selected_indices = []
	x_tr = x[train, :][:, train]
	x_te = x[test, :][:, train]
	y_tr = y[train, :]
	y_te = y[test, :]

	# train one model for each task
	for t in range(n_tasks):
		print('Dissimilarity {}: train_model: {}/{}'.format(printed, t + 1, n_tasks))
		sys.stdout.flush()

		y_tr_task = y_tr[:, t]
		if np.unique(y_tr_task).size == 1:
			continue
		index = np.arange(y_tr_task.size)
		selected = y_tr_task == 1
		id_neg = np.random.choice(index[y_tr_task != 1], size=np.sum(selected), replace=False)
		selected[id_neg] = True
		selected_indices.append(selected)
		model = SVC(kernel='precomputed', )

		model.fit(x_tr[selected, :][:, selected], y_tr_task[selected])
		pred = model.predict(x_te[:, selected]).flatten()
		preds.append(pred)
		models.append(model)

	# Create an array of the predictions, with each prediction in a column
	preds = np.array(preds).T
	# calculate losses by calculaten the difference between each prediction and eact task target array
	losses = np.mean((preds[:, None, :] == y_te[:, :, None]).astype(np.float32), axis=0)
	# Calculate the distances betweeen losses from the same task and calculate the maximum for each task
	disimilarity_matrix = np.max(np.abs(losses[None, :, :] - losses[:, None, :]), axis=2)

	# for t_1 in range(n_tasks):
	# 	for t_2 in range(t_1, n_tasks):
	# 		print('Dissimilarity {}: fill dissimilaries: {},{}/{}'.format(printed, t_1, t_2, n_tasks))
	# 		sys.stdout.flush()
	#
	# 		loss_differences = []
	# 		for i, _ in enumerate(models):
	# 			per1 = zero_one_loss(preds[i]
	# 			                     , y_te[:, t_1].flatten())
	# 			per2 = zero_one_loss(preds[i], y_te[:, t_2].flatten())
	# 			loss_differences.append(np.abs(per1 - per2))
	#
	# 		disimilarity_matrix[t_1, t_2] = np.max(loss_differences)
	# 		disimilarity_matrix[t_2, t_1] = disimilarity_matrix[t_1, t_2]

	return disimilarity_matrix


def make_prediction(x_tr, y_tr, x_te, disimilarity_matrix, leaveout):
	'''
	Trains a model according to the disimilarity between the task.
	:param x_tr: A kernel for the training data.
	:param y_tr: The training targets.
	:param x_te: The corresponding kernel for prediction.
	:param disimilarity_matrix: The disimilarity matrix between the tasks.
	:return: Return an array containing the predictions, and the final models.
	'''
	# Calculate task kernel
	print("Training Task Kernek {}".format(leaveout))
	sys.stdout.flush()
	task_kernel = linear_kernel(disimilarity_matrix)
	sys.stdout.flush()
	print("Finished Task Kernek {}".format(leaveout))
	# Train the model

	non_zero = np.argwhere(y_tr == 1)
	zero = np.argwhere(y_tr != 1)
	zero = zero[np.random.choice(np.arange(zero.shape[0]), size=non_zero.shape[0]), :]
	selected = np.concatenate((non_zero, zero), axis=0)
	final_y_tr = np.array([y_tr[i, j] for i, j in selected])

	print("Creating Multitask Matrix {}".format(leaveout))
	sys.stdout.flush()
	multitask_kernel_tr = np.zeros((selected.shape[0], selected.shape[0]), dtype=np.float32)
	print("Filling Multitask Matrix {}".format(leaveout))
	sys.stdout.flush()
	multitask_kernel_tr = x_tr[selected[:, 0], :][:, selected[:, 0]] * task_kernel[selected[:, 1], :][:, selected[:, 1]]
	print("Test MultiTask Kernel {}".format(leaveout))
	sys.stdout.flush()
	multitask_kernel_te = np.tile(x_te[:, selected[:, 0]], (task_kernel.shape[1], 1)) * \
	                      np.repeat(task_kernel[:, selected[:, 1]], x_te.shape[0], axis=0)

	print("Training SVM {}".format(leaveout))
	sys.stdout.flush()
	model = SVC(kernel='precomputed', probability=True)
	model.fit(multitask_kernel_tr, final_y_tr)
	# Predict
	print("Predicting SVM {}".format(leaveout))
	sys.stdout.flush()

	y_pred = model.predict(multitask_kernel_te)
	# Reshape the matrix as (n_te, T) array.
	return y_pred.reshape((x_te.shape[0], y_tr.shape[1])), model


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('-p', '--path', nargs='?', default='.', type=str, dest='base_data_path', action='store')
	parser.add_argument('-o', '--out', nargs='?', default='results.pickle', type=str, dest='output_filename')
	parser.add_argument('-l', '--leaveoneout', nargs='?', default=None, type=int, dest='leaveout')
	args = parser.parse_args()

	print(args)

	base_data_path = args.base_data_path
	output_filename = args.output_filename
	leaveout = args.leaveout

	files_paths = get_files_paths(base_data_path)
	K_mol, _, _, interaction_matrix = load_dataset(*files_paths)

	tr_idx = np.ones(K_mol.shape[0], dtype=bool)
	tr_idx[leaveout] = False
	te_idx = np.zeros(K_mol.shape[0], dtype=bool)
	te_idx[leaveout] = True

	x_training = K_mol[tr_idx, :][:, tr_idx]
	x_testing = K_mol[te_idx, :][:, tr_idx]
	y_training = interaction_matrix[tr_idx, :]
	y_testing = interaction_matrix[te_idx, :]
	disimilarities = calculate_disimilarity(x_training, y_training, leaveout)
	prediction, final_model = make_prediction(x_training, y_training, x_testing, disimilarities, leaveout)

	results = {
		'prediction': (prediction, y_testing),
		'model': final_model,
		'fold': (tr_idx, te_idx),
		'disimilarity': disimilarities,
	}

	with open(output_filename, 'wb') as f:
		pickle.dump(results, f)
