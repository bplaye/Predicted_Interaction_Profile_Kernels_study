import csv
import random
import numpy as np
import linecache
import math
import time
import statistics
import os
import sys
import pickle
import collections

from sklearn import svm
from sklearn import metrics
from sklearn.externals import joblib



	
############################################################################
############################################################################
	
def make_list_labels(dico_indice2mol_InMolKernel, list_ligand_of_prot):
	list_labels = []
	for i in range(len(dico_indice2mol_InMolKernel.keys())):
		if dico_indice2mol_InMolKernel[i] in list_ligand_of_prot:
			list_labels.append(1)
		else:
			list_labels.append(-1)
	
	return list_labels

############################################################################
############################################################################

def compute_profile(K_mol, C_SVM, list_labels, type_of_profile):
	profile = np.zeros(len(K_mol))
	
	for i in range(len(K_mol)):
		K_train = np.delete(K_mol, i, axis=0)
		K_train = np.delete(K_train, i, axis=1)
		K_test = np.delete(K_mol[i,:], i, axis=0)
		list_train_labels = list_labels.copy()
		del list_train_labels[i]
		
		dico_train_labels = collections.Counter(list_train_labels)
		dico_weight = {1:float(len(list_train_labels))/float(2*dico_train_labels[1]) , -1:float(len(list_train_labels))/float(2*dico_train_labels[-1])}
		
		clf = svm.SVC(kernel='precomputed', C=C_SVM)
		clf.fit(K_train, list_train_labels)
		if type_of_profile=='pred':
			if clf.predict(K_test).tolist()[0]==1:
				profile[i] = 1
			else:
				profile[i] = 0
		elif type_of_profile=='score':
			profile[i] = clf.decision_function(K_test).tolist()[0]
	if type_of_profile=='score':
		max_ = np.max(profile)
		min_ = np.min(profile)
		for i in range(len(K_mol)):
			profile[i] = (profile[i]-min_)/(max_-min_)
	return profile

############################################################################
############################################################################
############################################################################
############################################################################

def make_true_profile(dico_indice2mol_InMolKernel, list_ligand_of_prot):
	true_profile = np.zeros(len(dico_indice2mol_InMolKernel.keys()), np.int)
	for i in range(len(dico_indice2mol_InMolKernel.keys())):
		if dico_indice2mol_InMolKernel[i] in list_ligand_of_prot:
			true_profile[i] = 1
		else:
			true_profile[i] = 0
	
	return true_profile

############################################################################
############################################################################

def make_pred_profile(K_mol, C_SVM, dico_indice2mol_InMolKernel, list_ligand_of_prot):
	
	list_labels = make_list_labels(dico_indice2mol_InMolKernel, list_ligand_of_prot)
	return compute_profile(K_mol, C_SVM, list_labels, 'pred')

############################################################################
############################################################################

def make_TrueAndPred_profile(K_mol, C_SVM, dico_indice2mol_InMolKernel, list_ligand_of_prot):
	
	list_labels = make_list_labels(dico_indice2mol_InMolKernel, list_ligand_of_prot)
	pred_profile = compute_profile(K_mol, C_SVM, list_labels, 'pred')
	TrueAndPred_profile = np.zeros(len(list_labels))
	for ind in range(len(list_labels)):
		if list_labels[ind]==1:
			TrueAndPred_profile[ind] = 1
		else:
			TrueAndPred_profile[ind] = pred_profile[ind]

	return TrueAndPred_profile

############################################################################
############################################################################
	
def make_score_profile(K_mol, C_SVM, dico_indice2mol_InMolKernel, list_ligand_of_prot):

	list_labels = make_list_labels(dico_indice2mol_InMolKernel, list_ligand_of_prot)
	return compute_profile(K_mol, C_SVM, list_labels, 'score')

############################################################################
############################################################################
	
def make_TrueAndScore_profile(K_mol, C_SVM, dico_indice2mol_InMolKernel, list_ligand_of_prot):

	list_labels = make_list_labels(dico_indice2mol_InMolKernel, list_ligand_of_prot)
	score_profile =  compute_profile(K_mol, C_SVM, list_labels, 'score')
	TrueAndScore_profile = np.zeros(len(list_labels))
	for ind in range(len(list_labels)):
		if list_labels[ind]==1:
			TrueAndScore_profile[ind] = 1
		else:
			TrueAndScore_profile[ind] = score_profile[ind]

	return TrueAndScore_profile

############################################################################
############################################################################
############################################################################
############################################################################

def modify_IP_kernel(couple, IP_kernel, average_nbMol_per_target, dico_indice2prot_InIPKernel, ind_of_prot, dico_profile_per_prot, profile_of_prot, C_SVM):
	local_K_IP = IP_kernel.copy()
	
	for ind in range(len(IP_kernel)):
		local = profile_of_prot-dico_profile_per_prot[dico_indice2prot_InIPKernel[ind]][C_SVM]
		local_K_IP[ind_of_prot,ind] = np.exp((-1/average_nbMol_per_target)*np.dot(np.transpose(local),local))
		local_K_IP[ind,ind_of_prot] = local_K_IP[ind_of_prot,ind]
	
	return local_K_IP
	
def modify_GIP_kernel(couple, IP_kernel, average_nbMol_per_target, dico_indice2prot_InIPKernel, ind_of_prot, dico_profile_per_prot, profile_of_prot):
	local_K_IP = IP_kernel.copy()
	
	for ind in range(len(IP_kernel)):
		local = profile_of_prot-dico_profile_per_prot[dico_indice2prot_InIPKernel[ind]]
		local_K_IP[ind_of_prot,ind] = np.exp((-1/average_nbMol_per_target)*np.dot(np.transpose(local),local))
		local_K_IP[ind,ind_of_prot] = local_K_IP[ind_of_prot,ind]
	
	return local_K_IP

############################################################################
############################################################################
############################################################################
############################################################################

def make_train_and_test(element, K, dico_indice, list_pos, list_neg_for_train, label):
	
	list_train_labels = []
	list_train = []
	
	if label==1:
		for el in list_pos:
			if el!=element:
				list_train.append(el)
				list_train_labels.append(1)
		for el in list_neg_for_train[:-1]: ### on prend un nombre de pos et de neg balanced
				list_train.append(el)
				list_train_labels.append(-1)
	else:
		for el in list_pos:
				list_train.append(el)
				list_train_labels.append(1)
		for el in list_neg_for_train:
				list_train.append(el)
				list_train_labels.append(-1)
	
	K_train = np.zeros((len(list_train), len(list_train)))
	K_test = np.zeros(len(list_train))
	ind_test = dico_indice[element]
	for ind1 in range(len(list_train)):
		ind1_K = dico_indice[list_train[ind1]]
		for ind2 in range(ind1,len(list_train)):
			ind2_K = dico_indice[list_train[ind2]]
			K_train[ind1,ind2] = K[ind1_K, ind2_K]
			K_train[ind2,ind1] = K_train[ind1,ind2]
		K_test[ind1] = K[ind1_K, ind_test]
	
	return K_train, K_test, list_train_labels
	
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################


################## on télécharge le DAtaset
file_PosDic = "../dictionnaries_and_lists/dico_for_discrepency_study/SmallMolMWFilter_UniprotHumanProt_DrugBank_Dictionary.csv"
file_NegDic = "../dictionnaries_and_lists/dico_for_discrepency_study/bootstrapped:NegDictionnaries/SmallMolMWFilter_UniprotHumanProt_DrugBank_Bootstraped_NegDictionary_for_test_for_mol.csv"
file_NegDic_balanced = "../dictionnaries_and_lists/dico_for_discrepency_study/bootstrapped:NegDictionnaries/SmallMolMWFilter_UniprotHumanProt_DrugBank_Bootstraped_NegDictionary_for_balanced_for_mol.csv"
file_ProtList = "../dictionnaries_and_lists/dico_for_discrepency_study/list_MWFilter_UniprotHumanProt.txt"
file_MolList = "../dictionnaries_and_lists/dico_for_discrepency_study/list_MWFilter_mol.txt"


#### instances
list_prot_of_dataset = []
list_mol_of_dataset = []
dico_labels_per_couple = {}
dico_target_of_mol = {}
dico_ligand_of_prot = {}
list_pos_couples = []
list_neg_couples = []
####

####
##on charge list_prot_of_dataset
f_in = open(file_ProtList, 'r')
for line in f_in:
	list_prot_of_dataset.append(line.rstrip())
f_in.close()
##on charge list_mol_of_dataset
f_in = open(file_MolList, 'r')
for line in f_in:
	list_mol_of_dataset.append(line.rstrip())
f_in.close()
## on charge dico_labels_per_couple, dico_target_of_mol, dico_ligand_of_prot, list_pos_couples 
for mol in list_mol_of_dataset:
	dico_target_of_mol[mol] = [[],[],[],[]] # 1: list of target, 2: list of non target; 3: list of neg for test; 4: list of neg for balanced train
for prot in list_prot_of_dataset:
	dico_ligand_of_prot[prot] = [[],[],[],[]]

f_in = open(file_PosDic, 'r')
reader = csv.reader(f_in, delimiter='\t')
for row in reader:
	nb_prot = int(row[4])
	j=0
	while j<nb_prot:
		dico_target_of_mol[row[0]][0].append(row[5+j])
		dico_ligand_of_prot[row[5+j]][0].append(row[0])
		dico_labels_per_couple[row[0]+row[5+j]] = 1
		list_pos_couples.append((row[5+j],row[0]))
		j+=1
del reader
f_in.close()

for prot in list_prot_of_dataset:
	for cle, valeur in dico_target_of_mol.items():
		if prot not in valeur[0]:
			dico_target_of_mol[cle][1].append(prot)
			dico_ligand_of_prot[prot][1].append(cle)

f_in = open(file_NegDic, 'r')
reader = csv.reader(f_in, delimiter='\t')
for row in reader:
	nb_prot = int(row[4])
	j=0
	while j<nb_prot:
		dico_target_of_mol[row[0]][2].append(row[5+j])
		dico_ligand_of_prot[row[5+j]][2].append(row[0])
		dico_labels_per_couple[row[0]+row[5+j]] = -1
		list_neg_couples.append((row[5+j],row[0]))
		j+=1
del reader
f_in.close()

f_in = open(file_NegDic_balanced, 'r')
reader = csv.reader(f_in, delimiter='\t')
for row in reader:
	nb_prot = int(row[4])
	j=0
	while j<nb_prot:
		dico_target_of_mol[row[0]][3].append(row[5+j])
		dico_ligand_of_prot[row[5+j]][3].append(row[0])
		dico_labels_per_couple[row[0]+row[5+j]] = -1
		j+=1
del reader
f_in.close()
################## on télécharge le DAtaset

############################################################################
############################################################################

################## on télécharge tout selon on a besoin
file_MolKernel = "../kernels/kernels.data/Tanimoto_d=8_DrugBankSmallMolMWFilterHuman.data"
file_ProtKernel = "../kernels/kernels.data/allDrugBankHumanTarget_Profile_normalized_k5_threshold7.5.data"
file_DicoMolKernel_indice2 = "../kernels/dict/dico_indice2mol_InMolKernel.data"
file_DicoMolKernel_2indice = "../kernels/dict/dico_mol2indice_InMolKernel.data"
file_DicoProtKernel_indice2 = "../kernels/dict/dico_indice2prot_InProtKernel.data"
file_DicoProtKernel_2indice = "../kernels/dict/dico_prot2indice_InProtKernel.data"

C_SVM = 1




with open(file_MolKernel, 'rb') as fichier:
	pickler = pickle.Unpickler(fichier)
	K_mol = pickler.load()
with open(file_DicoMolKernel_indice2, 'rb') as fichier:
	pickler = pickle.Unpickler(fichier)
	dico_indice2mol_InMolKernel = pickler.load()
with open(file_DicoMolKernel_2indice, 'rb') as fichier:
	pickler = pickle.Unpickler(fichier)
	dico_mol2indice_InMolKernel = pickler.load()

if sys.argv[1]=="GIP" or sys.argv[1]=="GPIP_score" or sys.argv[1]=="GPIP_pred" or sys.argv[1]=="GPIP_TrueAndPred" or sys.argv[1]=="GPIP_TrueAndScore":
	with open("kernels/dico_prot2indice_InIPKernel.data", 'rb') as fichier:
		pickler = pickle.Unpickler(fichier)
		dico_prot2indice_InIPKernel = pickler.load()
	with open("kernels/dico_indice2prot_InIPKernel.data", 'rb') as fichier:
		pickler = pickle.Unpickler(fichier)
		dico_indice2prot_InIPKernel = pickler.load()

if sys.argv[1]=="GIP":
	with open("kernels/average_nbMol_per_target_for_GIP.data", 'rb') as fichier:
		pickler = pickle.Unpickler(fichier)
		average_nbMol_per_target_for_GIP = pickler.load()
	with open("kernels/K_prot_GIP.data", 'rb') as fichier:
		pickler = pickle.Unpickler(fichier)
		K_prot = pickler.load()
	with open("dico_profile/dico_true_profile_per_prot.data", 'rb') as fichier:
		pickler = pickle.Unpickler(fichier)
		dico_true_profile_per_prot= pickler.load()

elif sys.argv[1]=="GPIP_score":
	with open("kernels/average_nbMol_per_target_for_GPIP_score_C="+str(C_SVM)+".data", 'rb') as fichier:
		pickler = pickle.Unpickler(fichier)
		average_nbMol_per_target_for_GPIP_score = pickler.load()
	with open("kernels/K_prot_GPIP_score_C="+str(C_SVM)+".data", 'rb') as fichier:
		pickler = pickle.Unpickler(fichier)
		K_prot = pickler.load()
	with open("dico_profile/dico_score_profile_per_prot.data", 'rb') as fichier:
		pickler = pickle.Unpickler(fichier)
		dico_score_profile_per_prot = pickler.load()
	
elif sys.argv[1]=="GPIP_pred":
	with open("kernels/average_nbMol_per_target_for_GPIP_pred_C="+str(C_SVM)+".data", 'rb') as fichier:
		pickler = pickle.Unpickler(fichier)
		average_nbMol_per_target_for_GPIP_pred = pickler.load()
	with open("kernels/K_prot_GPIP_pred_C="+str(C_SVM)+".data", 'rb') as fichier:
		pickler = pickle.Unpickler(fichier)
		K_prot = pickler.load()
	with open("dico_profile/dico_pred_profile_per_prot.data", 'rb') as fichier:
		pickler = pickle.Unpickler(fichier)
		dico_pred_profile_per_prot = pickler.load()

elif sys.argv[1]=="GPIP_TrueAndPred":
	with open("kernels/average_nbMol_per_target_for_GPIP_TrueAndPred_C="+str(C_SVM)+".data", 'rb') as fichier:
		pickler = pickle.Unpickler(fichier)
		average_nbMol_per_target_for_GPIP_TrueAndPred = pickler.load()
	with open("kernels/K_prot_GPIP_TrueAndPred_C="+str(C_SVM)+".data", 'rb') as fichier:
		pickler = pickle.Unpickler(fichier)
		K_prot = pickler.load()
	with open("dico_profile/dico_TrueAndPred_profile_per_prot.data", 'rb') as fichier:
		pickler = pickle.Unpickler(fichier)
		dico_TrueAndPred_profile_per_prot = pickler.load()
		
elif sys.argv[1]=="GPIP_TrueAndScore":
	with open("kernels/average_nbMol_per_target_for_GPIP_TrueAndScore_C="+str(C_SVM)+".data", 'rb') as fichier:
		pickler = pickle.Unpickler(fichier)
		average_nbMol_per_target_for_GPIP_TrueAndScore = pickler.load()
	with open("kernels/K_prot_GPIP_TrueAndScore_C="+str(C_SVM)+".data", 'rb') as fichier:
		pickler = pickle.Unpickler(fichier)
		K_prot = pickler.load()
	with open("dico_profile/dico_TrueAndScore_profile_per_prot.data", 'rb') as fichier:
		pickler = pickle.Unpickler(fichier)
		dico_TrueAndScore_profile_per_prot = pickler.load()	
		
elif sys.argv[1]=="sequence":
	with open(file_ProtKernel, 'rb') as fichier:
		pickler = pickle.Unpickler(fichier)
		K_prot = pickler.load()
	with open(file_DicoProtKernel_indice2, 'rb') as fichier:
		pickler = pickle.Unpickler(fichier)
		dico_indice2prot_InProtKernel = pickler.load()
	with open(file_DicoProtKernel_2indice, 'rb') as fichier:
		pickler = pickle.Unpickler(fichier)
		dico_prot2indice_InProtKernel = pickler.load()
################## on télécharge tout selon on a besoin	

############################################################################
############################################################################

list_C = [0.01, 0.1, 0.5, 1, 5, 10, 100]

Y_pred = []
Y_score = []
for c in range(len(list_C)):
	Y_pred.append([])
	Y_score.append([])
dico_result_per_couple = {}
Y_true = []

############################################################################
############################################################################

list_mol_of_dataset = list_mol_of_dataset[:200].copy()
granularity = math.ceil(float(len(list_mol_of_dataset))/float(40))
list_mol_of_dataset = list_mol_of_dataset[int(sys.argv[2])*granularity:min((int(sys.argv[2])+1)*granularity,len(list_mol_of_dataset))].copy()


for mol_current in list_mol_of_dataset:
	print(mol_current)
	for prot_current in dico_target_of_mol[mol_current][0]:

			if len(dico_target_of_mol[mol_current][0])!=1 or dico_target_of_mol[mol_current][0][0]!=prot_current:
				print(prot_current)
				Y_true.append(1)
				dico_result_per_couple[(prot_current,mol_current)] = {}
	
				if sys.argv[1]=="GIP" or sys.argv[1]=="GPIP_score" or sys.argv[1]=="GPIP_pred" or sys.argv[1]=="GPIP_TrueAndPred" or sys.argv[1]=="GPIP_TrueAndScore":
					modified_list_ligand_of_prot = dico_ligand_of_prot[prot_current][0].copy()
					deleting_index = modified_list_ligand_of_prot.index(mol_current)
					del modified_list_ligand_of_prot[deleting_index]
		
				if sys.argv[1]=="GIP":
					true_profile_of_prot = make_true_profile(dico_indice2mol_InMolKernel, modified_list_ligand_of_prot)
					local_GIP = modify_GIP_kernel((prot_current,mol_current), K_prot, average_nbMol_per_target_for_GIP, dico_indice2prot_InIPKernel, dico_prot2indice_InIPKernel[prot_current], dico_true_profile_per_prot, true_profile_of_prot)
					K_train, K_test, list_train_labels = make_train_and_test(prot_current, local_GIP, dico_prot2indice_InIPKernel, dico_target_of_mol[mol_current][0], dico_target_of_mol[mol_current][3], 1)
		
				elif sys.argv[1]=="GPIP_score":
					score_profile_of_prot = make_score_profile(K_mol, C_SVM, dico_indice2mol_InMolKernel, modified_list_ligand_of_prot)
					local_GPIP_score = modify_IP_kernel((prot_current,mol_current), K_prot, average_nbMol_per_target_for_GPIP_score, dico_indice2prot_InIPKernel, dico_prot2indice_InIPKernel[prot_current], dico_score_profile_per_prot, score_profile_of_prot, C_SVM)
					K_train, K_test, list_train_labels = make_train_and_test(prot_current, local_GPIP_score, dico_prot2indice_InIPKernel, dico_target_of_mol[mol_current][0], dico_target_of_mol[mol_current][3], 1)
		
				elif sys.argv[1]=="GPIP_pred":
					pred_profile_of_prot = make_pred_profile(K_mol, C_SVM, dico_indice2mol_InMolKernel, modified_list_ligand_of_prot)
					local_GPIP_pred = modify_IP_kernel((prot_current,mol_current), K_prot, average_nbMol_per_target_for_GPIP_pred, dico_indice2prot_InIPKernel, dico_prot2indice_InIPKernel[prot_current], dico_pred_profile_per_prot, pred_profile_of_prot, C_SVM)
					K_train, K_test, list_train_labels = make_train_and_test(prot_current, local_GPIP_pred, dico_prot2indice_InIPKernel, dico_target_of_mol[mol_current][0], dico_target_of_mol[mol_current][3], 1)
		
				elif sys.argv[1]=="GPIP_TrueAndPred":
					TrueAndPred_profile_of_prot = make_pred_profile(K_mol, C_SVM, dico_indice2mol_InMolKernel, modified_list_ligand_of_prot)
					local_GPIP_TrueAndPred = modify_IP_kernel((prot_current,mol_current), K_prot, average_nbMol_per_target_for_GPIP_TrueAndPred, dico_indice2prot_InIPKernel, dico_prot2indice_InIPKernel[prot_current], dico_TrueAndPred_profile_per_prot, TrueAndPred_profile_of_prot, C_SVM)
					K_train, K_test, list_train_labels = make_train_and_test(prot_current, local_GPIP_TrueAndPred, dico_prot2indice_InIPKernel, dico_target_of_mol[mol_current][0], dico_target_of_mol[mol_current][3], 1)
		
				elif sys.argv[1]=="GPIP_TrueAndScore":
					TrueAndScore_profile_of_prot = make_pred_profile(K_mol, C_SVM, dico_indice2mol_InMolKernel, modified_list_ligand_of_prot)
					local_GPIP_TrueAndScore = modify_IP_kernel((prot_current,mol_current), K_prot, average_nbMol_per_target_for_GPIP_TrueAndScore, dico_indice2prot_InIPKernel, dico_prot2indice_InIPKernel[prot_current], dico_TrueAndScore_profile_per_prot, TrueAndScore_profile_of_prot, C_SVM)
					K_train, K_test, list_train_labels = make_train_and_test(prot_current, local_GPIP_TrueAndScore, dico_prot2indice_InIPKernel, dico_target_of_mol[mol_current][0], dico_target_of_mol[mol_current][3], 1)
	
				elif sys.argv[1]=="sequence":
					K_train, K_test, list_train_labels = make_train_and_test(prot_current, K_prot, dico_prot2indice_InProtKernel, dico_target_of_mol[mol_current][0], dico_target_of_mol[mol_current][3], 1)


				for c in range(len(list_C)): ### pour chaque C on entraine, on test, et on enregistre la prédiction
					clf = svm.SVC(kernel='precomputed', C=list_C[c])
					clf.fit(K_train, list_train_labels)
					Y_test_pred = clf.predict(K_test).tolist()
					Y_test_score = clf.decision_function(K_test).tolist()

					dico_result_per_couple[(prot_current,mol_current)]["C="+str(list_C[c])] = (Y_test_pred[0], Y_test_score[0])

					Y_pred[c] += Y_test_pred
					Y_score[c] += Y_test_score

					del clf
					del Y_test_pred
					del Y_test_score
				del K_train
				del list_train_labels
				del K_test	



	for prot_current in dico_target_of_mol[mol_current][2]:
		print(prot_current)
		Y_true.append(-1)
		dico_result_per_couple[(prot_current,mol_current)] = {}

		if sys.argv[1]=="GIP" or sys.argv[1]=="GPIP_score" or sys.argv[1]=="GPIP_pred" or sys.argv[1]=="GPIP_TrueAndPred" or sys.argv[1]=="GPIP_TrueAndScore":
			K_train, K_test, list_train_labels = make_train_and_test(prot_current, K_prot, dico_prot2indice_InIPKernel, dico_target_of_mol[mol_current][0], dico_target_of_mol[mol_current][3], -1)
		elif sys.argv[1]=="sequence":
			K_train, K_test, list_train_labels = make_train_and_test(prot_current, K_prot, dico_prot2indice_InProtKernel, dico_target_of_mol[mol_current][0], dico_target_of_mol[mol_current][3], -1)

		for c in range(len(list_C)): ### pour chaque C on entraine, on test, et on enregistre la prédiction
			clf = svm.SVC(kernel='precomputed', C=list_C[c])
			clf.fit(K_train, list_train_labels)
			Y_test_pred = clf.predict(K_test).tolist()
			Y_test_score = clf.decision_function(K_test).tolist()

			dico_result_per_couple[(prot_current,mol_current)]["C="+str(list_C[c])] = (Y_test_pred[0], Y_test_score[0])

			Y_pred[c] += Y_test_pred
			Y_score[c] += Y_test_score

			del clf
			del Y_test_pred
			del Y_test_score
		del K_train
		del list_train_labels
		del K_test
				
	
############################################################################
############################################################################

with open("result/dict/"+sys.argv[1]+"_Ytrue_"+sys.argv[2]+".data", 'wb') as fichier:
	pickler = pickle.Pickler(fichier)
	pickler.dump(Y_true)
with open("result/dict/"+sys.argv[1]+"_Ypred_"+sys.argv[2]+".data", 'wb') as fichier:
	pickler = pickle.Pickler(fichier)
	pickler.dump(Y_pred)
with open("result/dict/"+sys.argv[1]+"_Yscore_"+sys.argv[2]+".data", 'wb') as fichier:
	pickler = pickle.Pickler(fichier)
	pickler.dump(Y_score)
with open("result/dict/"+sys.argv[1]+"_DicoResultPerCouple_"+sys.argv[2]+".data", 'wb') as fichier:
	pickler = pickle.Pickler(fichier)
	pickler.dump(dico_result_per_couple)
	
