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

# BASE PATH DEFINITIONS
DATA_BASE_PATH = '..'
OUTPUT_BASE_PATH = 'dico_profile/temp/'


#DATA PATHS
file_MolKernel = os.path.join(DATA_BASE_PATH, "kernels/kernels.data/Tanimoto_d=8_DrugBankSmallMolMWFilterHuman.data")
file_ProtKernel = os.path.join(DATA_BASE_PATH, "kernels/kernels.data/allDrugBankHumanTarget_Profile_normalized_k5_threshold7.5.data")
file_DicoMolKernel_indice2 = os.path.join(DATA_BASE_PATH, "kernels/dict/dico_indice2mol_InMolKernel.data")
file_DicoMolKernel_2indice = os.path.join(DATA_BASE_PATH, "kernels/dict/dico_mol2indice_InMolKernel.data")
file_DicoProtKernel_indice2 = os.path.join(DATA_BASE_PATH, "kernels/dict/dico_indice2prot_InProtKernel.data")
file_DicoProtKernel_2indice = os.path.join(DATA_BASE_PATH, "kernels/dict/dico_prot2indice_InProtKernel.data")

file_PosDic = os.path.join(DATA_BASE_PATH, "dictionnaries_and_lists/SmallMolMWFilter_UniprotHumanProt_DrugBank_Dictionary.csv")
file_NegDic = os.path.join(DATA_BASE_PATH, "dictionnaries_and_lists/bootstrapped:NegDictionnaries/SmallMolMWFilter_UniprotHumanProt_DrugBank_Bootstraped_NegDictionary_for_test_for_prot.csv")
file_NegDic_balanced = os.path.join(DATA_BASE_PATH, "dictionnaries_and_lists/bootstrapped:NegDictionnaries/SmallMolMWFilter_UniprotHumanProt_DrugBank_Bootstraped_NegDictionary_for_balanced_for_prot.csv")

file_ProtList = os.path.join(DATA_BASE_PATH, "dictionnaries_and_lists/list_MWFilter_UniprotHumanProt.txt")
file_MolList = os.path.join(DATA_BASE_PATH,"dictionnaries_and_lists/list_MWFilter_mol.txt")


############################################################################
############################################################################
############################################################################


def make_list_labels(dico_indice2mol_InMolKernel, list_ligand_of_prot):
	list_labels = []
	true_profile = np.zeros(len(dico_indice2mol_InMolKernel.keys()), np.int)
	for i in range(len(dico_indice2mol_InMolKernel.keys())):
		if dico_indice2mol_InMolKernel[i] in list_ligand_of_prot:
			list_labels.append(1)
			true_profile[i] = 1
		else:
			list_labels.append(-1)
			true_profile[i] = 0
	
	return list_labels, true_profile




def predict_profile_SingleFullClassifier(K_mol, list_C, list_labels):
	pred_profile_depending_on_C = {}
	score_profile_depending_on_C = {}
	for c in range(len(list_C)):
		pred_profile_depending_on_C[list_C[c]]=np.zeros(len(dico_indice2mol_InMolKernel.keys()))
		score_profile_depending_on_C[list_C[c]]=np.zeros(len(dico_indice2mol_InMolKernel.keys()))

	for i in range(len(K_mol)):
		K_train = np.delete(K_mol, i, axis=0)
		K_train = np.delete(K_train, i, axis=1)
		K_test = np.delete(K_mol[i,:], i, axis=0)
		list_train_labels = list_labels.copy()
		del list_train_labels[i]
		
		dico_train_labels = collections.Counter(list_train_labels)
		dico_weight = {1:float(len(list_train_labels))/float(2*dico_train_labels[1]) , -1:float(len(list_train_labels))/float(2*dico_train_labels[-1])}
		
		for c in range(len(list_C)):
			clf = svm.SVC(kernel='precomputed', C=list_C[c])
			clf.fit(K_train, list_train_labels)
			if clf.predict(K_test).tolist()[0]==1:
				pred_profile_depending_on_C[list_C[c]][i] = 1
			else:
				pred_profile_depending_on_C[list_C[c]][i] = 0
			score_profile_depending_on_C[list_C[c]][i] = clf.decision_function(K_test).tolist()[0]
	for c in range(len(list_C)):
		max_ = np.max(score_profile_depending_on_C[list_C[c]])
		min_ = np.min(score_profile_depending_on_C[list_C[c]])
		for i in range(len(K_mol)):		
			score_profile_depending_on_C[list_C[c]][i] = (score_profile_depending_on_C[list_C[c]][i]-min_)/(max_-min_)
			#if score_profile_depending_on_C[list_C[c]][i]<0 or score_profile_depending_on_C[list_C[c]][i]>1:
			#	print(score_profile_depending_on_C[list_C[c]][i])
	return pred_profile_depending_on_C, score_profile_depending_on_C







def mix_true_in_pred(true_profile, pred_profile, score_profile, list_C):
	TrueAndPred_profile = {}
	TrueAndScore_profile = {}
	for c in range(len(list_C)):
		TrueAndPred_profile[list_C[c]] = np.zeros(len(true_profile))
		TrueAndScore_profile[list_C[c]] = np.zeros(len(true_profile))
		for ind in range(len(true_profile)):
			if true_profile[ind]==1:
				TrueAndPred_profile[list_C[c]][ind]=1
				TrueAndScore_profile[list_C[c]][ind]=1
			else:
				TrueAndPred_profile[list_C[c]][ind]=pred_profile[list_C[c]][ind]
				TrueAndScore_profile[list_C[c]][ind]=score_profile[list_C[c]][ind]
				
	
	return TrueAndPred_profile, TrueAndScore_profile




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
	nb_mol = int(row[1])
	j=0
	while j<nb_mol:
		dico_target_of_mol[row[2+j]][2].append(row[0])
		dico_ligand_of_prot[row[0]][2].append(row[2+j])
		dico_labels_per_couple[row[2+j]+row[0]] = -1
		list_neg_couples.append((row[0],row[2+j]))
		j+=1
del reader

f_in = open(file_NegDic_balanced, 'r')
reader = csv.reader(f_in, delimiter='\t')
for row in reader:
	nb_mol = int(row[1])
	j=0
	while j<nb_mol:
		dico_target_of_mol[row[2+j]][3].append(row[0])
		dico_ligand_of_prot[row[0]][3].append(row[2+j])
		dico_labels_per_couple[row[2+j]+row[0]] = -1
		j+=1
del reader
f_in.close()

####

#### on charge les kernels et dico des kernels
with open(file_MolKernel, 'rb') as fichier:
	pickler = pickle.Unpickler(fichier)
	K_mol = pickler.load()
with open(file_DicoMolKernel_indice2, 'rb') as fichier:
	pickler = pickle.Unpickler(fichier)
	dico_indice2mol_InMolKernel = pickler.load()
with open(file_DicoMolKernel_2indice, 'rb') as fichier:
	pickler = pickle.Unpickler(fichier)
	dico_mol2indice_InMolKernel = pickler.load()
####

############################################################################
############################################################################
############################################################################

list_C = [0.01, 0.1, 0.5, 1, 5, 10, 100]
list_C = [0.5,1]

dico_pred_profile_per_prot = {}
dico_score_profile_per_prot = {}
dico_true_profile_per_prot = {}
dico_TrueAndScore_profile_per_prot = {}
dico_TrueAndPred_profile_per_prot = {}

############################################################################
############################################################################
############################################################################

list_prot_of_dataset = list_prot_of_dataset.copy()
granularity = math.ceil(float(len(list_prot_of_dataset))/float(120))
list_prot_of_dataset = list_prot_of_dataset[int(sys.argv[1])*granularity:min((int(sys.argv[1])+1)*granularity,len(list_prot_of_dataset))].copy()

for prot in list_prot_of_dataset:
	print(prot)
	if len(dico_ligand_of_prot[prot][0])!=1:
		list_labels, dico_true_profile_per_prot[prot] = make_list_labels(dico_indice2mol_InMolKernel, dico_ligand_of_prot[prot][0])
		dico_pred_profile_per_prot[prot], dico_score_profile_per_prot[prot] = predict_profile_SingleFullClassifier(K_mol, list_C, list_labels)
		dico_TrueAndPred_profile_per_prot[prot], dico_TrueAndScore_profile_per_prot[prot] = mix_true_in_pred(dico_true_profile_per_prot[prot], dico_pred_profile_per_prot[prot], dico_score_profile_per_prot[prot], list_C)
		print(dico_TrueAndPred_profile_per_prot)
		
############################################################################
############################################################################
############################################################################



with open(os.path.join(OUTPUT_BASE_PATH, "dico_pred_profile_per_prot_"+sys.argv[1]+".data"), 'wb') as fichier:
	pickler = pickle.Pickler(fichier)
	pickler.dump(dico_pred_profile_per_prot)
with open(os.path.join(OUTPUT_BASE_PATH, "dico_score_profile_per_prot_"+sys.argv[1]+".data"), 'wb') as fichier:
	pickler = pickle.Pickler(fichier)
	pickler.dump(dico_score_profile_per_prot)
with open(os.path.join(OUTPUT_BASE_PATH, "dico_true_profile_per_prot_"+sys.argv[1]+".data"), 'wb') as fichier:
	pickler = pickle.Pickler(fichier)
	pickler.dump(dico_true_profile_per_prot)
with open(os.path.join(OUTPUT_BASE_PATH, "dico_TrueAndPred_profile_per_prot_"+sys.argv[1]+".data"), 'wb') as fichier:
	pickler = pickle.Pickler(fichier)
	pickler.dump(dico_TrueAndPred_profile_per_prot)
with open(os.path.join(OUTPUT_BASE_PATH, "dico_TrueAndScore_profile_per_prot_"+sys.argv[1]+".data"), 'wb') as fichier:
	pickler = pickle.Pickler(fichier)
	pickler.dump(dico_TrueAndScore_profile_per_prot)
	
	
	
