### à chaque test sur un exemple pos (leave one out), il faut enlever ce pos dans le calcul du predicteur qui sert à établir le profile
### et re calculer la ligne et la colonne correspondante au profile recalculé => recomputer la matrice de couple pour les couples touchés par le changement de profile

### à chaque test sur un exemple neg (leave one out), on a rien à faire où il faut enlever l'exemple neg du profile => rien à faire



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
############################################################################


file_MolKernel = "../kernels/kernels.data/Tanimoto_d=8_DrugBankSmallMolMWFilterHuman.data"
file_ProtKernel = "../kernels/kernels.data/allDrugBankHumanTarget_Profile_normalized_k5_threshold7.5.data"
file_DicoMolKernel_indice2 = "../kernels/dict/dico_indice2mol_InMolKernel.data"
file_DicoMolKernel_2indice = "../kernels/dict/dico_mol2indice_InMolKernel.data"
file_DicoProtKernel_indice2 = "../kernels/dict/dico_indice2prot_InProtKernel.data"
file_DicoProtKernel_2indice = "../kernels/dict/dico_prot2indice_InProtKernel.data"

file_PosDic = "../dictionnaries_and_lists/SmallMolMWFilter_UniprotHumanProt_DrugBank_Dictionary.csv"
file_NegDic = "../dictionnaries_and_lists/bootstrapped:NegDictionnaries/SmallMolMWFilter_UniprotHumanProt_DrugBank_Bootstraped_NegDictionary_for_test_for_prot.csv"
file_NegDic_balanced = "../dictionnaries_and_lists/bootstrapped:NegDictionnaries/SmallMolMWFilter_UniprotHumanProt_DrugBank_Bootstraped_NegDictionary_for_balanced_for_prot.csv"

file_ProtList = "../dictionnaries_and_lists/list_MWFilter_UniprotHumanProt.txt"
file_MolList = "../dictionnaries_and_lists/list_MWFilter_mol.txt"


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
list_C = [1]

with open("dico_profile/dico_pred_profile_per_prot.data", 'rb') as fichier:
	pickler = pickle.Unpickler(fichier)
	dico_pred_profile_per_prot = pickler.load()
with open("dico_profile/dico_score_profile_per_prot.data", 'rb') as fichier:
	pickler = pickle.Unpickler(fichier)
	dico_score_profile_per_prot = pickler.load()
with open("dico_profile/dico_true_profile_per_prot.data", 'rb') as fichier:
	pickler = pickle.Unpickler(fichier)
	dico_true_profile_per_prot= pickler.load()
with open("dico_profile/dico_TrueAndPred_profile_per_prot.data", 'rb') as fichier:
	pickler = pickle.Unpickler(fichier)
	dico_TrueAndPred_profile_per_prot = pickler.load()
with open("dico_profile/dico_TrueAndScore_profile_per_prot.data", 'rb') as fichier:
	pickler = pickle.Unpickler(fichier)
	dico_TrueAndScore_profile_per_prot = pickler.load()	


############################################################################
############################################################################
############################################################################
list_prot_of_dataset = list_prot_of_dataset.copy()

dico_indice2prot_inIPKernel = {}
dico_prot2indice_inIPKernel = {}
i1=0
for ind1 in range(len(list_prot_of_dataset)):
	if len(dico_ligand_of_prot[list_prot_of_dataset[ind1]][0])!=1:
		dico_indice2prot_inIPKernel[i1] = list_prot_of_dataset[ind1]
		dico_prot2indice_inIPKernel[list_prot_of_dataset[ind1]] = i1
		i1+=1
with open("kernels/dico_prot2indice_InIPKernel.data", 'wb') as fichier:
	pickler = pickle.Pickler(fichier)
	pickler.dump(dico_prot2indice_inIPKernel)
with open("kernels/dico_indice2prot_InIPKernel.data", 'wb') as fichier:
	pickler = pickle.Pickler(fichier)
	pickler.dump(dico_indice2prot_inIPKernel)	

############################################################################
############################################################################
if sys.argv[1]=="GIP":
	average_nbMol_per_target = 0
	nb_tot = 0
	for ind1 in range(len(list_prot_of_dataset)):
		if len(dico_ligand_of_prot[list_prot_of_dataset[ind1]][0])!=1:
			prot1 = list_prot_of_dataset[ind1]
			average_nbMol_per_target+=np.dot(np.transpose(dico_true_profile_per_prot[prot1]), dico_true_profile_per_prot[prot1])
			#print(collections.Counter((dico_true_profile_per_prot[prot1])))
			#print(np.dot(np.transpose(dico_true_profile_per_prot[prot1]), dico_true_profile_per_prot[prot1]))
			nb_tot+=1
	average_nbMol_per_target/=nb_tot
	print(average_nbMol_per_target)
	with open("kernels/average_nbMol_per_target_for_GIP.data", 'wb') as fichier:
		pickler = pickle.Pickler(fichier)
		pickler.dump(average_nbMol_per_target)

	K_prot_GIP = np.zeros((nb_tot, nb_tot))
	
	i1=0
	for ind1 in range(len(list_prot_of_dataset)):
		if len(dico_ligand_of_prot[list_prot_of_dataset[ind1]][0])!=1:
			i2=0
			for ind2 in range(ind1,len(list_prot_of_dataset)):
				if len(dico_ligand_of_prot[list_prot_of_dataset[ind2]][0])!=1:
					local = dico_true_profile_per_prot[list_prot_of_dataset[ind1]]-dico_true_profile_per_prot[list_prot_of_dataset[ind2]]
					K_prot_GIP[i1,i2] = np.exp((-1/average_nbMol_per_target)*np.dot(np.transpose(local),local))
					K_prot_GIP[i2,i1] = K_prot_GIP[i1,i2]
					i2+=1
			i1+=1

	with open("kernels/K_prot_GIP.data", 'wb') as fichier:
		pickler = pickle.Pickler(fichier)
		pickler.dump(K_prot_GIP)
############################################################################
############################################################################
elif sys.argv[1]=="GPIP_score":

	average_nbMol_per_target = {}
	for c in range(len(list_C)):
		average_nbMol_per_target[list_C[c]] = 0
		nb_tot = 0
		for ind1 in range(len(list_prot_of_dataset)):
			if len(dico_ligand_of_prot[list_prot_of_dataset[ind1]][0])!=1:
				prot1 = list_prot_of_dataset[ind1]
				print(dico_score_profile_per_prot[prot1])
				average_nbMol_per_target[list_C[c]]+=np.dot(np.transpose(dico_score_profile_per_prot[prot1][list_C[c]]),dico_score_profile_per_prot[prot1][list_C[c]])
				#print(np.dot(np.transpose(dico_score_profile_per_prot[prot1][list_C[c]]),dico_score_profile_per_prot[prot1][list_C[c]]))
				nb_tot+=1
		average_nbMol_per_target[list_C[c]]/=nb_tot
		print(average_nbMol_per_target[list_C[c]])
		with open("kernels/average_nbMol_per_target_for_GPIP_score_C="+str(list_C[c])+".data", 'wb') as fichier:
			pickler = pickle.Pickler(fichier)
			pickler.dump(average_nbMol_per_target[list_C[c]])

		K_prot_GPIP_score = np.zeros((nb_tot, nb_tot))
	
		i1=0
		for ind1 in range(len(list_prot_of_dataset)):
			if len(dico_ligand_of_prot[list_prot_of_dataset[ind1]][0])!=1:
				i2=0
				for ind2 in range(ind1,len(list_prot_of_dataset)):
					if len(dico_ligand_of_prot[list_prot_of_dataset[ind2]][0])!=1:
						local = dico_score_profile_per_prot[list_prot_of_dataset[ind1]][list_C[c]]-dico_score_profile_per_prot[list_prot_of_dataset[ind2]][list_C[c]]
						K_prot_GPIP_score[i1,i2] = np.exp((-1/average_nbMol_per_target[list_C[c]])*np.dot(np.transpose(local),local))
						K_prot_GPIP_score[i2,i1] = K_prot_GPIP_score[i1,i2]
						i2+=1
				i1+=1
		
		with open("kernels/K_prot_GPIP_score_C="+str(list_C[c])+".data", 'wb') as fichier:
			pickler = pickle.Pickler(fichier)
			pickler.dump(K_prot_GPIP_score)	
############################################################################
############################################################################
elif sys.argv[1]=="GPIP_TrueAndScore":

	average_nbMol_per_target = {}
	for c in range(len(list_C)):
		average_nbMol_per_target[list_C[c]] = 0
		nb_tot = 0
		for ind1 in range(len(list_prot_of_dataset)):
			if len(dico_ligand_of_prot[list_prot_of_dataset[ind1]][0])!=1:
				prot1 = list_prot_of_dataset[ind1]
				average_nbMol_per_target[list_C[c]]+=np.dot(np.transpose(dico_TrueAndScore_profile_per_prot[prot1][list_C[c]]),dico_TrueAndScore_profile_per_prot[prot1][list_C[c]])
				#print(np.dot(np.transpose(dico_TrueAndScore_profile_per_prot[prot1][list_C[c]]),dico_TrueAndScore_profile_per_prot[prot1][list_C[c]]))
				nb_tot+=1	
		average_nbMol_per_target[list_C[c]]/=nb_tot
		print(average_nbMol_per_target[list_C[c]])
		with open("kernels/average_nbMol_per_target_for_GPIP_TrueAndScore_C="+str(list_C[c])+".data", 'wb') as fichier:
			pickler = pickle.Pickler(fichier)
			pickler.dump(average_nbMol_per_target[list_C[c]])

		K_prot_GPIP_TrueAndScore = np.zeros((nb_tot, nb_tot))
	
		i1=0
		for ind1 in range(len(list_prot_of_dataset)):
			if len(dico_ligand_of_prot[list_prot_of_dataset[ind1]][0])!=1:
				i2=0
				for ind2 in range(ind1,len(list_prot_of_dataset)):
					if len(dico_ligand_of_prot[list_prot_of_dataset[ind2]][0])!=1:
						local = dico_TrueAndScore_profile_per_prot[list_prot_of_dataset[ind1]][list_C[c]]-dico_TrueAndScore_profile_per_prot[list_prot_of_dataset[ind2]][list_C[c]]
						K_prot_GPIP_TrueAndScore[i1,i2] = np.exp((-1/average_nbMol_per_target[list_C[c]])*np.dot(np.transpose(local),local))
						K_prot_GPIP_TrueAndScore[i2,i1] = K_prot_GPIP_TrueAndScore[i1,i2]
						i2+=1
				i1+=1
		
		with open("kernels/K_prot_GPIP_TrueAndScore_C="+str(list_C[c])+".data", 'wb') as fichier:
			pickler = pickle.Pickler(fichier)
			pickler.dump(K_prot_GPIP_TrueAndScore)		
############################################################################
############################################################################
elif sys.argv[1]=="GPIP_pred":

	average_nbMol_per_target = {}
	for c in range(len(list_C)):
		average_nbMol_per_target[list_C[c]] = 0
		nb_tot = 0
		for ind1 in range(len(list_prot_of_dataset)):
			if len(dico_ligand_of_prot[list_prot_of_dataset[ind1]][0])!=1:
				prot1 = list_prot_of_dataset[ind1]
				average_nbMol_per_target[list_C[c]]+=np.dot(np.transpose(dico_pred_profile_per_prot[prot1][list_C[c]]),dico_pred_profile_per_prot[prot1][list_C[c]])
				#print(collections.Counter(dico_pred_profile_per_prot[prot1][list_C[c]]))
				#print(np.dot(np.transpose(dico_pred_profile_per_prot[prot1][list_C[c]]),dico_pred_profile_per_prot[prot1][list_C[c]]))
				nb_tot+=1
		average_nbMol_per_target[list_C[c]]/=nb_tot
		print(average_nbMol_per_target[list_C[c]])
		with open("kernels/average_nbMol_per_target_for_GPIP_pred_C="+str(list_C[c])+".data", 'wb') as fichier:
			pickler = pickle.Pickler(fichier)
			pickler.dump(average_nbMol_per_target[list_C[c]])
		
		K_prot_GPIP_pred = np.zeros((nb_tot, nb_tot))
	
		i1=0
		for ind1 in range(len(list_prot_of_dataset)):
			if len(dico_ligand_of_prot[list_prot_of_dataset[ind1]][0])!=1:
				i2=0
				for ind2 in range(ind1,len(list_prot_of_dataset)):
					if len(dico_ligand_of_prot[list_prot_of_dataset[ind2]][0])!=1:
						local = dico_pred_profile_per_prot[list_prot_of_dataset[ind1]][list_C[c]]-dico_pred_profile_per_prot[list_prot_of_dataset[ind2]][list_C[c]]
						K_prot_GPIP_pred[i1,i2] = np.exp((-1/average_nbMol_per_target[list_C[c]])*np.dot(np.transpose(local),local))
						K_prot_GPIP_pred[i2,i1] = K_prot_GPIP_pred[i1,i2]
						i2+=1
				i1+=1
	
		with open("kernels/K_prot_GPIP_pred_C="+str(list_C[c])+".data", 'wb') as fichier:
			pickler = pickle.Pickler(fichier)
			pickler.dump(K_prot_GPIP_pred)
############################################################################
############################################################################			
elif sys.argv[1]=="GPIP_TrueAndPred":

	average_nbMol_per_target = {}
	for c in range(len(list_C)):
		average_nbMol_per_target[list_C[c]] = 0
		nb_tot = 0
		for ind1 in range(len(list_prot_of_dataset)):
			if len(dico_ligand_of_prot[list_prot_of_dataset[ind1]][0])!=1:
				prot1 = list_prot_of_dataset[ind1]
				average_nbMol_per_target[list_C[c]]+=np.dot(np.transpose(dico_TrueAndPred_profile_per_prot[prot1][list_C[c]]),dico_TrueAndPred_profile_per_prot[prot1][list_C[c]])
				nb_tot+=1
		average_nbMol_per_target[list_C[c]]/=nb_tot
		print(average_nbMol_per_target[list_C[c]])
		with open("kernels/average_nbMol_per_target_for_GPIP_TrueAndPred_C="+str(list_C[c])+".data", 'wb') as fichier:
			pickler = pickle.Pickler(fichier)
			pickler.dump(average_nbMol_per_target[list_C[c]])
		
		K_prot_GPIP_TrueAndPred = np.zeros((nb_tot, nb_tot))
	
		i1=0
		for ind1 in range(len(list_prot_of_dataset)):
			if len(dico_ligand_of_prot[list_prot_of_dataset[ind1]][0])!=1:
				i2=0
				for ind2 in range(ind1,len(list_prot_of_dataset)):
					if len(dico_ligand_of_prot[list_prot_of_dataset[ind2]][0])!=1:
						local = dico_TrueAndPred_profile_per_prot[list_prot_of_dataset[ind1]][list_C[c]]-dico_TrueAndPred_profile_per_prot[list_prot_of_dataset[ind2]][list_C[c]]
						K_prot_GPIP_TrueAndPred[i1,i2] = np.exp((-1/average_nbMol_per_target[list_C[c]])*np.dot(np.transpose(local),local))
						K_prot_GPIP_TrueAndPred[i2,i1] = K_prot_GPIP_TrueAndPred[i1,i2]
						i2+=1
				i1+=1
	
		with open("kernels/K_prot_GPIP_TrueAndPred_C="+str(list_C[c])+".data", 'wb') as fichier:
			pickler = pickle.Pickler(fichier)
			pickler.dump(K_prot_GPIP_TrueAndPred)		
			
			
			
