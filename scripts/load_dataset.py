import csv
import numpy as np
import os
import sys
import pickle

# BASE PATH DEFINITIONS
DATA_BASE_PATH = '.'
OUTPUT_BASE_PATH = '.'

default_FileName_PositiveInstancesDictionnary = os.path.join(DATA_BASE_PATH, "dictionnaries_and_lists/SmallMolMWFilter_UniprotHumanProt_DrugBank_Dictionary.csv")
default_FileName_ListProt = os.path.join(DATA_BASE_PATH, "dictionnaries_and_lists/list_MWFilter_UniprotHumanProt.txt")
default_FileName_ListMol = os.path.join(DATA_BASE_PATH, "dictionnaries_and_lists/list_MWFilter_mol.txt")

default_FileName_MolKernel = os.path.join(DATA_BASE_PATH, "kernels/kernels.data/Tanimoto_d=8_DrugBankSmallMolMWFilterHuman.data")
default_FileName_DicoMolKernel_indice2instance = os.path.join(DATA_BASE_PATH, "kernels/dict/dico_indice2mol_InMolKernel.data")
default_FileName_DicoMolKernel_instance2indice = os.path.join(DATA_BASE_PATH, "kernels/dict/dico_mol2indice_InMolKernel.data")

def load_dataset(FileName_PositiveInstancesDictionnary=default_FileName_PositiveInstancesDictionnary, FileName_ListProt=default_FileName_ListProt, FileName_ListMol=default_FileName_ListMol, FileName_MolKernel=default_FileName_MolKernel, FileName_DicoMolKernel_indice2instance=default_FileName_DicoMolKernel_indice2instance, FileName_DicoMolKernel_instance2indice=default_FileName_DicoMolKernel_instance2indice):
	"""
	Loading the dataset and the molecule kernel
	:param FileName_PositiveInstancesDictionnary: (string) tsv file name: each line corresponds to a molecule; 
									1rst column: gives the DrugBank ID of the molecule
									2nd column: gives the number of targets of the corresponding molecule
									other columns: gives the UniprotIDs of molecule targets (one per column)
	:param FileName_ListProt: (string) txt file name: each line gives the UniprotID of a protein of the dataset
	:param FileName_ListMol: (string) txt file name: each line gives the DrugBankID of a molecule of the dataset
	:param FileName_kernel: (string)  pickle file name: contains the molecule kernel (np.array)
	:param FileName_DicoKernel_indice2instance: (string) pickle file name: contains the dictionnary linking indices of the molecule kernel
									to its corresponding molecule ID
	:param FileName_DicoKernel_instance2indice: (string)  pickle file name: contains the dictionnary linking molecule IDs to indices 
									in the molecule kernel
	
	:return K_mol: (np.array: number of mol^2) molecule kernel
	:return DicoMolKernel_ind2mol: (dictionnary) keys are indices of the molecule kernel (i.e. integers between 0 and number_of_mol)
								    and corresponding values are DrugbankIDS of the molecule corresponding to the index
	:return DicoMolKernel_mol2ind: (dictionnary) keys are DrugbankIDs and values are their corresponding indices of the molecule kernel
	:return interaction_matrix: (np.array: number_of_mol*number_of_prot) array whose values are 1 if the molecule/protein couple
						is interaction or 0 otherwise
	"""
	##loading molecule kernel and its associated dictionnaries
	with open(FileName_MolKernel, 'rb') as fichier:
		pickler = pickle.Unpickler(fichier)
		K_mol = pickler.load().astype(np.float32)
	with open(FileName_DicoMolKernel_indice2instance, 'rb') as fichier:
		pickler = pickle.Unpickler(fichier)
		DicoMolKernel_ind2mol = pickler.load()
	with open(FileName_DicoMolKernel_instance2indice, 'rb') as fichier:
		pickler = pickle.Unpickler(fichier)
		DicoMolKernel_mol2ind = pickler.load()
	
	##charging protein list of dataset
	list_prot_of_dataset = []
	f_in = open(FileName_ListProt, 'r')
	for line in f_in:
		list_prot_of_dataset.append(line.rstrip())
	f_in.close()
	##charging list_mol_of_dataset
	list_mol_of_dataset = []
	f_in = open(FileName_ListMol, 'r')
	for line in f_in:
		list_mol_of_dataset.append(line.rstrip())
	f_in.close()

	##charging list of targets per molecule of the dataset
	#initialization
	dico_targets_per_mol = {}
	for mol in list_mol_of_dataset:
		dico_targets_per_mol[mol] = []
		
	#filling
	f_in = open(FileName_PositiveInstancesDictionnary, 'r')
	reader = csv.reader(f_in, delimiter='\t')
	for row in reader:
		nb_prot = int(row[1])
		for j in range(nb_prot):
			dico_targets_per_mol[row[0]].append(row[2+j])
	del reader
	f_in.close()
	
	##making interaction_matrix
	interaction_matrix = np.zeros((len(list_mol_of_dataset), len(list_prot_of_dataset)), dtype=np.float32)
	for i in range(len(list_mol_of_dataset)):
		list_of_targets = dico_targets_per_mol[list_mol_of_dataset[i]]
		nb=0
		for j in range(len(list_prot_of_dataset)):
			if list_prot_of_dataset[j] in list_of_targets:
				interaction_matrix[i,j] = 1
				nb+=1
		###FOR TESTING
		#if len(list_of_targets)!=nb:
		#	print("alerte")
		#	exit(1)
	
	return K_mol, DicoMolKernel_ind2mol, DicoMolKernel_mol2ind, interaction_matrix

###FOR TESTING	
#K_mol, DicoMolKernel_ind2mol, DicoMolKernel_mol2ind, interaction_matrix = load_dataset(FileName_PositiveInstancesDictionnary, FileName_ListProt, FileName_ListMol, FileName_MolKernel, FileName_DicoMolKernel_indice2instance, FileName_DicoMolKernel_instance2indice)
	
	

	
	
	
	
