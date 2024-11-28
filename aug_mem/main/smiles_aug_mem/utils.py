import torch
import numpy as np
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem.rdmolops import RenumberAtoms
from rdkit import Chem, rdBase
import random
from rdkit.Chem.rdchem import Mol
import selfies as sf
import yaml
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from vocabulary import SMILESTokenizer
import crossover as co, mutation as mu
from biot5 import BioT5
from GPT4 import GPT4
from typing import List
from tdc import Oracle
_ST = SMILESTokenizer()

MINIMUM = 1e-10

def make_mating_pool(population_mol: List[Mol], population_scores, offspring_size: int):
    """
    Given a population of RDKit Mol and their scores, sample a list of the same size
    with replacement using the population_scores as weights
    Args:
        population_mol: list of RDKit Mol
        population_scores: list of un-normalised scores given by ScoringFunction
        offspring_size: number of molecules to return
    Returns: a list of RDKit Mol (probably not unique)
    """
    # scores -> probs
    all_tuples = list(zip(population_scores, population_mol))
    population_scores = [s + MINIMUM for s in population_scores]
    sum_scores = sum(population_scores)
    population_probs = [p / sum_scores for p in population_scores]
    mating_indices = np.random.choice(len(all_tuples), p=population_probs, size=offspring_size, replace=True)
    
    mating_tuples = [all_tuples[indice] for indice in mating_indices]
    
    return mating_tuples

def reproduce(mating_tuples, mutation_rate, mol_lm=None, net=None):
    """
    Args:
        mating_pool: list of RDKit Mol
        mutation_rate: rate of mutation
    Returns:
    """
    parent = []
    parent.append(random.choice(mating_tuples))
    parent.append(random.choice(mating_tuples))

    parent_mol = [t[1] for t in parent]
    new_child = co.crossover(parent_mol[0], parent_mol[1])
    new_child_mutation = None
    if new_child is not None:
        new_child_mutation = mu.mutate(new_child, mutation_rate, mol_lm)
    return new_child, new_child_mutation

def get_randomized_smiles(task_name, lm_name, mol_lm, bin_size, smiles_list, prior) -> list:
    """takes a list of SMILES and returns a list of randomized SMILES"""
    randomized_smiles_list = []

    if lm_name == "BioT5":
        prepared_mol = [Chem.MolFromSmiles(s) for s in smiles_list]
        offspring_smi = []
        for smiles in smiles_list:
            mol = MolFromSmiles(smiles)
            if mol:
                try:
                    randomized_smiles = randomize_smiles(mol)
                    # there may be tokens in the randomized SMILES that are not in the Vocabulary
                    #check if the randomized SMILES can be encoded
                    tokens = _ST.tokenize(randomized_smiles)
                    encoded = prior.vocabulary.encode(tokens)
                    offspring_smi.append(randomized_smiles)
                except KeyError:
                    offspring_smi.append(smiles)
            else:
                offspring_smi.append(smiles)

        editted_smi = []
        for m in offspring_smi:
            try:
                test = Chem.MolFromSmiles(m)
                mo = Chem.MolToSmiles(test)
                editted_smi.append(mo)
            except:
                continue
        print(len(editted_smi))
        ii = 0
        idxs = list(range(0, len(smiles_list)))
        random.shuffle(idxs)
        while len(editted_smi) < bin_size:
            if ii == len(idxs):
                print("exiting while loop before filling up bin..........")
                break
            m = prepared_mol[idxs[ii]]
            try:
                editted_mol = mol_lm.edit([m])[0]

                if editted_mol != None:
                    s = Chem.MolToSmiles(editted_mol)
                    if s != None:
                        print("adding editted molecule!!!")
                        editted_smi.append(s)
                ii += 1
            except:
                ii += 1
        print(len(editted_smi))
        editted_smi = np.array(editted_smi).tolist()




    elif lm_name == 'GPT-4':
        prepared_mol = [Chem.MolFromSmiles(s) for s in smiles_list]

        offspring_smi = []
        for smiles in smiles_list:
            mol = MolFromSmiles(smiles)
            if mol:
                try:
                    randomized_smiles = randomize_smiles(mol)
                    # there may be tokens in the randomized SMILES that are not in the Vocabulary
                    #check if the randomized SMILES can be encoded
                    tokens = _ST.tokenize(randomized_smiles)
                    encoded = prior.vocabulary.encode(tokens)
                    offspring_smi.append(randomized_smiles)
                except KeyError:
                    offspring_smi.append(smiles)
            else:
                offspring_smi.append(smiles)

        editted_smi = []
        for m in offspring_smi:
            try:
                test = Chem.MolFromSmiles(m)
                mo = Chem.MolToSmiles(test)
                editted_smi.append(mo)
            except:
                continue
        print(len(editted_smi))
        ii = 0
        #idxs = np.argsort(population_scores)[::-1]
        idxs = list(range(0, len(smiles_list)))
        random.shuffle(idxs)
        while len(editted_smi) < bin_size:
            if ii == len(idxs):
                print("exiting while loop before filling up bin..........")
                break
            m = prepared_mol[idxs[ii]]
            try:
                editted_mol = mol_lm.edit(m, 0.067)

                if editted_mol != None:
                    s = Chem.MolToSmiles(editted_mol)
                    if s != None:
                        print("adding editted molecule!!!")
                        editted_smi.append(s)
                ii += 1
            except:
                ii += 1
        print(len(editted_smi))
        editted_smi = np.array(editted_smi).tolist()

    else:
        editted_smi = []
        for smiles in smiles_list:
            mol = MolFromSmiles(smiles)
            if mol:
                try:
                    randomized_smiles = randomize_smiles(mol)
                    # there may be tokens in the randomized SMILES that are not in the Vocabulary
                    #check if the randomized SMILES can be encoded
                    tokens = _ST.tokenize(randomized_smiles)
                    encoded = prior.vocabulary.encode(tokens)
                    editted_smi.append(randomized_smiles)
                except KeyError:
                    editted_smi.append(smiles)
            else:
                editted_smi.append(smiles)
    for idx, smiles in enumerate(smiles_list):
        

        try:
            #randomized_smiles = randomize_smiles(mol)
            # there may be tokens in the randomized SMILES that are not in the Vocabulary
            #check if the randomized SMILES can be encoded
            if idx < len(editted_smi):
                randomized_smiles = editted_smi[idx]
                tokens = _ST.tokenize(randomized_smiles)
                encoded = prior.vocabulary.encode(tokens)
                randomized_smiles_list.append(randomized_smiles)
            else:
                randomized_smiles_list.append(smiles)
        except KeyError:
            randomized_smiles_list.append(smiles)


    return randomized_smiles_list


def randomize_smiles(mol) -> str:
    """
    Returns a randomized SMILES given an RDKit Mol object.
    :param mol: An RDKit Mol object
    :return : A random SMILES string of the same molecule or None if the molecule is invalid.
    from reinvent-chemistry
    """
    new_atom_order = list(range(mol.GetNumHeavyAtoms()))
    # reinvent-chemistry uses random.shuffle
    # use np.random.shuffle for reproducibility since PMO fixes the np seed
    np.random.shuffle(new_atom_order)
    random_mol = RenumberAtoms(mol, newOrder=new_atom_order)
    return MolToSmiles(random_mol, canonical=False, isomericSmiles=False)


def to_tensor(tensor):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)


def get_fp_scores(smiles_back, target_smi):
    """
    Given a list of SMILES (smiles_back), tanimoto similarities are calculated 
    (using Morgan fingerprints) to SMILES (target_smi). 
    Parameters
    ----------
    smiles_back : (list)
        List of valid SMILE strings. 
    target_smi : (str)
        Valid SMILES string. 
    Returns
    -------
    smiles_back_scores : (list of floats)
        List of fingerprint similarity scores of each smiles in input list. 
    """
    smiles_back_scores = []
    target = MolFromSmiles(target_smi)
    fp_target = AllChem.GetMorganFingerprint(target, 2)
    for item in smiles_back:
        mol = MolFromSmiles(item)
        fp_mol = AllChem.GetMorganFingerprint(mol, 2)
        score = TanimotoSimilarity(fp_mol, fp_target)
        smiles_back_scores.append(score)
    return smiles_back_scores