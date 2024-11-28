import random

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

import crossover as co 
#from utils import get_fp_scores
from network import obtain_model_pred
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem.rdmolops import RenumberAtoms
import random

import selfies as sf
import yaml
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from vocabulary import SMILESTokenizer
rdBase.DisableLog('rdApp.error')

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

def delete_atom():
    choices = ['[*:1]~[D1:2]>>[*:1]', '[*:1]~[D2:2]~[*:3]>>[*:1]-[*:3]',
               '[*:1]~[D3:2](~[*;!H0:3])~[*:4]>>[*:1]-[*:3]-[*:4]',
               '[*:1]~[D4:2](~[*;!H0:3])(~[*;!H0:4])~[*:5]>>[*:1]-[*:3]-[*:4]-[*:5]',
               '[*:1]~[D4:2](~[*;!H0;!H1:3])(~[*:4])~[*:5]>>[*:1]-[*:3](-[*:4])-[*:5]']
    p = [0.25, 0.25, 0.25, 0.1875, 0.0625]

    return np.random.choice(choices, p=p)


def append_atom():
    choices = [['single', ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br'], 7 * [1.0 / 7.0]],
               ['double', ['C', 'N', 'O'], 3 * [1.0 / 3.0]],
               ['triple', ['C', 'N'], 2 * [1.0 / 2.0]]]
    p_BO = [0.60, 0.35, 0.05]

    index = np.random.choice(list(range(3)), p=p_BO)

    BO, atom_list, p = choices[index]
    new_atom = np.random.choice(atom_list, p=p)

    if BO == 'single':
        rxn_smarts = '[*;!H0:1]>>[*:1]X'.replace('X', '-' + new_atom)
    if BO == 'double':
        rxn_smarts = '[*;!H0;!H1:1]>>[*:1]X'.replace('X', '=' + new_atom)
    if BO == 'triple':
        rxn_smarts = '[*;H3:1]>>[*:1]X'.replace('X', '#' + new_atom)

    return rxn_smarts


def insert_atom():
    choices = [['single', ['C', 'N', 'O', 'S'], 4 * [1.0 / 4.0]],
               ['double', ['C', 'N'], 2 * [1.0 / 2.0]],
               ['triple', ['C'], [1.0]]]
    p_BO = [0.60, 0.35, 0.05]

    index = np.random.choice(list(range(3)), p=p_BO)

    BO, atom_list, p = choices[index]
    new_atom = np.random.choice(atom_list, p=p)

    if BO == 'single':
        rxn_smarts = '[*:1]~[*:2]>>[*:1]X[*:2]'.replace('X', new_atom)
    if BO == 'double':
        rxn_smarts = '[*;!H0:1]~[*:2]>>[*:1]=X-[*:2]'.replace('X', new_atom)
    if BO == 'triple':
        rxn_smarts = '[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#X-[*:2]'.replace('X', new_atom)

    return rxn_smarts


def change_bond_order():
    choices = ['[*:1]!-[*:2]>>[*:1]-[*:2]', '[*;!H0:1]-[*;!H0:2]>>[*:1]=[*:2]',
               '[*:1]#[*:2]>>[*:1]=[*:2]', '[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#[*:2]']
    p = [0.45, 0.45, 0.05, 0.05]

    return np.random.choice(choices, p=p)


def delete_cyclic_bond():
    return '[*:1]@[*:2]>>([*:1].[*:2])'


def add_ring():
    choices = ['[*;!r;!H0:1]~[*;!r:2]~[*;!r;!H0:3]>>[*:1]1~[*:2]~[*:3]1',
               '[*;!r;!H0:1]~[*!r:2]~[*!r:3]~[*;!r;!H0:4]>>[*:1]1~[*:2]~[*:3]~[*:4]1',
               '[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*;!r;!H0:5]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]1',
               '[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*!r:5]~[*;!r;!H0:6]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]~[*:6]1']
    p = [0.05, 0.05, 0.45, 0.45]

    return np.random.choice(choices, p=p)


def change_atom(mol):
    choices = ['#6', '#7', '#8', '#9', '#16', '#17', '#35']
    p = [0.15, 0.15, 0.14, 0.14, 0.14, 0.14, 0.14]

    X = np.random.choice(choices, p=p)
    while not mol.HasSubstructMatch(Chem.MolFromSmarts('[' + X + ']')):
        X = np.random.choice(choices, p=p)
    Y = np.random.choice(choices, p=p)
    while Y == X:
        Y = np.random.choice(choices, p=p)

    return '[X:1]>>[Y:1]'.replace('X', X).replace('Y', Y)


def mutate(mol, mutation_rate, mol_lm=None, net=None):

    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except ValueError:
        #return mol
        return None

    if mol_lm != None:
        mols = mol_lm.edit([mol])
        new_mols = []
        #for m in mols[0]:
        for m in mols:
            if m != None:# and co.mol_ok(m) and co.ring_OK(m):
                smi = [Chem.MolToSmiles(m)]
                fp_score = get_fp_scores(smi, Chem.MolToSmiles(mol))[0]
                print("original mol", Chem.MolToSmiles(mol))
                print("smiles being considered", smi)
                print("fp_scores", fp_score)
                T=0.4
                #if random.random() < (np.exp(fp_score/T) / np.exp(1/T)):
                new_mols.append(m)
        if len(new_mols) > 0:
            if net != None:
                smi_list = [Chem.MolToSmiles(m) for m in new_mols]
                preds = obtain_model_pred(smi_list, net).squeeze()
                print("NN preds", preds)
                if preds >= 0.5:
                    print("returning molecule editted using llm!")
                    return np.random.choice(new_mols)

            print("returning molecule editted using llm!")
            return np.random.choice(new_mols)

    if random.random() > mutation_rate:
        #return mol
        return None

    p = [0.15, 0.14, 0.14, 0.14, 0.14, 0.14, 0.15]
    for i in range(10):
        rxn_smarts_list = 7 * ['']
        rxn_smarts_list[0] = insert_atom()
        rxn_smarts_list[1] = change_bond_order()
        rxn_smarts_list[2] = delete_cyclic_bond()
        rxn_smarts_list[3] = add_ring()
        rxn_smarts_list[4] = delete_atom()
        rxn_smarts_list[5] = change_atom(mol)
        rxn_smarts_list[6] = append_atom()
        rxn_smarts = np.random.choice(rxn_smarts_list, p=p)

        # print 'mutation',rxn_smarts

        rxn = AllChem.ReactionFromSmarts(rxn_smarts)

        new_mol_trial = rxn.RunReactants((mol,))

        new_mols = []
        for m in new_mol_trial:
            m = m[0]
            # print Chem.MolToSmiles(mol),mol_ok(mol)
            if co.mol_ok(m) and co.ring_OK(m):
                new_mols.append(m)

        if len(new_mols) > 0:
            return random.choice(new_mols)

    return None