import openai
import re
from rdkit import Chem
import crossover as co, mutation as mu
openai.api_type = 
openai.api_base = 
openai.api_version = 
openai.api_key = 
import random
MINIMUM = 1e-10

def query_LLM(question, model="gpt-4", temperature=0.0):
    message = [{"role": "system", "content": "You are a helpful agent who can answer the question based on your molecule knowledge."}]

    prompt1 = question
    message.append({"role": "user", "content": prompt1})

    params = {
        "engine": , #Please use your own engine
        "max_tokens": 2048,
        "temperature": temperature,
        "messages": message
    }

    for retry in range(3):
        try:
            response = openai.ChatCompletion.create(**params)["choices"][0]["message"]["content"]
            message.append({"role": "assistant", "content": response})
            break
        except Exception as e:
            print(f"{type(e).__name__} {e}")


    print("=>")
    return message, response

class GPT4:
    def __init__(self):
        self.task2description = {
                'qed': 'I have a molecule and I want to increase its QED score. The QED score measures the drug-likeness of the molecule.\n\n',
                'jnk3': 'I have a molecule and I want to increase its JNK3 score. The JNK3 score measures a molecular\'s biological activity against JNK3.\n\n',
                'drd2': 'I have a molecule and I want to increase its DRD2 score. The DRD2 score measures a molecule\'s biological activity against a biological target named the dopamine type 2 receptor (DRD2).\n\n',
                'gsk3b': 'I have a molecule and I want to increase its GSK3$\beta$ score. The GSK3$\beta$ score measures a molecular\'s biological activity against Glycogen Synthase Kinase 3 Beta.\n\n',
                'isomers_C9H10N2O2PF2Cl': 'I have a molecule and I want to increase its isomer score. The isomer score measures a molecule\'s similarity in terms of atom counter to C9H10N2O2PF2Cl.\n\n',
                'perindopril_mpo': 'I have a molecule and I want to increase its perindopril multiproperty objective score. The perindopril multiproperty objective score measures the geometric means of several scores, including the molecule\'s Tanimoto similarity to perindopril and number of aromatic rings.\n\n',
                'sitagliptin_mpo': 'I have a molecule and I want to increase its sitagliptin multiproperty objective score. The sitagliptin rediscovery score measures the geometric means of several scores, including the molecule\'s Tanimoto similarity to sitagliptin, TPSA score, LogP score and isomer score with C16H15F6N5O.\n\n',
                'ranolazine_mpo': 'I have a molecule and I want to increase its ranolazine multiproperty objective score. The ranolazine multiproperty objective score measures the geometric means of several scores, including the molecule\'s Tanimoto similarity to ranolazine, TPSA score LogP score and number of fluorine atoms.\n\n',
                'thiothixene_rediscovery': 'I have a molecule and I want to increase its thiothixene rediscovery scrore. The thiothixene rediscovery scrore measures a molecule\'s Tanimoto similarity with thiothixene\'s SMILES to check whether it could be rediscovered.\n\n',
                'mestranol_similarity': 'I have a molecule and I want to increase its mestranol similarity score. The mestranol similarity score measures a molecule\'s Tanimoto similarity with Mestranol.\n\n',
                'albuterol_similarity': 'I have a molecule and I want to increase its albuterol similarity score. The albuterol similarity score measures a molecule\'s Tanimoto similarity with Albuterol.\n\n'
                }
        self.task2objective = {
                'qed': 'Please propose a new molecule that has a higher QED score. You can either make mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'jnk3': 'Please propose a new molecule that has a higher JNK3 score. You can either make mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'drd2': 'Please propose a new molecule that has a higher DRD2 score. You can either make mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'gsk3b': 'Please propose a new molecule that has a higher GSK3$\beta$ score. You can either make mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'isomers_C9H10N2O2PF2Cl': 'Please propose a new molecule that has a higher isomer score. You can either make mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'perindopril_mpo': 'Please propose a new molecule that has a higher perindopril multiproperty objective score. You can either make mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'sitagliptin_mpo': 'Please propose a new molecule that has a higher sitagliptin multiproperty objective score. You can either make mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'ranolazine_mpo': 'Please propose a new molecule that has a higher ranolazine multiproperty objective score. You can either make mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'thiothixene_rediscovery': 'Please propose a new molecule that has a higher thiothixene rediscovery score. You can either make mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'mestranol_similarity': 'Please propose a new molecule that has a higher mestranol similarity score. You can either make mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'albuterol_similarity': 'Please propose a new molecule that has a higher albuterol similarity score. You can either make mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n'
                }
        self.requirements = """\n\nYour output should follow the format: {<<<Explaination>>>: $EXPLANATION, <<<Molecule>>>: \\box{$Molecule}}. Here are the requirements:\n
        \n\n1. $EXPLANATION should be your analysis.\n2. The $Molecule should be the smiles of your propsosed molecule.\n3. The molecule should be valid.
        """
        self.task=None

    def edit(self, molecule_smile, mutation_rate):
        task = self.task
        task_definition = self.task2description[task[0]]
        task_objective = self.task2objective[task[0]]
        SMILES = Chem.MolToSmiles(molecule_smile)
        try:
            mol_tuple = ''
            tu = '\n' + SMILES
            mol_tuple = mol_tuple + tu
            prompt = task_definition + mol_tuple + task_objective + self.requirements
            _, r = query_LLM(prompt)
            proposed_smiles = re.search(r'\\box\{(.*?)\}', r).group(1)
            proposed_smiles = sanitize_smiles(proposed_smiles)
            print(proposed_smiles)
            assert proposed_smiles != None
            new_child = Chem.MolFromSmiles(proposed_smiles)

            return new_child
        except Exception as e:
            print(f"{type(e).__name__} {e}")

            new_child = mu.mutate(molecule_smile, mutation_rate)
            return new_child
    
def sanitize_smiles(smi):
    """
    Return a canonical smile representation of smi 

    Parameters
    ----------
    smi : str
        smile string to be canonicalized 

    Returns
    -------
    mol (rdkit.Chem.rdchem.Mol) : 
        RdKit mol object (None if invalid smile string smi)
    smi_canon (string)          : 
        Canonicalized smile representation of smi (None if invalid smile string smi)
    conversion_successful (bool): 
        True/False to indicate if conversion was  successful 
    """
    if smi == '':
        return None
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        smi_canon = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        return smi_canon
    except:
        return None