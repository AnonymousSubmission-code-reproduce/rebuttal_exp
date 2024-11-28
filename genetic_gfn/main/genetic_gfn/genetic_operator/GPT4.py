import openai
import re
from rdkit import Chem
import main.genetic_gfn.genetic_operator.crossover as co
import main.genetic_gfn.genetic_operator.mutate as mu
openai.api_type =
openai.api_base = 
openai.api_version =
openai.api_key = 
import random
MINIMUM = 1e-10

def query_LLM(question, engine, model="gpt-4", temperature=0.0):
    message = [{"role": "system", "content": "You are a helpful agent who can answer the question based on your molecule knowledge."}]

    prompt1 = question
    message.append({"role": "user", "content": prompt1})

    params = {
        "engine": engine,
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
                'qed': 'I have two molecules. The QED score measures the drug-likeness of the molecule. Here are the molecules: \n\n',
                'jnk3': 'I have two molecules. The JNK3 score measures a molecular\'s biological activity against JNK3. Here are the molecules: \n\n',
                'drd2': 'I have two molecules. The DRD2 score measures a molecule\'s biological activity against a biological target named the dopamine type 2 receptor (DRD2). Here are the molecules: \n\n',
                'gsk3b': 'I have two molecules. The GSK3$\beta$ score measures a molecular\'s biological activity against Glycogen Synthase Kinase 3 Beta. Here are the molecules: \n\n',
                'isomers_C9H10N2O2PF2Cl': 'I have two molecules. The isomer score measures a molecule\'s similarity in terms of atom counter to C9H10N2O2PF2Cl. Here are the molecules: \n\n',
                'perindopril_mpo': 'I have two molecules. The perindopril multiproperty objective score measures the geometric means of several scores, including the molecule\'s Tanimoto similarity to perindopril and number of aromatic rings. Here are the molecules: \n\n',
                'sitagliptin_mpo': 'I have two molecules. The sitagliptin rediscovery score measures the geometric means of several scores, including the molecule\'s Tanimoto similarity to sitagliptin, TPSA score, LogP score and isomer score with C16H15F6N5O. Here are the molecules: \n\n',
                'ranolazine_mpo': 'I have two molecules. The ranolazine multiproperty objective score measures the geometric means of several scores, including the molecule\'s Tanimoto similarity to ranolazine, TPSA score LogP score and number of fluorine atoms. Here are the molecules: \n\n',
                'thiothixene_rediscovery': 'I have two molecules. The thiothixene rediscovery measures a molecule\'s Tanimoto similarity with thiothixene\'s SMILES to check whether it could be rediscovered. Here are the molecules: \n\n',
                'mestranol_similarity': 'I have two molecules. The mestranol similarity score measures a molecule\'s Tanimoto similarity with Mestranol. Here are the molecules: \n\n',
                'albuterol_similarity': 'I have two molecules. The albuterol similarity score measures a molecule\'s Tanimoto similarity with Albuterol. Here are the molecules: \n\n',
                'amlodipine_mpo': 'I have two molecules. The amlodipine multiproperty objective score measures the geometric means of several scores, including the molecule\'s Tanimoto similarity to amlodipine and number rings. Here are the molecules: \n\n',
                'celecoxib_rediscovery': 'I have two molecules. The celecoxib rediscovery score measures a molecule\'s Tanimoto similarity with celecoxib\'s SMILES to check whether it could be rediscovered. Here are the molecules: \n\n',
                'fexofenadine_mpo': 'I have two molecules. The fexofenadine multiproperty objective score measures the geometric means of several scores, including the molecule\'s Tanimoto similarity to fexofenadine, TPSA score and LogP score. Here are the molecules: \n\n',
                'isomers_c7h8n2o2': 'I have two molecules. The isomer score measures measures a molecule\'s similarity in terms of atom counter to C7H8N2O2. Here are the molecules: \n\n',
                'troglitazone_rediscovery': 'I have two molecules. The troglitazone rediscovery score measures a molecule\'s Tanimoto similarity with troglitazone\'s SMILES to check whether it could be rediscovered. Here are the molecules: \n\n',
                'zaleplon_mpo': 'I have two molecules. The zaleplon multiproperty objective score measures the geometric means of several scores, including the molecule\'s Tanimoto similarity to zaleplon and the molecule\'s similarity in terms of atom counter to C19H17N3O2. Here are the molecules: \n\n',
                'median1': 'I have two molecules. The median multiproperty objective score measures the average score of the molecule\'s Tanimoto similarity to Camphor and Menthol. Here are the molecules: \n\n'
                }
        self.task2objective = {
                'qed': 'Please propose a new molecule that has a higher QED score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'jnk3': 'Please propose a new molecule that has a higher JNK3 score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'drd2': 'Please propose a new molecule that has a higher DRD2 score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'gsk3b': 'Please propose a new molecule that has a higher GSK3$\beta$ score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'isomers_C9H10N2O2PF2Cl': 'Please propose a new molecule that has a higher isomer score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'perindopril_mpo': 'Please propose a new molecule that has a higher perindopril multiproperty objective score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'sitagliptin_mpo': 'Please propose a new molecule that has a higher sitagliptin multiproperty objective score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'ranolazine_mpo': 'Please propose a new molecule that has a higher ranolazine multiproperty objective score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'thiothixene_rediscovery': 'Please propose a new molecule that has a higher thiothixene rediscovery score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'mestranol_similarity': 'Please propose a new molecule that has a higher mestranol similarity score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'albuterol_similarity': 'Please propose a new molecule that has a higher albuterol similarity score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'amlodipine_mpo': 'Please propose a new molecule that has a higher amlodipine multiproperty objective score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'celecoxib_rediscovery': 'Please propose a new molecule that has a higher celecoxib rediscovery score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'fexofenadine_mpo': 'Please propose a new molecule that has a higher fexofenadine multiproperty objective score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'isomers_c7h8n2o2': 'Please propose a new molecule that has a higher isomer score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'troglitazone_rediscovery': 'Please propose a new molecule that has a higher troglitazone rediscovery score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'zaleplon_mpo': 'Please propose a new molecule that has a higher zaleplon multiproperty objective score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'median1': 'Please propose a new molecule that has a higher median multiproperty objective score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                }
        self.requirements = """\n\nYour output should follow the format: {<<<Explaination>>>: $EXPLANATION, <<<Molecule>>>: \\box{$Molecule}}. Here are the requirements:\n
        \n\n1. $EXPLANATION should be your analysis.\n2. The $Molecule should be the smiles of your propsosed molecule.\n3. The molecule should be valid.
        """
        self.task=None
        self.API=None

    def edit(self, parent_a, parent_b, mutation_rate):
        task = self.task
        task_definition = self.task2description[task[0]]
        task_objective = self.task2objective[task[0]]

        engine = #input your engine here

        parent = []
        parent.append(parent_a)
        parent.append(parent_b)
        try:
            mol_tuple = ''
            for i in range(2):
                tu = '\n[' + Chem.MolToSmiles(parent[i]) + ']'
                mol_tuple = mol_tuple + tu
            prompt = task_definition + mol_tuple + task_objective + self.requirements
            _, r = query_LLM(prompt, engine)
            proposed_smiles = re.search(r'\\box\{(.*?)\}', r).group(1)
            proposed_smiles = sanitize_smiles(proposed_smiles)
            print(proposed_smiles)
            assert proposed_smiles != None
            new_child = Chem.MolFromSmiles(proposed_smiles)

            return new_child
        except Exception as e:
            print(f"{type(e).__name__} {e}")
            new_child = co.crossover(parent[0], parent[1])
            if new_child is not None:
                new_child = mu.mutate(new_child, mutation_rate)
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