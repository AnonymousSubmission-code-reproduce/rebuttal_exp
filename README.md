# rebuttal_exp
Experiments in the rebuttal

Genetic GFN is implemented based on https://github.com/hyeonahkimm/genetic_gfn/tree/main/pmo/main/genetic_gfn

Augmented memory is implemented based on https://github.com/wenhao-gao/mol_opt/tree/main/main/smiles_aug_mem

To run augmented memory (example):
```bash
cd aug_mem
python run.py smiles_aug_mem --oracles qed --mol_lm BioT5 --seeds 15
```

To run genetic GFN (example):
```bash
cd genetic_gfn
python run.py genetic_gfn --task simple --wandb online --oracle qed --seed 15 --mol_lm BioT5
```
