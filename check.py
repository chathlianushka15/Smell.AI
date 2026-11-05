from rdkit import Chem
import sklearn, pandas, streamlit

mol = Chem.MolFromSmiles("CC(=O)OCC")   # ethyl acetate, a fruity molecule
print("Atoms in the molecule:", mol.GetNumAtoms())
print("All libraries working!")