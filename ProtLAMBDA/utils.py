import os 

def load_fastas(fasta_dir):
    """
    Load all FASTA files from a directory and return a dictionary of sequences.
    
    Args:
        fasta_dir (str): Path to the directory containing FASTA files.

    Returns:
        list of list: A list of lists, where each inner list contains the sequence ID and the corresponding sequence.
    """
    sequences = []
    for filename in os.listdir(fasta_dir):
        if filename.endswith(".fasta") or filename.endswith(".fa"):
            with open(os.path.join(fasta_dir, filename), "r") as f:
                seq_id = filename.rsplit('.', 1)[0]  # Use the filename (without extension) as the sequence ID
                seq = ''
                for line in f:
                    if line.startswith(">"):
                        continue  # Skip header lines
                    else:
                        seq += line.strip()
                if seq:                         
                    sequences.append([seq_id, seq])
    return sequences



import pickle
import numpy as np
from .openfold.model.openfold.feats import atom14_to_atom37
from .openfold.model.dataset.af2_util import protein
import torch

def prepare_pdb_file(struc_out, af2_aa_tokens, plddt, atom37_conversion_data_pth='ProtLAMBDA/openfold/model/dataset/atom37_conversion_data.p'):

    atom37_conversion_data = pickle.load(open(atom37_conversion_data_pth, 'rb'))

    plddt = np.argmax(plddt,axis=-1)

    pdb_dt_hlpr = {'residue_index':[],'aatype':[],'residx_atom37_to_atom14':[],'atom37_atom_exists':[]}
    for i3 in range(len(af2_aa_tokens[0])):
        if(af2_aa_tokens[0][i3]<20):
            pdb_dt_hlpr['residue_index'].append(i3)
            pdb_dt_hlpr['aatype'].append(af2_aa_tokens[0][i3])
            pdb_dt_hlpr['residx_atom37_to_atom14'].append(atom37_conversion_data['residx_atom37_to_atom14'][af2_aa_tokens[0][i3].item()])
            pdb_dt_hlpr['atom37_atom_exists'].append(atom37_conversion_data['atom37_atom_exists'][af2_aa_tokens[0][i3].item()])
    pdb_dt_hlpr['residue_index'] = torch.Tensor(pdb_dt_hlpr['residue_index']).long()
    pdb_dt_hlpr['aatype'] = torch.Tensor(pdb_dt_hlpr['aatype']).long()
    pdb_dt_hlpr['residx_atom37_to_atom14'] = torch.stack(pdb_dt_hlpr['residx_atom37_to_atom14'],dim=0)
    pdb_dt_hlpr['atom37_atom_exists'] = torch.stack(pdb_dt_hlpr['atom37_atom_exists'],dim=0).long()


    pred_prtn = protein.Protein(
            atom_positions=np.array(atom14_to_atom37(struc_out['positions'][-1,0,pdb_dt_hlpr['residue_index'],:,:], pdb_dt_hlpr)),
            atom_mask=np.array(pdb_dt_hlpr['atom37_atom_exists']),
            aatype=np.array(pdb_dt_hlpr['aatype']),
            residue_index=np.array(pdb_dt_hlpr['residue_index']+1), #+idxs[i]
            b_factors=np.repeat(plddt[:, None], 37, axis=1)
    )
            #b_factors=np.array(np.zeros((len(pdb_dt_hlpr['aatype']),37)))
    #)

    return protein.to_pdb(pred_prtn)

