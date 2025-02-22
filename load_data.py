import os
import numpy as np
import pandas as pd
import math
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pdb

def get_csvfile(path):
    file_list = []
    for r, _, fs in os.walk(path):
        file_list += [ os.path.join(r, f) for f in fs if f.endswith('csv')]
    file_list = np.array(sorted(file_list))
    return file_list

def train_csvfile():
    fname_list = ['../train/enhanced_final.csv', '../train/dereverb_final.csv', '../train/vocoder_final.csv']
    file_list = np.array(fname_list)
    return file_list

def valid_csvfile():
    fname_list = ['../valid/enhanced_final.csv', '../valid/dereverb_final.csv', '../valid/vocoder_final.csv']
    file_list = np.array(fname_list)
    return file_list

def test_csvfile(): 
    fname_list_unseen = ['../test/enhanced_unseen_final.csv', '../test/dereverb_unseen_final.csv', '../test/vocoder_unseen_final.csv']
    file_list_unseen= np.array(fname_list_unseen)
    return file_list_unseen
    
def load_csvfile(csvfile, args):
    list_of_dataframes = []
    for filename in csvfile:
        f = pd.read_csv(filename)
        list_of_dataframes.append(f)
        print(filename, len(f))
    merged_df = pd.concat(list_of_dataframes, ignore_index=True)
    return merged_df

class Dataset_train(Dataset):  
    def __init__(self, filepath, args):
        filepath = filepath.sample(frac=1, random_state=args.seed).reset_index(drop=True) # suffle
        self.data_list = filepath
        self.data = self.data_list['data'].tolist()
        str2array = lambda x: np.fromstring(x.replace('\n','').replace('[','').replace(']','').replace('  ',' '), sep=' ')
        self.hl = self.data_list['HL'].astype(str).apply(str2array).tolist()
        self.HASQIscore = self.data_list['HASQI'].tolist()
        self.modHASPIscore = self.data_list['HASPI'].astype('float16').tolist()
        self.hltype = self.data_list['HLType'].tolist()
                    
    def __getitem__(self, idx):
        data_name = self.data[idx]  
        Sxx_data, sample_rate = torchaudio.load(data_name)
        Sxx_data = Sxx_data.squeeze(0)
        hl = self.hl[idx]
        hasqi = self.HASQIscore[idx]
        haspi = self.modHASPIscore[idx]
        hltype = self.hltype[idx]
        return data_name, Sxx_data, torch.from_numpy(hl).float(), torch.from_numpy(np.asarray(hasqi)).float(), \
               torch.from_numpy(np.asarray(haspi)).float(), hltype
       
    def __len__(self):
        return len(self.data_list)
    
def custom_collate(batch):
    name, Sxx_data, hl, hasqi, haspi, hltype = zip(*batch)
    Sxx_data = list(Sxx_data)
    max_len = max(len(row) for row in Sxx_data)
    output_wavs = []
    for wav in Sxx_data:
        amount_to_pad = max_len - len(wav)
        padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), 'constant', 0)
        output_wavs.append(padded_wav)
    output_wavs = torch.stack(output_wavs, dim=0)
    hl  = torch.stack(hl, dim=0)
    hasqi  = torch.stack(hasqi, dim=0)
    haspi  = torch.stack(haspi, dim=0)     
    return name, output_wavs, hl, hasqi, haspi, hltype    
 
class Dataset_test(Dataset):  
    def __init__(self, filepath, args):
        self.data_list = filepath
        self.data = self.data_list['data'].to_numpy()
        str2array = lambda x: np.fromstring(x.replace('\n','').replace('[','').replace(']','').replace('  ',' '), sep=' ')
        self.hl = self.data_list['HL'].astype(str).apply(str2array).to_numpy()
        self.HASQIscore = self.data_list['HASQI'].astype('float16').to_numpy()
        self.modHASPIscore = self.data_list['HASPI'].astype('float16').tolist()
        self.hltype = self.data_list['HLType'].to_numpy()
    
    def __getitem__(self, idx):
        data_name = self.data[idx]
        Sxx_data, sample_rate = torchaudio.load(data_name)
        Sxx_data = Sxx_data.squeeze(0)
        hl = self.hl[idx]
        hasqi = self.HASQIscore[idx]
        haspi = self.modHASPIscore[idx]
        hltype = self.hltype[idx]
        return data_name, Sxx_data,\
               torch.from_numpy(hl).float(), torch.from_numpy(np.asarray(hasqi)).float(), \
               torch.from_numpy(np.asarray(haspi)).float(), hltype          
    
    def __len__(self):
        return len(self.data_list)
    
