import os
import gc
import yaml
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import module 
from torch.utils.data import DataLoader
from load_data import train_csvfile, valid_csvfile, test_csvfile,  load_csvfile, Dataset_train, Dataset_test, custom_collate
from trainer import train, test
import pdb 
from WavLM import WavLM, WavLMConfig

def yaml_config_hook(config_file):
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)
    if "defaults" in cfg.keys():
        del cfg["defaults"]
    return cfg

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    
def main(config_path): 
    print(torch.cuda.device_count())
    # Arguments
    parser = argparse.ArgumentParser(description="HASANet_PLUS")
    config = yaml_config_hook(config_path)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    torch.cuda.empty_cache()
    
    print(f'savedic path:{args.train_dir}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device:{device}')
    setup_seed(args.seed)
    
    
    if args.Train:           
        csv_train, csv_valid = train_csvfile(), valid_csvfile()
        print(f'csv list length:{len(csv_train)}, {len(csv_valid)}')
        df_train, df_valid = load_csvfile(csv_train, args), load_csvfile(csv_valid, args)

        train_data = Dataset_train(df_train, args)
        valid_data = Dataset_test(df_valid, args)
        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, drop_last=True, 
                                  num_workers=args.num_workers, shuffle=False, pin_memory=True, collate_fn=custom_collate)
        valid_loader = DataLoader(dataset=valid_data, batch_size=2, drop_last=True,
                                  num_workers=args.num_workers, shuffle=True, pin_memory=True, collate_fn=custom_collate) 
        
        cp_path = args.SSL_pth
        ssl_model_type = cp_path.split('/')[-1]
        checkpoint = torch.load(cp_path)
        cfg = WavLMConfig(checkpoint['cfg'])
        ssl_model = WavLM(cfg)
        ssl_model.load_state_dict(checkpoint['model'])
        SSL_OUT_DIM = 1024
        WEIGHT_LAYER = 25
        
        net = getattr(module, args.model)(ssl_model, SSL_OUT_DIM, WEIGHT_LAYER)    
        net = torch.nn.DataParallel(net, device_ids=[0,1])
        net = net.to(device) 
        train(net, train_loader, valid_loader, device, args)
        
    else:
        csv_file = test_csvfile()
        df = load_csvfile(csv_file, args)
        data = Dataset_test(df, args)
        data_loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
        cp_path = args.SSL_pth
        ssl_model_type = cp_path.split('/')[-1]
        checkpoint = torch.load(cp_path)
        cfg = WavLMConfig(checkpoint['cfg'])
        ssl_model = WavLM(cfg)
        ssl_model.load_state_dict(checkpoint['model'])
        SSL_OUT_DIM = 1024
        WEIGHT_LAYER = 25
        
        net = getattr(module, args.model)(ssl_model, SSL_OUT_DIM, WEIGHT_LAYER)  
        net = torch.nn.DataParallel(net, device_ids=[0])
        net = net.to(device)
        print(f'Loading the model from training dic:{args.train_dir}best_loss.pth')   
        ckpt = torch.load(f'{args.train_dir}best_loss.pth')['model']
        net.load_state_dict(ckpt) 
        
        test(net, data_loader, 'unseen', device, args)
        
        
if __name__ == "__main__":    
    config_path = 'config.yaml' 
    main(config_path)
        
