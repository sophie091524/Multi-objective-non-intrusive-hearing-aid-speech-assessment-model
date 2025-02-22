import os
import gc
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from sklearn.metrics import mean_squared_error
import scipy.stats
import module 
from torch.utils.data import DataLoader
from load_data import train_csvfile, valid_csvfile, test_csvfile,  load_csvfile, Dataset_train, Dataset_test
import pdb

def train(model, train_loader, valid_loader, device, args):   
    optimizer = getattr(optim, args.optim)(filter(lambda p: p.requires_grad, model.parameters()), lr=args.train_lr)
    frame = getattr(module, f'{args.frameloss}')().to(device)
    average = getattr(nn, f'{args.loss}')().to(device)
    
    dic_path = f'{args.train_dir}'
    if not os.path.exists(os.path.dirname(dic_path)):
        os.makedirs(os.path.dirname(dic_path))
    logger_path = f'{args.train_dir}' + f'loss.log'    
    with open(logger_path, 'w') as logger_file:
        logger_file.write('epoch,train_loss,valid_loss,QPearson,IPearson\n') 
    
    args.writer = SummaryWriter(f'{args.train_dir}logs/')
    best_valid_loss = np.inf
    best_valid_lcc = -10
    patience_counter = 0
        
    total_steps = int(len(train_loader))
    for epoch in range(1,args.train_epoch+1):
        tr_loss, tr_hasqi, tr_haspi = 0, 0, 0
        tr_hasqi_fram, tr_hasqi_avg, tr_haspi_fram, tr_haspi_avg = 0,0,0,0
        model.train() 
            
        print('Epoch: ' + str(epoch)+'/'+str(args.train_epoch))       
        pbar = tqdm(total=total_steps)
        pbar.n = 0 # move process bar back to zero
            
        for step, (name, Sxx_data, hl, HASQIscore, HASPIscore, hltype) in enumerate(train_loader):
            Sxx_data = Sxx_data.to(device)
            hl = hl.to(device) 
            HASQIscore = HASQIscore.to(device)
            HASPIscore = HASPIscore.to(device)
            optimizer.zero_grad()
            hasqi_fram, haspi_fram, hasqi_avg, haspi_avg = model(Sxx_data, hl) #(B)
            hasqi_loss_fram = frame(hasqi_fram, HASQIscore) #(B,1,T) (B)
            haspi_loss_fram = frame(haspi_fram, HASPIscore) #(B,1,T) (B)
            hasqi_loss_avg = average(hasqi_avg.squeeze(1), HASQIscore) #(B,1)->(B) (B)
            haspi_loss_avg = average(haspi_avg.squeeze(1), HASPIscore) #(B,1)->(B) (B)
        
            hasqi_loss = hasqi_loss_fram+hasqi_loss_avg
            haspi_loss = haspi_loss_fram+haspi_loss_avg
            loss = hasqi_loss + haspi_loss
            loss.backward()
            optimizer.step()
                
            tr_hasqi_fram += hasqi_loss_fram.item()
            tr_hasqi_avg += hasqi_loss_avg.item()
            tr_haspi_fram += haspi_loss_fram.item() 
            tr_haspi_avg += haspi_loss_avg.item()
            tr_hasqi+=hasqi_loss_fram.item()+hasqi_loss_avg.item()
            tr_haspi+=haspi_loss_fram.item()+haspi_loss_avg.item()
            tr_loss+=hasqi_loss_fram.item()+hasqi_loss_avg.item()+\
                     haspi_loss_fram.item()+haspi_loss_avg.item()
              
            pbar.update(1)
            
        pbar.close()  
        epoch_train_loss = tr_loss/len(train_loader) 
        epoch_hasqi = tr_hasqi/len(train_loader) 
        epoch_haspi = tr_haspi/len(train_loader) 
        epoch_hasqi_fram, epoch_hasqi_avg = tr_hasqi_fram/len(train_loader),tr_hasqi_avg/len(train_loader)
        epoch_haspi_fram, epoch_haspi_avg = tr_haspi_fram/len(train_loader),tr_haspi_avg/len(train_loader)
        print(f'train:{len(train_loader)}')
        print(f'Epoch:{epoch}')
        print(f'Train loss:{epoch_train_loss:.3f},HASQI:{epoch_hasqi:.3f},HASPI:{epoch_haspi:.3f}')
        epoch_valid_loss, epoch_valid_Q, epoch_valid_I, PearsonQ, PearsonI = evaluate(model, valid_loader, epoch, device, args)
        print(f'Validation loss:{epoch_valid_loss:.3f},HASQI:{epoch_valid_Q:.3f},HASPI:{epoch_valid_I:.3f},Pearson:{PearsonQ:.3f},{PearsonI:.3f}')
        
        with open(logger_path, 'a') as logger_file:
            logger_file.write(f'{epoch:03d},{epoch_train_loss:4.3e},{epoch_valid_loss:4.3e},{PearsonQ:3.3e},{PearsonI:3.3e}\n')
        
        args.writer.add_scalars(f'{args.loss}', {'train': epoch_train_loss}, epoch)
        args.writer.add_scalars(f'{args.loss}', {'valid': epoch_valid_loss}, epoch)
        args.writer.add_scalars('hasqi', {'train': epoch_hasqi}, epoch)
        args.writer.add_scalars('hasqi', {'valid': epoch_valid_Q}, epoch)
        args.writer.add_scalars('detail_hasqi', {'fram': epoch_hasqi_fram}, epoch)
        args.writer.add_scalars('detail_hasqi', {'avg': epoch_hasqi_avg}, epoch)
        args.writer.add_scalars('haspi', {'train': epoch_haspi}, epoch)
        args.writer.add_scalars('haspi', {'valid': epoch_valid_I}, epoch)
        args.writer.add_scalars('detail_haspi', {'fram': epoch_haspi_fram}, epoch)
        args.writer.add_scalars('detail_haspi', {'avg': epoch_haspi_avg}, epoch)
        args.writer.add_scalars('Pearson_cc', {'hasqi':PearsonQ}, epoch)
        args.writer.add_scalars('Pearson_cc', {'haspi':PearsonI}, epoch)
        
        if PearsonQ + PearsonI > best_valid_lcc:
            patience_counter = 0
            best_valid_lcc = PearsonQ+PearsonI
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }            
            torch.save(state, args.train_dir+f'best_loss.pth')
        else:
            patience_counter += 1
            if patience_counter == args.train_patience:
                print(f'Early stopping... no improvement at epoch:{epoch} after {args.train_patience} epochs')
                break

        epoch_train_loss, epoch_hasqi, epoch_haspi = None, None, None
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        gc.collect()
        
def evaluate(model, valid_loader, epoch, device, args): 
    frame = getattr(module, f'{args.frameloss}')().to(device)
    average = getattr(nn, f'{args.loss}')().to(device)
    model.eval()
    
    epoch_valid_loss, epoch_valid_hasqi, epoch_valid_haspi = None, None, None 
    epoch_valid_hasqi_fram, epoch_valid_hasqi_avg, epoch_valid_haspi_fram, epoch_valid_haspi_avg = None, None, None, None
    total_steps = int(len(valid_loader))
    print(f'valid:{len(valid_loader)}')
    
    output_path = args.train_dir + 'validresult.csv'
    output_file = open(output_path, 'w')
    print('data,TrueHASQI,PredictHASQI,TrueHASPI,PredictHASPI,HL,HLType', file=output_file)
    
    with torch.no_grad(): 
        valid_loss, valid_hasqi, valid_haspi = 0, 0, 0
        valid_hasqi_fram, valid_hasqi_avg, valid_haspi_fram, valid_haspi_avg = 0,0,0,0
        for step, (name, Sxx_data, hl, HASQIscore, HASPIscore, hltype) in enumerate(valid_loader):
            Sxx_data = Sxx_data.to(device)
            hl = hl.to(device)
            HASQIscore = HASQIscore.to(device)
            HASPIscore = HASPIscore.to(device)
            hasqi_fram, haspi_fram, hasqi_avg, haspi_avg = model(Sxx_data, hl) #(B)
            hasqi_loss_fram = frame(hasqi_fram, HASQIscore) #(B,1,T) (B)
            haspi_loss_fram = frame(haspi_fram, HASPIscore) #(B,1,T) (B)
            hasqi_loss_avg = average(hasqi_avg.squeeze(1), HASQIscore) #(B,1)->(B) (B)
            haspi_loss_avg = average(haspi_avg.squeeze(1), HASPIscore) #(B,1)->(B) (B)
            
            valid_hasqi_fram += hasqi_loss_fram.item()
            valid_hasqi_avg += hasqi_loss_avg.item()
            valid_haspi_fram += haspi_loss_fram.item() 
            valid_haspi_avg += haspi_loss_avg.item()
            valid_hasqi+=hasqi_loss_fram.item()+hasqi_loss_avg.item()
            valid_haspi+= haspi_loss_fram.item()+haspi_loss_avg.item()
            valid_loss+=hasqi_loss_fram.item()+hasqi_loss_avg.item()+\
                         haspi_loss_fram.item()+haspi_loss_avg.item()
                    
            hasqi_score, predict_hasqi_score= HASQIscore.cpu().numpy(), hasqi_avg.squeeze(1).cpu().numpy()
            haspi_score, predict_haspi_score= HASPIscore.cpu().numpy(), haspi_avg.squeeze(1).cpu().numpy()
            
            for i,j,k,l,m,n,o in zip(name,hasqi_score,predict_hasqi_score,haspi_score, predict_haspi_score,hltype,hl.cpu().numpy()):
                print(i,j,k,l,m,n,o, sep=',',file=output_file) 
        
        output_file.close()
        epoch_valid_loss = valid_loss/len(valid_loader) 
        epoch_valid_hasqi = valid_hasqi/len(valid_loader) 
        epoch_valid_haspi = valid_haspi/len(valid_loader)     
        epoch_valid_hasqi_fram =valid_hasqi_fram/len(valid_loader)
        epoch_valid_hasqi_avg = valid_hasqi_avg/len(valid_loader)
        epoch_valid_haspi_fram = valid_haspi_fram/len(valid_loader)
        epoch_valid_haspi_avg = valid_haspi_avg/len(valid_loader)
        
        df = pd.read_csv(output_path)
        hasqi, predict_hasqi = df['TrueHASQI'].astype('float32').to_numpy(), df['PredictHASQI'].astype('float32').to_numpy()
        haspi, predict_haspi = df['TrueHASPI'].astype('float32').to_numpy(), df['PredictHASPI'].astype('float32').to_numpy()
        
        srcc1, pvalue = scipy.stats.spearmanr(hasqi,predict_hasqi)
        PearsonQ, p_value = scipy.stats.pearsonr(hasqi, predict_hasqi)
        mse1 = mean_squared_error(hasqi, predict_hasqi)
        
        srcc2, pvalue = scipy.stats.spearmanr(haspi,predict_haspi)
        PearsonI, p_value = scipy.stats.pearsonr(haspi,predict_haspi)
        mse2 = mean_squared_error(haspi,predict_haspi)
        
        plt.clf() 
        fig, axs = plt.subplots(1, 2, figsize=(8,4))
        axs[0].plot(hasqi,predict_hasqi, 'o', ms=5)
        axs[0].tick_params(labelsize=8)
        axs[0].set_xlabel("True HASQI")
        axs[0].set_ylabel("Predicted HASQI")
        axs[0].set_title(f'LCC:{PearsonQ:5f},SRCC:{srcc1:5f}')
        
        axs[1].plot(haspi,predict_haspi, 'o', ms=5)
        axs[1].tick_params(labelsize=8)
        axs[1].set_xlabel("True HASPI")
        axs[1].set_ylabel("Predicted HASPI")
        axs[1].set_title(f'LCC:{PearsonI:5f},SRCC:{srcc2:5f}')
        Pearson_cc = (PearsonQ+PearsonI)*0.5
        plt.tight_layout()
        plt.savefig(f'{args.train_dir}scatter.png')
    
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    gc.collect()  
    
    return epoch_valid_loss,epoch_valid_hasqi,epoch_valid_haspi,PearsonQ, PearsonI

def test(model, test_loader, mode, device, args): 
    frame = getattr(module, f'{args.frameloss}')().to(device)
    average = getattr(nn, f'{args.loss}')().to(device)
    model.eval()
    
    if not os.path.exists(f'{args.train_dir}'):
        os.makedirs(f'{args.train_dir}')
    
    output_path = args.train_dir + f'result_{mode}.csv'
    output_file = open(output_path, 'w')
    print('data,TrueHASQI,PredictHASQI,TrueHASPI,PredictHASPI,HL,HLType', file=output_file)
    
    test_total_loss, test_total_hasqi, test_total_haspi = None, None, None
    total_steps = int(len(test_loader))
    
    with torch.no_grad():          
        test_loss, test_hasqi, test_haspi = 0, 0, 0
        for step, (name, Sxx_data, hl, HASQIscore, HASPIscore, hltype) in enumerate(test_loader):
            Sxx_data = Sxx_data.to(device)
            hl = hl.to(device)
            HASQIscore = HASQIscore.to(device)
            HASPIscore = HASPIscore.to(device)
            hasqi_fram, haspi_fram, hasqi_avg, haspi_avg = model(Sxx_data, hl) #(B)
            hasqi_loss_fram = frame(hasqi_fram, HASQIscore) #(B,1,T) (B)
            haspi_loss_fram = frame(haspi_fram, HASPIscore) #(B,1,T) (B)
            hasqi_loss_avg = average(hasqi_avg.squeeze(1), HASQIscore) #(B,1)->(B) (B)
            haspi_loss_avg = average(haspi_avg.squeeze(1), HASPIscore) #(B,1)->(B) (B)
                
            hasqi_loss = hasqi_loss_fram+hasqi_loss_avg
            haspi_loss = haspi_loss_fram+haspi_loss_avg
            loss = hasqi_loss + haspi_loss
            test_hasqi += hasqi_loss_fram.item() + hasqi_loss_avg.item()
            test_haspi += haspi_loss_fram.item() + haspi_loss_avg.item()
            test_loss += test_hasqi+test_haspi
            
            hasqi_score, predict_hasqi_score= HASQIscore.cpu().numpy(), hasqi_avg.squeeze(1).cpu().numpy()
            haspi_score, predict_haspi_score= HASPIscore.cpu().numpy(), haspi_avg.squeeze(1).cpu().numpy()
            
            for i,j,k,l,m,n,o in zip(name,hasqi_score,predict_hasqi_score,haspi_score, predict_haspi_score,hltype,hl.cpu().numpy()):
                print(i,j,k,l,m,n,o, sep=',',file=output_file)      
        output_file.close() 
        
        test_total_loss = test_loss/len(test_loader) 
        test_total_hasqi = test_hasqi/len(test_loader) 
        test_total_haspi = test_haspi/len(test_loader)
        # calculate Pearson_cc and draw figure
        df = pd.read_csv(output_path)
        # SRCC is the Spearman Rank Correlation Coefficent
        # LCC is the normal Linear Correlation Coefficient
        hasqi = df['TrueHASQI'].astype('float32').to_numpy()
        predict_hasqi = df['PredictHASQI'].astype('float32').to_numpy()
        srcc1, pvalue = scipy.stats.spearmanr(hasqi, predict_hasqi)
        PearsonQ, p_value = scipy.stats.pearsonr(hasqi, predict_hasqi)
        mse1 = mean_squared_error(hasqi, predict_hasqi)
        
        haspi = df['TrueHASPI'].astype('float32').to_numpy()
        predict_haspi = df['PredictHASPI'].astype('float32').to_numpy()
        srcc2, pvalue = scipy.stats.spearmanr(haspi, predict_haspi)
        PearsonI, p_value = scipy.stats.pearsonr(haspi, predict_haspi)
        mse2 = mean_squared_error(haspi, predict_haspi)
        plt.clf() # clear the fig before we draw
        fig, axs = plt.subplots(1, 2, figsize=(8,4))
        axs[0].plot(hasqi,predict_hasqi, 'o', ms=5)
        axs[0].tick_params(labelsize=8)
        axs[0].set_xlabel("True HASQI")
        axs[0].set_ylabel("Predicted HASQI")
        axs[0].set_title(f'LCC:{PearsonQ:3f},SRCC:{srcc1:3f}')
        
        axs[1].plot(haspi,predict_haspi, 'o', ms=5)
        axs[1].tick_params(labelsize=8)
        axs[1].set_xlabel("True HASPI")
        axs[1].set_ylabel("Predicted HASPI")
        axs[1].set_title(f'LCC:{PearsonI:3f},SRCC:{srcc2:3f}')
        # we obtain the average Pearson_cc    
        Pearson_cc = (PearsonQ+PearsonI)*0.5
        # If you don't do tight_layout() you'll have weird overlaps
        plt.tight_layout()
        plt.savefig(f'{args.train_dir}scatter_{mode}.png')
        plt.close()
        
        print(f'HASQI Test Loss:{test_total_hasqi:.3f}, Pearson_cc:{PearsonQ:.3f}, SRCC:{srcc1:.3f}, MSE:{mse1:.3f}')
        print(f'HASPI Test Loss:{test_total_haspi:.3f}, Pearson_cc:{PearsonI:.3f}, SRCC:{srcc2:.3f}, MSE:{mse2:.3f}')
        
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    gc.collect()  
   
    return test_total_loss, Pearson_cc
