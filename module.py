import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import torchaudio

class frame_mse(nn.Module):
    def __init__(self):
        super(frame_mse, self).__init__()

    def forward(self, y_pred, y_true): #(B,1,T) (B)
        y_pred = y_pred.squeeze(1) #(B,T)
        B,T = y_pred.size()
        y_true_repeat = y_true.unsqueeze(1).repeat(1,T) #(B,T)
        loss = torch.mean((y_true_repeat - y_pred.detach()) ** 2)
        return loss

def get_act_fn(act_fn):
    if act_fn == 'relu':
        return nn.ReLU()
    elif act_fn == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.3)
    elif act_fn == 'elu':
        return nn.ELU()
    elif act_fn == 'sigmoid':
        return nn.Sigmoid()
    elif act_fn == 'softplus':
        return nn.Softplus()
    else:
        raise ValueError('Invalid argument for act_fn') 

class HASANet_PLUS(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim, num_layers):
        super(HASANet_PLUS, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_linear = nn.Linear(ssl_out_dim, 257)
        self.num_layers = num_layers
        self.hl_linear  = nn.Linear(6, 257)
        
        for p in self.ssl_model.parameters():
            p.requires_grad = False
                
        self.weight = nn.Parameter(torch.ones(1,1,1,num_layers))
        self.softmax = nn.Softmax(-1)
        self.weight.requiresGrad = True  
        self.softmax.requiresGrad = True  
        
        self.blstm = nn.LSTM(input_size = 257, 
                             hidden_size = 100, 
                             num_layers = 1, 
                             dropout = 0, 
                             bidirectional = True, 
                             batch_first = True)
        self.linear1 = nn.Linear(100*2, 128, bias=True)
        self.act_fn = get_act_fn('relu')
        self.dropout = nn.Dropout(p=0.3)
        self.hasqiAtt_layer = nn.MultiheadAttention(128, num_heads=8)
        self.haspiAtt_layer = nn.MultiheadAttention(128, num_heads=8)
        
        self.hasqiframe_score = nn.Linear(128, 1)
        self.haspiframe_score = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        self.hasqiaverage_score = nn.AdaptiveAvgPool1d(1)  
        self.haspiaverage_score = nn.AdaptiveAvgPool1d(1)  
            
    def forward(self, wav, hl): # (B, audio_len) hl:(B,6)
        rep, layer_results = self.ssl_model.extract_features(wav, output_layer=self.ssl_model.cfg.encoder_layers, ret_layer_results=True)[0]
        layer_reps = [result.transpose(0, 1) for result, _ in layer_results]  #x:(T,1,D)
        x = torch.stack(layer_reps) #(num_layer, B, T, D)
        
        x = x.permute(1,2,3,0)
        x   = (x*self.softmax(self.weight)).sum(-1)
        x = self.ssl_linear(x) #(B, -1, 257)
        B, T, Freq = x.size()
        hl = hl.unsqueeze(1) #hl:(B,1,6)
        hl = self.hl_linear(hl)
        hl_repeat = hl.repeat(1,T,1)
        x_concate = x+hl_repeat #263
        out, _ = self.blstm(x_concate) #(B,-1, 2*hidden)
        out = self.dropout(self.act_fn(self.linear1(out))).transpose(0,1) #(T_length, B,  128) 
        
        hasqi, _ = self.hasqiAtt_layer(out,out,out)
        haspi, _ = self.haspiAtt_layer(out,out,out) 
        hasqi, haspi = hasqi.transpose(0,1), haspi.transpose(0,1) #(B, T_length, 128)  
        hasqi, haspi = self.hasqiframe_score(hasqi), self.haspiframe_score(haspi) #(B, T_length, 1) 
        hasqi, haspi = self.sigmoid(hasqi), self.sigmoid(haspi) #pass a sigmoid
        hasqi_fram, haspi_fram = hasqi.permute(0,2,1), haspi.permute(0,2,1) #(B, 1, T_length) 
        hasqi_avg, haspi_avg = self.hasqiaverage_score(hasqi_fram), self.haspiaverage_score(haspi_fram)  #(B,1,1)
        
        return hasqi_fram, haspi_fram, hasqi_avg.squeeze(1), haspi_avg.squeeze(1) #(B, 1, T_length) (B,1) 
    