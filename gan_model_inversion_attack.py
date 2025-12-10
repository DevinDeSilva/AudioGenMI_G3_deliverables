# %%
import os
from random import random
import torch
import torch.nn as nn
import torch

import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.autograd as autograd
import dycomutils
from sentence_transformers import SentenceTransformer
import pandas as pd
import torch.nn.functional as F
import math
import time

import os
import numpy as np
import matplotlib.pyplot as plt
import neptune
from dotenv import load_dotenv
import dycomutils
plt.rcParams.update({'font.size': 24})

import warnings
warnings.filterwarnings('ignore')

# Torch
import torch
import torchaudio
#from torch.utils import tensorboard
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
from typing import Tuple, List
from tqdm import tqdm

# Load Configs
load_dotenv()
cuda = True if torch.cuda.is_available() else False

ROOT_DIR = "tacotron2"
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple, List

# --- Helper Function for Input Initialization ---

# run = neptune.init_run(
#     project="Botz/Audio-MI",
#     name="sinc-net-MI-attack",
#     api_token=os.getenv("NEPTUNE_API_TOKEN")
# )

# %%
def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


def sinc(band,t_right):
    y_right= torch.sin(2*math.pi*band*t_right)/(2*math.pi*band*t_right)
    y_left= flip(y_right,0)

    y=torch.cat([y_left,Variable(torch.ones(1)).cuda(),y_right])

    return y
    

class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):

        super(SincConv_fast,self).__init__()

        if in_channels != 1:
            #msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)
        

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin=torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
        self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size);


        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axes

 


    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz  + torch.abs(self.low_hz_)
        
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),self.min_low_hz,self.sample_rate/2)
        band=(high-low)[:,0]
        
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left=((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.n_/2))*self.window_ # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations. 
        band_pass_center = 2*band.view(-1,1)
        band_pass_right= torch.flip(band_pass_left,dims=[1])
        
        
        band_pass=torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1)

        
        band_pass = band_pass / (2*band[:,None])
        

        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1) 


        
        
class sinc_conv(nn.Module):

    def __init__(self, N_filt,Filt_dim,fs):
        super(sinc_conv,self).__init__()

        # Mel Initialization of the filterbanks
        low_freq_mel = 80
        high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, N_filt)  # Equally spaced in Mel scale
        f_cos = (700 * (10**(mel_points / 2595) - 1)) # Convert Mel to Hz
        b1=np.roll(f_cos,1)
        b2=np.roll(f_cos,-1)
        b1[0]=30
        b2[-1]=(fs/2)-100
                
        self.freq_scale=fs*1.0
        self.filt_b1 = nn.Parameter(torch.from_numpy(b1/self.freq_scale))
        self.filt_band = nn.Parameter(torch.from_numpy((b2-b1)/self.freq_scale))

        
        self.N_filt=N_filt
        self.Filt_dim=Filt_dim
        self.fs=fs
        

    def forward(self, x):
        
        filters=Variable(torch.zeros((self.N_filt,self.Filt_dim))).cuda()
        N=self.Filt_dim
        t_right=Variable(torch.linspace(1, (N-1)/2, steps=int((N-1)/2))/self.fs).cuda()
        
        
        min_freq=50.0;
        min_band=50.0;
        
        filt_beg_freq=torch.abs(self.filt_b1)+min_freq/self.freq_scale
        filt_end_freq=filt_beg_freq+(torch.abs(self.filt_band)+min_band/self.freq_scale)
       
        n=torch.linspace(0, N, steps=N)

        # Filter window (hamming)
        window=0.54-0.46*torch.cos(2*math.pi*n/N);
        window=Variable(window.float().cuda())

        
        for i in range(self.N_filt):
                        
            low_pass1 = 2*filt_beg_freq[i].float()*sinc(filt_beg_freq[i].float()*self.freq_scale,t_right)
            low_pass2 = 2*filt_end_freq[i].float()*sinc(filt_end_freq[i].float()*self.freq_scale,t_right)
            band_pass=(low_pass2-low_pass1)

            band_pass=band_pass/torch.max(band_pass)

            filters[i,:]=band_pass.cuda()*window

        out=F.conv1d(x, filters.view(self.N_filt,1,self.Filt_dim))
    
        return out
    

def act_fun(act_type):

 if act_type=="relu":
    return nn.ReLU()
            
 if act_type=="tanh":
    return nn.Tanh()
            
 if act_type=="sigmoid":
    return nn.Sigmoid()
           
 if act_type=="leaky_relu":
    return nn.LeakyReLU(0.2)
            
 if act_type=="elu":
    return nn.ELU()
                     
 if act_type=="softmax":
    return nn.LogSoftmax(dim=1)
        
 if act_type=="linear":
    return nn.LeakyReLU(1) # initializzed like this, but not used in forward!
            
            
class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class MLP(nn.Module):
    def __init__(self, options):
        super(MLP, self).__init__()
        
        self.input_dim=int(options['input_dim'])
        self.fc_lay=options['fc_lay']
        self.fc_drop=options['fc_drop']
        self.fc_use_batchnorm=options['fc_use_batchnorm']
        self.fc_use_laynorm=options['fc_use_laynorm']
        self.fc_use_laynorm_inp=options['fc_use_laynorm_inp']
        self.fc_use_batchnorm_inp=options['fc_use_batchnorm_inp']
        self.fc_act=options['fc_act']
        
       
        self.wx  = nn.ModuleList([])
        self.bn  = nn.ModuleList([])
        self.ln  = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])
       

       
        # input layer normalization
        if self.fc_use_laynorm_inp:
           self.ln0=LayerNorm(self.input_dim)
          
        # input batch normalization    
        if self.fc_use_batchnorm_inp:
           self.bn0=nn.BatchNorm1d([self.input_dim],momentum=0.05)
           
           
        self.N_fc_lay=len(self.fc_lay)
             
        current_input=self.input_dim
        
        # Initialization of hidden layers
        
        for i in range(self.N_fc_lay):
            
         # dropout
         self.drop.append(nn.Dropout(p=self.fc_drop[i]))
         
         # activation
         self.act.append(act_fun(self.fc_act[i]))
         
         
         add_bias=True
         
         # layer norm initialization
         self.ln.append(LayerNorm(self.fc_lay[i]))
         self.bn.append(nn.BatchNorm1d(self.fc_lay[i],momentum=0.05))
         
         if self.fc_use_laynorm[i] or self.fc_use_batchnorm[i]:
             add_bias=False
         
              
         # Linear operations
         self.wx.append(nn.Linear(current_input, self.fc_lay[i],bias=add_bias))
         
         # weight initialization
         self.wx[i].weight = torch.nn.Parameter(torch.Tensor(self.fc_lay[i],current_input).uniform_(-np.sqrt(0.01/(current_input+self.fc_lay[i])),np.sqrt(0.01/(current_input+self.fc_lay[i]))))
         self.wx[i].bias = torch.nn.Parameter(torch.zeros(self.fc_lay[i]))
         
         current_input=self.fc_lay[i]
         
         
    def forward(self, x):
        
      # Applying Layer/Batch Norm
      if bool(self.fc_use_laynorm_inp):
        x=self.ln0((x))
        
      if bool(self.fc_use_batchnorm_inp):
        x=self.bn0((x))
        
      for i in range(self.N_fc_lay):

        if self.fc_act[i]!='linear':
            
          if self.fc_use_laynorm[i]:
           x = self.drop[i](self.act[i](self.ln[i](self.wx[i](x))))
          
          if self.fc_use_batchnorm[i]:
           x = self.drop[i](self.act[i](self.bn[i](self.wx[i](x))))
          
          if self.fc_use_batchnorm[i]==False and self.fc_use_laynorm[i]==False:
           x = self.drop[i](self.act[i](self.wx[i](x)))
           
        else:
          if self.fc_use_laynorm[i]:
           x = self.drop[i](self.ln[i](self.wx[i](x)))
          
          if self.fc_use_batchnorm[i]:
           x = self.drop[i](self.bn[i](self.wx[i](x)))
          
          if self.fc_use_batchnorm[i]==False and self.fc_use_laynorm[i]==False:
           x = self.drop[i](self.wx[i](x)) 
          
      return x



class SincNet(nn.Module):
    
    def __init__(self,options):
       super(SincNet,self).__init__()
    
       self.cnn_N_filt=options['cnn_N_filt']
       self.cnn_len_filt=options['cnn_len_filt']
       self.cnn_max_pool_len=options['cnn_max_pool_len']
       
       
       self.cnn_act=options['cnn_act']
       self.cnn_drop=options['cnn_drop']
       
       self.cnn_use_laynorm=options['cnn_use_laynorm']
       self.cnn_use_batchnorm=options['cnn_use_batchnorm']
       self.cnn_use_laynorm_inp=options['cnn_use_laynorm_inp']
       self.cnn_use_batchnorm_inp=options['cnn_use_batchnorm_inp']
       
       self.input_dim=int(options['input_dim'])
       
       self.fs=options['fs']
       
       self.N_cnn_lay=len(options['cnn_N_filt'])
       self.conv  = nn.ModuleList([])
       self.bn  = nn.ModuleList([])
       self.ln  = nn.ModuleList([])
       self.act = nn.ModuleList([])
       self.drop = nn.ModuleList([])
       
             
       if self.cnn_use_laynorm_inp:
           self.ln0=LayerNorm(self.input_dim)
           
       if self.cnn_use_batchnorm_inp:
           self.bn0=nn.BatchNorm1d([self.input_dim],momentum=0.05)
           
       current_input=self.input_dim 
       
       for i in range(self.N_cnn_lay):
         
         N_filt=int(self.cnn_N_filt[i])
         len_filt=int(self.cnn_len_filt[i])
         
         # dropout
         self.drop.append(nn.Dropout(p=self.cnn_drop[i]))
         
         # activation
         self.act.append(act_fun(self.cnn_act[i]))
                    
         # layer norm initialization         
         self.ln.append(LayerNorm([N_filt,int((current_input-self.cnn_len_filt[i]+1)/self.cnn_max_pool_len[i])]))

         self.bn.append(nn.BatchNorm1d(N_filt,int((current_input-self.cnn_len_filt[i]+1)/self.cnn_max_pool_len[i]),momentum=0.05))
            

         if i==0:
          self.conv.append(SincConv_fast(self.cnn_N_filt[0],self.cnn_len_filt[0],self.fs))
              
         else:
          self.conv.append(nn.Conv1d(self.cnn_N_filt[i-1], self.cnn_N_filt[i], self.cnn_len_filt[i]))
          
         current_input=int((current_input-self.cnn_len_filt[i]+1)/self.cnn_max_pool_len[i])

         
       self.out_dim=current_input*N_filt



    def forward(self, x):
       batch=x.shape[0]
       seq_len=x.shape[1]
       
       if bool(self.cnn_use_laynorm_inp):
        x=self.ln0((x))
        
       if bool(self.cnn_use_batchnorm_inp):
        x=self.bn0((x))
        
       x=x.view(batch,1,seq_len)

       
       for i in range(self.N_cnn_lay):
           
         if self.cnn_use_laynorm[i]:
          if i==0:
           x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(torch.abs(self.conv[i](x)), self.cnn_max_pool_len[i]))))  
          else:
           x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))   
          
         if self.cnn_use_batchnorm[i]:
          x = self.drop[i](self.act[i](self.bn[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))

         if self.cnn_use_batchnorm[i]==False and self.cnn_use_laynorm[i]==False:
          x = self.drop[i](self.act[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i])))

       
       x = x.view(batch,-1)

       return x
   
def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError 

    
   

# %%
import random

fs="6000"
cw_len="1024"
cw_shift="10"   

cnn_N_filt="100,80,80"
cnn_len_filt="251,5,5"
cnn_max_pool_len="3,3,3"
cnn_use_laynorm_inp="True"
cnn_use_batchnorm_inp="False"
cnn_use_laynorm="True,True,True"
cnn_use_batchnorm="False,False,False"
cnn_act="leaky_relu,leaky_relu,leaky_relu"
cnn_drop="0.1,0.1,0.1"


fc_lay="2048,2048,2048"
fc_drop="0.1,0.1,0.1"
fc_use_laynorm_inp="True"
fc_use_batchnorm_inp="False"
fc_use_batchnorm="True,True,True"
fc_use_laynorm="False,False,False"
fc_act="leaky_relu,leaky_relu,leaky_relu"

class_lay="35"
class_drop="0.0"
class_use_laynorm_inp="False"
class_use_batchnorm_inp="False"
class_use_batchnorm="False"
class_use_laynorm="False"
class_act="softmax"

lr="0.0001"
batch_size="128"
N_epochs="1500"
N_batches="800"
N_eval_epoch="8"

# %%
cnn_N_filt=list(map(int, cnn_N_filt.split(',')))
cnn_len_filt=list(map(int, cnn_len_filt.split(',')))
cnn_max_pool_len=list(map(int, cnn_max_pool_len.split(',')))
cnn_use_laynorm_inp=str_to_bool(cnn_use_laynorm_inp)
cnn_use_batchnorm_inp=str_to_bool(cnn_use_batchnorm_inp)
cnn_use_laynorm=list(map(str_to_bool, cnn_use_laynorm.split(',')))
cnn_use_batchnorm=list(map(str_to_bool, cnn_use_batchnorm.split(',')))
cnn_act=list(map(str, cnn_act.split(',')))
cnn_drop=list(map(float, cnn_drop.split(',')))


#[dnn]
fc_lay=list(map(int, fc_lay.split(',')))
fc_drop=list(map(float, fc_drop.split(',')))
fc_use_laynorm_inp=str_to_bool(fc_use_laynorm_inp)
fc_use_batchnorm_inp=str_to_bool(fc_use_batchnorm_inp)
fc_use_batchnorm=list(map(str_to_bool, fc_use_batchnorm.split(',')))
fc_use_laynorm=list(map(str_to_bool, fc_use_laynorm.split(',')))
fc_act=list(map(str, fc_act.split(',')))

#[class]
class_lay=list(map(int, class_lay.split(',')))
class_drop=list(map(float, class_drop.split(',')))
class_use_laynorm_inp=str_to_bool(class_use_laynorm_inp)
class_use_batchnorm_inp=str_to_bool(class_use_batchnorm_inp)
class_use_batchnorm=list(map(str_to_bool, class_use_batchnorm.split(',')))
class_use_laynorm=list(map(str_to_bool, class_use_laynorm.split(',')))
class_act=list(map(str, class_act.split(',')))



# loss function
cost = nn.NLLLoss()
wlen=4000

# Feature extractor CNN
CNN_arch = {'input_dim': wlen,
          'fs': int(fs),
          'cnn_N_filt': cnn_N_filt,
          'cnn_len_filt': cnn_len_filt,
          'cnn_max_pool_len':cnn_max_pool_len,
          'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
          'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
          'cnn_use_laynorm':cnn_use_laynorm,
          'cnn_use_batchnorm':cnn_use_batchnorm,
          'cnn_act': cnn_act,
          'cnn_drop':cnn_drop,          
          }


CNN_net=SincNet(CNN_arch)
CNN_net.cuda()



DNN1_arch = {'input_dim': CNN_net.out_dim,
          'fc_lay': fc_lay,
          'fc_drop': fc_drop, 
          'fc_use_batchnorm': fc_use_batchnorm,
          'fc_use_laynorm': fc_use_laynorm,
          'fc_use_laynorm_inp': fc_use_laynorm_inp,
          'fc_use_batchnorm_inp':fc_use_batchnorm_inp,
          'fc_act': fc_act,
          }

DNN1_net=MLP(DNN1_arch)
DNN1_net.cuda()


DNN2_arch = {'input_dim':fc_lay[-1] ,
          'fc_lay': class_lay,
          'fc_drop': class_drop, 
          'fc_use_batchnorm': class_use_batchnorm,
          'fc_use_laynorm': class_use_laynorm,
          'fc_use_laynorm_inp': class_use_laynorm_inp,
          'fc_use_batchnorm_inp':class_use_batchnorm_inp,
          'fc_act': class_act,
          }


DNN2_net=MLP(DNN2_arch)
DNN2_net.cuda()

print(os.getcwd())
# %%
load_dict = torch.load('/home/desild/work/academic/sem3/TrustworthyML-assignment/tacotron2/vctk/models/SINCNET_SR/20251129_142613/checkpoint.pth', weights_only=False)

CNN_net.load_state_dict(load_dict["CNN_net"])
DNN1_net.load_state_dict(load_dict["DNN1_net"])
DNN2_net.load_state_dict(load_dict["DNN2_net"])

# %%
inp = torch.randn(2,  wlen).cuda()
out1 = CNN_net(inp)
print(out1.shape)
out2 = DNN1_net(out1)
print(out2.shape)
out3 = DNN2_net(out2)
print(out3.shape)

# %%

# You can just define the dictionary directly
opt = {
    "lr": 0.0002,
    "b1": 0.5,
    "b2": 0.999,
    "n_cpu": 8,
    "latent_dim": 128,
    "img_size": (1,80, 16),
    "sample_interval": 400,
    "gamma": 0.75,
    "lambda_k": 0.001,
    "lambda_gp": 10,
    "small_dist":{
        "patch_size": 24,
        "num_patches": 1,
        },
    "load_gen": None, #os.path.join(MAIN_DIR, "Code", "saved_models", "generator.pth"),
    "load_dis": None, #os.path.join(MAIN_DIR, "Code", "saved_models", "discriminator.pth"),
    "load_dis_small": None,
}


device = "cuda" if torch.cuda.is_available() else "cpu"
opt = dycomutils.config.ConfigDict(opt)

# %%
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

LRELU_SLOPE = 0.1

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class HIFIGenerator(torch.nn.Module):
    def __init__(self, h):
        super(HIFIGenerator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x
    
    
hifigan_config = {
    "resblock": "1",
    "num_gpus": 0,
    "batch_size": 16,
    "learning_rate": 0.0002,
    "adam_b1": 0.8,
    "adam_b2": 0.99,
    "lr_decay": 0.999,
    "seed": 1234,

    "upsample_rates": [8,8,2,2],
    "upsample_kernel_sizes": [16,16,4,4],
    "upsample_initial_channel": 512,
    "resblock_kernel_sizes": [3,7,11],
    "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],

    "segment_size": 8192,
    "num_mels": 80,
    "num_freq": 1025,
    "n_fft": 1024,
    "hop_size": 256,
    "win_size": 1024,

    "sampling_rate": 22050,

    "fmin": 0,
    "fmax": 8000,
    "fmax_for_loss": None,
    "num_workers": 4,

    "dist_config": {
        "dist_backend": "nccl",
        "dist_url": "tcp://localhost:54321",
        "world_size": 1
    }
}

hifigan_config = dycomutils.config.ConfigDict(hifigan_config)

# %%

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = (80//16, 16 // 16)
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 64 * self.init_size[0] * self.init_size[1]))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 64, self.init_size[0], self.init_size[1])
        img = self.conv_blocks(out)
        return img.squeeze(1)

class SincNetCompound(nn.Module):
    def __init__(self, cnn, dnn1, dnn2):
        super(SincNetCompound, self).__init__()
        self.cnn = cnn
        self.dnn1 = dnn1
        self.dnn2 = dnn2
        
    def forward(self, x):
        x = self.cnn(x)
        x = self.dnn1(x)
        x = self.dnn2(x)
        return x
    
class AudioDiscriminator(nn.Module):
    def __init__(self, vector_size=4000):
        super(AudioDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(vector_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, img):
        validity = self.model(img)
        validity = torch.sigmoid(validity)
        return validity
    
class ImgDiscriminator(nn.Module):
    def __init__(self, img_size=opt.img_size):
        super(ImgDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_size)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.reshape(img.shape[0], -1)
        validity = self.model(img_flat)
        validity = torch.sigmoid(validity)
        return validity



    

    
compound_model = SincNetCompound(CNN_net, DNN1_net, DNN2_net)
compound_model.cuda()

gen_weight = torch.load("/home/desild/work/academic/sem3/TrustworthyML-assignment/tacotron2/vctk/models/HIFIGAN_ADV/20251209_231554/checkpoint.pth")

generator = Generator()
hifigan = HIFIGenerator(hifigan_config)
aud_disc = AudioDiscriminator(vector_size=wlen)
img_disc = ImgDiscriminator(img_size=opt.img_size)

hifigan.load_state_dict(gen_weight['hifigan'])
generator.load_state_dict(gen_weight['gen'])
aud_disc.load_state_dict(gen_weight['dis_aud'])
img_disc.load_state_dict(gen_weight['dis_img'])

hifigan.cuda()
generator.cuda()
aud_disc.cuda()
img_disc.cuda()
# %%
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple, List


import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple, List, Dict, Any

# --- Helper Function for Input Initialization ---

def create_initial_vector(
    length: int, 
    init_type: str = 'white_noise_tanh'
) -> torch.Tensor:
    """
    Creates an initial input vector for the MI attack.

    The paper finds that the choice of initialization impacts success[cite: 97, 167].
    'white noise and tanh activation' achieved good results[cite: 169].
    
    Args:
        length: The desired length of the vector (e.g., 3200 samples [cite: 145]).
        init_type: The initialization strategy. 
                   'zeros', 'white_noise_tanh', 'laplace' etc.

    Returns:
        A torch.Tensor to be used as the initial input x_0[cite: 71].
    """
    if init_type == 'zeros':
        # 'plain zeros perform best when inverting d-vectors' [cite: 200]
        # but are less suited for audio waveforms[cite: 167].
        return torch.zeros(1, length)
    
    elif init_type == 'laplace':
        # 'The Laplace distributions achieves the overall highest results'
        # for audio samples[cite: 171].
        # Paper used b=0.07[cite: 149].
        m = torch.distributions.laplace.Laplace(loc=0.0, scale=0.07)
        return m.sample((1, length))
        
    elif init_type == 'white_noise_tanh':
        # 'The best classification accuracy can be achieved 
        # with white noise and tanh activation.' [cite: 169]
        # This creates uniform noise and applies tanh 
        # to transform to range [-1, 1][cite: 148].
        noise = torch.rand(1, length) * 2 - 1  # Uniform noise in [-1, 1]
        return torch.tanh(noise)
        
    else:
        # Default to white noise
        noise = torch.rand(1, length) * 2 - 1
        return torch.tanh(noise)

# --- Algorithm 1: Standard Model Inversion ---

# Pure Model Inversion in generative setting
from typing import Any, Dict, List
import time

def gen_inversion_attack(
    model: nn.Module,
    target_class: int,
    input_vector: torch.Tensor,
    alpha_iterations: int = 1000,
    lambda_lr: float = 0.001,
    gamma_min_cost: float = 0.01,
    beta_patience: int = 10,
    lambda_cost: float = 1,
    norm:bool = True,
) -> Dict[str, Any]:
    """
    Implements the Standard Model Inversion (MI) attack (Algorithm 1).

    This function uses gradient descent to find an input vector 'x' that
    minimizes the cost function c(x) = 1 - p_t for a target class 't'[cite: 70, 73].

    Args:
        model: The trained SincNet model (or any nn.Module) to attack.
        target_class: The integer index of the speaker class to invert.
        input_vector: The initial vector 'x_0'[cite: 71]. 
                      Use create_initial_vector() to generate this.
        alpha_iterations: Max number of gradient descent iterations[cite: 61, 151].
        lambda_lr: The learning rate for gradient descent[cite: 61, 151].
        gamma_min_cost: Early stopping cost threshold[cite: 61].
        beta_patience: Early stopping patience (stops if cost doesn't 
                       improve for 'beta' steps)[cite: 61].

    Returns:
        The inverted audio sample as a torch.Tensor.
    """
    stime = time.perf_counter()
    device = next(model.parameters()).device
    model.eval()  # Set model to evaluation mode

    # The input vector 'x' is the parameter we want to optimize
    x = input_vector.clone().to(device).requires_grad_()

    # We can use a standard optimizer to perform the update step:
    # x_i <- x_{i-1} - lambda * grad(c(x_{i-1})) [cite: 61]
    optimizer = optim.SGD([x], lr=lambda_lr)
    
    best_cost = float('inf')
    best_x = x.clone()
    best_img = x.clone()
    best_audio = x.clone()
    cost_history: List[float] = []
    x = x.cuda()

    print(f"[Standard MI] Attacking class {target_class} for {alpha_iterations} iterations...")

    for i in range(alpha_iterations):
        optimizer.zero_grad()
        
        # 1. Get model output (logits)
        img = generator(x)
        audio_sample = hifigan(img.squeeze(1)).squeeze(1)
        audio_sample =audio_sample[:, audio_sample.shape[-1]//2 - wlen//2: audio_sample.shape[-1]//2 + wlen//2]
        
        if norm:
            audio_sample = audio_sample/32768.0
        probabilities = model(audio_sample)
        #print(probabilities)
        
        # 2. Convert to probabilities (softmax)
        # The cost function is defined on the probability p_t [cite: 67, 70]
        #print(audio_sample.shape, img.shape)
        # 3. Calculate cost c(x) = 1 - p_t [cite: 70]
        #print(probabilities[0, target_class].item(), torch.argmax(probabilities, dim=1).item())
        # print(probabilities[0, target_class].item(),  
        #       aud_disc(audio_sample)[0].item(), 
        #       img_disc(img)[0].item())
        
        cost = - probabilities[0, target_class] - lambda_cost*(aud_disc(audio_sample)[0] + img_disc(img)[0])
        
        # 4. Calculate gradient: grad(c(x))
        cost.backward()
        
        # 5. Apply gradient descent step
        optimizer.step()
        
        current_cost = cost.item()
        cost_history.append(current_cost)
        
        # 6. Store the best-performing vector
        if current_cost < best_cost:
            best_cost = current_cost
            best_x = x.clone().detach()
            best_img = img.clone().detach()
            best_audio = audio_sample.clone().detach()
        
        # 7. Check early stopping conditions
        
        # If c(x_i) <= gamma (minimum cost threshold) [cite: 61]
        if current_cost <= gamma_min_cost:
            print(f"Iteration {i}: Cost ({current_cost:.4f}) <= threshold. Stopping.")
            break
            
        # If c(x_i) >= max(c(x_{i-1}), ..., c(x_{i-beta})) (patience) [cite: 61]
        if i > beta_patience:
            recent_costs = cost_history[-beta_patience:]
            if current_cost >= max(recent_costs):
                print(f"Iteration {i}: Cost ({current_cost:.4f}) not improving. Stopping.")
                break
        
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {current_cost:.4f}")

    # Return argmin_x c(x) [cite: 61]
    etime = time.perf_counter()
    return {
        "best_x": best_x.detach(), "best_img": best_img, "best_audio": best_audio, 
        "cost_history": cost_history, "best_cost":    best_cost,
        "time_taken": etime - stime
        }




# --- 2. Setup Parameters ---

# Model and device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Instantiate your *real* trained SincNet model here
my_model = SincNetCompound(CNN_net, DNN1_net, DNN2_net).to(DEVICE)
# my_model.load_state_dict(torch.load("your_sincnet_weights.pth"))
my_model.eval()

# Attack parameters


def run_attack(speaker_class: int, idx:int, init_type: str = 'zeros'):
    # setting seed
    torch.manual_seed(random.randint(0, 10000))
    torch.cuda.manual_seed(random.randint(0, 10000))
    np.random.seed(random.randint(0, 10000))

    os.makedirs(f"inverted_samples/{init_type}/{speaker_class}/{idx}", exist_ok=True)

    # --- 3. Run Standard MI Attack (Algorithm 1) ---
    #print("\n--- Running Standard MI (Algorithm 1) ---")

    # Create an initial vector. Laplace performed well[cite: 171].
    initial_x0 = create_initial_vector(
        length=128, 
        init_type=init_type
    ).to(DEVICE)

    inverted_sample_gan_norm = gen_inversion_attack(
        model=my_model,
        target_class=speaker_class,
        input_vector=initial_x0,
        alpha_iterations=10000,
        lambda_lr=1, # [cite: 218]
        gamma_min_cost= 1e-8,
        beta_patience=200, # Increased patience
        norm=True,
    )
    
    inverted_sample_gan = gen_inversion_attack(
        model=my_model,
        target_class=speaker_class,
        input_vector=initial_x0,
        alpha_iterations=10000,
        lambda_lr=1, # [cite: 218]
        gamma_min_cost= 1e-8,
        beta_patience=200, # Increased patience
        norm=False,
    )

    #print(f"Sliding MI complete. Inverted sample shape: {inverted_sample_sliding.shape}")

    torch.save(inverted_sample_gan_norm, f"inverted_samples/{init_type}/{speaker_class}/{idx}/inverted_sample_gan-normaud2.pt")
    torch.save(inverted_sample_gan, f"inverted_samples/{init_type}/{speaker_class}/{idx}/inverted_sample_gan2.pt")
    return True

TYPE_INITS = ['zeros', 'white_noise_tanh', 'laplace']
SPEAKERS = list(range(35))  # Example speaker classes
NUM_TRYS = range(3)

with tqdm(total=len(TYPE_INITS)*len(SPEAKERS)*len(NUM_TRYS), desc="Processing TENSOR") as pbar:
    for init in TYPE_INITS:
        for speaker in SPEAKERS:
            for idx in NUM_TRYS:
                print(f"\n=== Running attack for speaker class {speaker}, index {idx}, init {init} ===")
                run_attack(speaker, idx, init)
                pbar.update(1)


