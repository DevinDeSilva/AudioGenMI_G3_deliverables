# %%
import os
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
import sys
sys.path.append("/home/desild/work/academic/sem3/TrustworthyML-assignment/tacotron2")
print(sys.path)

# Load Configs
load_dotenv("/home/desild/work/academic/sem3/TrustworthyML-assignment/.env")
cuda = True if torch.cuda.is_available() else False

run = neptune.init_run(
    project="Botz/Audio-MI",
    name="sinc-net-training",
    api_token=os.getenv("NEPTUNE_API_TOKEN")
)

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
RESAMPLE_RATE = 6000

fs=f"{RESAMPLE_RATE}"
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

class_lay="2"
class_drop="0.0"
class_use_laynorm_inp="False"
class_use_batchnorm_inp="False"
class_use_batchnorm="False"
class_use_laynorm="False"
class_act="softmax"

lr="0.0004"
batch_size="128"
N_epochs="1500"
N_batches="800"
N_eval_epoch="8"
seed="1234"

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


# %%
from tacotron2.text import symbols
global symbols
import json
import numpy as np


class Config:
    # ** Audio params **
    sampling_rate = 16000                        # Sampling rate
    filter_length = 1024                         # Filter length
    hop_length = 256                             # Hop (stride) length
    win_length = 1024                            # Window length
    mel_fmin = 0.0                               # Minimum mel frequency
    mel_fmax = 8000.0                            # Maximum mel frequency
    n_mel_channels = 80                          # Number of bins in mel-spectrograms
    max_wav_value = 32768.0                      # Maximum audiowave value

    # Audio postprocessing params
    snst = 0.00005                               # filter sensitivity
    wdth = 1000                                  # width of filter

    # ** Tacotron Params **
    # Symbols
    n_symbols = len(symbols)                     # Number of symbols in dictionary
    symbols_embedding_dim = 512                  # Text input embedding dimension

    # Speakers
    n_speakers = 35                             # Number of speakers
    speakers_embedding_dim = 32                  # Speaker embedding dimension
    try:
        speaker_coefficients = json.load(open('train/speaker_coefficients.json'))  # Dict with speaker coefficients
    except IOError:
        print("Speaker coefficients dict is not available")
        speaker_coefficients = None

    # Emotions
    use_emotions = False                         # Use emotions
    n_emotions = 15                              # N emotions
    emotions_embedding_dim = 8                   # Emotion embedding dimension
    try:
        emotion_coefficients = json.load(open('train/emotion_coefficients.json'))  # Dict with emotion coefficients
    except IOError:
        print("Emotion coefficients dict is not available")
        emotion_coefficients = None

    # Encoder
    encoder_kernel_size = 5                      # Encoder kernel size
    encoder_n_convolutions = 3                   # Number of encoder convolutions
    encoder_embedding_dim = 512                  # Encoder embedding dimension

    # Attention
    attention_rnn_dim = 1024                     # Number of units in attention LSTM
    attention_dim = 128                          # Dimension of attention hidden representation

    # Attention location
    attention_location_n_filters = 32            # Number of filters for location-sensitive attention
    attention_location_kernel_size = 31          # Kernel size for location-sensitive attention

    # Decoder
    n_frames_per_step = 1                        # Number of frames processed per step
    max_frames = 2000                            # Maximum number of frames for decoder
    decoder_rnn_dim = 1024                       # Number of units in decoder LSTM
    prenet_dim = 256                             # Number of ReLU units in prenet layers
    max_decoder_steps = int(max_frames / n_frames_per_step)  # Maximum number of output mel spectrograms
    gate_threshold = 0.5                         # Probability threshold for stop token
    p_attention_dropout = 0.1                    # Dropout probability for attention LSTM
    p_decoder_dropout = 0.1                      # Dropout probability for decoder LSTM
    decoder_no_early_stopping = False            # Stop decoding once all samples are finished

    # Postnet
    postnet_embedding_dim = 512                  # Postnet embedding dimension
    postnet_kernel_size = 5                      # Postnet kernel size
    postnet_n_convolutions = 5                   # Number of postnet convolutions

    # Optimization
    mask_padding = False                         # Use mask padding
    use_loss_coefficients = False                # Use balancing coefficients
    # Loss scale for coefficients
    if emotion_coefficients is not None and speaker_coefficients is not None:
        loss_scale = 1.5 / (np.mean(list(speaker_coefficients.values())) * np.mean(list(emotion_coefficients.values())))
    else:
        loss_scale = None

    # ** Waveglow params **
    n_flows = 12                                 # Number of steps of flow
    n_group = 8                                  # Number of samples in a group processed by the steps of flow
    n_early_every = 4                            # Determines how often (i.e., after how many coupling layers) a number of channels (defined by --early-size parameter) are output to the loss function
    n_early_size = 2                             # Number of channels output to the loss function
    wg_sigma = 1.0                               # Standard deviation used for sampling from Gaussian
    segment_length = 4000                        # Segment length (audio samples) processed per iteration
    wn_config = dict(
        n_layers=8,                              # Number of layers in WN
        kernel_size=3,                           # Kernel size for dialted convolution in the affine coupling layer (WN)
        n_channels=512                           # Number of channels in WN
    )

    # ** Script args **
    model_name = "WaveGlow"
    output_directory = "logs"               # Directory to save checkpoints
    log_file = "nvlog.json"                      # Filename for logging

    anneal_steps = None                          # Epochs after which decrease learning rate
    anneal_factor = 0.1                          # Factor for annealing learning rate

    tacotron2_checkpoint = 'pretrained/t2_fp32_torch'   # Path to pre-trained Tacotron2 checkpoint for sample generation
    waveglow_checkpoint = 'pretrained/wg_fp32_torch'    # Path to pre-trained WaveGlow checkpoint for sample generation
    restore_from = 'pretrained/wg_fp32_torch'           # Checkpoint path to restore from

    # Training params
    epochs = 500                                # Number of total epochs to run
    epochs_per_checkpoint = 5                   # Number of epochs per checkpoint
    seed = 1234                                  # Seed for PyTorch random number generators
    dynamic_loss_scaling = True                  # Enable dynamic loss scaling
    amp_run = False                              # Enable AMP (FP16) # TODO: Make it work
    cudnn_enabled = True                         # Enable cudnn
    cudnn_benchmark = False                      # Run cudnn benchmark

    # Optimization params
    use_saved_learning_rate = False
    learning_rate = 1e-3                         # Learning rate
    weight_decay = 1e-6                          # Weight decay
    grad_clip_thresh = 3.4028234663852886e+38    # Clip threshold for gradients
    batch_size = 1024                              # Batch size per GPU

    # Dataset
    load_mel_from_dist = False                   # Loads mel spectrograms from disk instead of computing them on the fly
    text_cleaners = ['english_cleaners']         # Type of text cleaners for input text
    training_files = 'train/train.txt'      # Path to training filelist
    validation_files = 'train/val.txt'      # Path to validation filelist

    dist_url = 'tcp://localhost:23456'           # Url used to set up distributed training
    group_name = "group_name"                    # Distributed group name
    dist_backend = "nccl"                        # Distributed run backend

    # Sample phrases
    phrases = {
        'speaker_ids': [0, 2],
        'texts': [
            'Hello, how are you doing today?',
            'I would like to eat a Hamburger.',
            'Hi.',
            'I would like to eat a Hamburger. Would you like to join me?',
            'Do you have any hobbies?'
        ]
    }



# %%
# setting seed
torch.manual_seed(1710)
np.random.seed(1710)

config = Config()

# Converting context and shift in samples
wlen=config.segment_length


# %%
# Feature extractor CNN
CNN_arch = {
    'input_dim': wlen,
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



DNN1_arch = {
    'input_dim': CNN_net.out_dim,
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

# %%
inp = torch.randn(2,  wlen).cuda()
out1 = CNN_net(inp)
print(out1.shape)

print(CNN_net.out_dim)
out2 = DNN1_net(out1)
print(out2.shape)

# %%
pout=DNN2_net(DNN1_net(CNN_net(inp)))
print(pout.shape)

# %%

model_config = dict(
            n_mel_channels=config.n_mel_channels,
            n_flows=config.n_flows,
            n_group=config.n_group,
            n_early_every=config.n_early_every,
            n_early_size=config.n_early_size,
            sigma=config.wg_sigma,
            WN_config=config.wn_config
        )

# You can just define the dictionary directly
opt = {
    "lr": 0.0002,
    "b1": 0.5,
    "b2": 0.999,
    "n_cpu": 8,
    "latent_dim": 64,
    "img_size": (1,128,128),
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


p_height = 30  # Rectangular patch height
p_width = 10   # Rectangular patch width
device = "cuda" if torch.cuda.is_available() else "cpu"
opt = dycomutils.config.ConfigDict(opt)

# %%
ROOT_FOL = '../..'

# config.training_files = os.path.join(ROOT_FOL, 'PREPROCESSED_TIMIT', config.training_files)
# config.validation_files = os.path.join(ROOT_FOL, 'PREPROCESSED_TIMIT', config.validation_files)
# config.output_directory = os.path.join(ROOT_FOL, 'TRAINING', config.output_directory)
# config.tacotron2_checkpoint = os.path.join(ROOT_FOL, 'TRAINING', 'CHECKPOINTS', config.tacotron2_checkpoint) if config.tacotron2_checkpoint is not None else None
# config.waveglow_checkpoint = os.path.join(ROOT_FOL, 'TRAINING', 'CHECKPOINTS', config.waveglow_checkpoint) if config.waveglow_checkpoint is not None else None
# config.restore_from = os.path.join(ROOT_FOL, 'TRAINING', 'CHECKPOINTS', config.restore_from) if config.restore_from is not None else None

#os.makedirs(config.output_directory, exist_ok=True)

# %%
from common_local.utils import load_wav_to_torch, load_filepaths_and_text, to_gpu
from common_local.audio_processing import dynamic_range_compression, dynamic_range_decompression
from sentence_transformers import SentenceTransformer
import librosa
import random
from typing import Optional, List, Tuple

class MelAudioLoaderVoxCeleb(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) computes mel-spectrograms from audio files.
    """

    def __init__(self, main_file, filter_length, hop_length, win_length,
                 n_mel_channels, sampling_rate, mel_fmin, mel_fmax, segment_length, max_wav_value, base_directory='..',
                 filter_speakers: Optional[List[str]] = None):
        
        self.main_file = main_file
        if filter_speakers is not None:
            self.main_file = self.main_file[self.main_file['speaker_id'].isin(filter_speakers)]
            print(f"Filtered dataset to {len(self.main_file)} items for speakers: {filter_speakers}")

        self.max_wav_value = max_wav_value
        self.sampling_rate = sampling_rate
        self.base_directory = base_directory
        self.random_amp = 0.2
        
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.stats = {
            "audio_len_distribution": {},
        }
        
        self.load_in_memory = {}
        
        self.speaker_id2id_map = {}
        self.edit_file_indexes()
        self.segment_length = segment_length
        random.seed(1234)
        self.main_file = self.main_file.sample(frac=1).reset_index(drop=True)
        
    def edit_file_indexes(self):
        self.speaker_id2id_map = SPEAKER_TO_ID
        print("Speaker ID to Index Map:", len(self.speaker_id2id_map))

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output
    
    def load_text(self, file_path):
        file_path = file_path.replace("WAV.wav", "WRD")
        with open(file_path, 'r') as f:
            text = f.read().strip()
            
        text = text.split('\n')
        text = [x.strip().split(' ', 2) for x in text if len(x.split(' ', 2)) == 3]
        return [int(x[0]) for x in text], [int(x[1]) for x in text], [x[2] for x in text]
    
    def get_mel_audio_pair(self, filename):
        if filename not in self.load_in_memory:
            audio, sampling_rate = librosa.load(filename, sr=None)
            self.load_in_memory[filename] = {}
            self.load_in_memory[filename]["audio"] = (audio, sampling_rate)
        else:
            audio, sampling_rate = self.load_in_memory[filename]["audio"]

        # if sampling_rate != self.sampling_rate:
        #     raise ValueError("{} {} SR doesn't match target {} SR".format(
        #         sampling_rate, self.sampling_rate))
        
        #
        # if RESAMPLE_RATE != self.sampling_rate:
        #import soundfile as sf
        #sf.write("before_resample.wav", audio, sampling_rate)
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=sampling_rate//8)
        #sf.write("after_resample.wav", audio, sampling_rate//8)
        audio = torch.from_numpy(audio)

        self.stats["audio_len_distribution"][filename] = audio.size(0)
        # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start:audio_start+self.segment_length]
        else:
            audio = torch.nn.functional.pad(
                audio, (0, self.segment_length - audio.size(0)), 'constant').data
        
        #sf.write("after_resample2.wav", audio.numpy(), sampling_rate//8)
        audio = audio * random.uniform(1.0 - self.random_amp, 1.0 + self.random_amp)
        audio_norm = audio / self.max_wav_value
        mel_spec = librosa.feature.melspectrogram(
            y=audio_norm.numpy(),
            sr=self.sampling_rate,
            n_fft=self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mel_channels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax
        )
        
        mel_spec = torch.from_numpy(mel_spec).unsqueeze(0)
        mel_output = self.spectral_normalize(mel_spec)
        mel_output = mel_output.squeeze(0)
        return mel_output, audio, len(audio)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        output = self.get_mel_audio_pair(self.main_file.iloc[index,2])
        ids = self.speaker_id2id_map[self.main_file.iloc[index,6]]

        output = (output[0], output[1], output[2], ids)
        return output

    def __len__(self):
        return self.main_file.shape[0]-1


def batch_to_gpu(batch) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, List[str]]: 
    x, y, len_y, text = batch
    x = to_gpu(x).float()
    y = to_gpu(y).float()
    len_y = to_gpu(torch.sum(len_y))

    return (x, y), y, len_y, text

def get_text_embedding(text, model):
    embeddings = model.encode([text])
    #print(embeddings.shape)
    return embeddings


# %%

val_data = pd.read_csv("/home/desild/work/academic/sem3/TrustworthyML-assignment/data/raw/vctk/val_data_top.csv")
train_data = pd.read_csv("/home/desild/work/academic/sem3/TrustworthyML-assignment/data/raw/vctk/train_data_top.csv")

SPEAKER_TO_ID = {k:v for v,k in enumerate(sorted(train_data['GENDER'].unique()))}

print("Train class balance:")
print((train_data['GENDER'].map(SPEAKER_TO_ID).value_counts()/len(train_data)).sort_index())
print("Validation class balance:")
print((val_data['GENDER'].map(SPEAKER_TO_ID).value_counts()/len(val_data)).sort_index())

weights = 1.0 / train_data['GENDER'].map(SPEAKER_TO_ID).value_counts().sort_index()
class_weights = torch.FloatTensor(weights.values).cuda()
print(class_weights)

# loss function
cost = nn.NLLLoss(weight=class_weights)

# %%
len(train_data), len(val_data)

# %%
train_ds = MelAudioLoaderVoxCeleb(train_data, 
                                  config.filter_length, 
                                  config.hop_length, 
                                  config.win_length,
                                  config.n_mel_channels, 
                                  config.sampling_rate, 
                                  config.mel_fmin, 
                                  config.mel_fmax,
                                  config.segment_length, 
                                  config.max_wav_value
                                  )

train_loader = DataLoader(train_ds,
                        num_workers=8,
                        shuffle=True,
                        batch_size=config.batch_size,
                        drop_last=False)

val_ds = MelAudioLoaderVoxCeleb(val_data, 
                                  config.filter_length, 
                                  config.hop_length, 
                                  config.win_length,
                                  config.n_mel_channels, 
                                  config.sampling_rate, 
                                  config.mel_fmin, 
                                  config.mel_fmax,
                                  config.segment_length, 
                                  config.max_wav_value
                                  )

val_loader = DataLoader(val_ds,
                        num_workers=8,
                        shuffle=True,
                        batch_size=config.batch_size,
                        drop_last=False)

# criterion = WaveGlowLoss(sigma=1.0)

# models = [model_waveglow, audio_discriminator, generator]
# optimizer_waveglow = torch.optim.Adam(
#     (params for model in models for params in model.parameters()),
#     lr=config.learning_rate,
#     weight_decay=config.weight_decay)

# # Optimizers
# optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# optimizer_D_small = torch.optim.Adam(discriminator_small.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))



optimizer_CNN = torch.optim.Adam(CNN_net.parameters(), lr=float(lr), eps=1e-8) 
optimizer_DNN1 = torch.optim.Adam(DNN1_net.parameters(), lr=float(lr), eps=1e-8) 
optimizer_DNN2 = torch.optim.Adam(DNN2_net.parameters(), lr=float(lr), eps=1e-8)

# %%
inp = torch.randn(2,  wlen).cuda()
out1 = CNN_net(inp)
print(out1.shape)

print(CNN_net.out_dim)
out2 = DNN1_net(out1)
print(out2.shape)

pout=DNN2_net(DNN1_net(CNN_net(inp)))
print(pout.shape)

# %%
for d in train_ds:
    #print(d)
    pass

# %%
import seaborn as sns

sns.histplot(train_ds.stats['audio_len_distribution'].values())

# %%
from IPython.display import Audio, display

# already_displayed = set()
# for d in train_ds:
#     mel, aud, _, sp_id = d

#     if sp_id in already_displayed:
#         continue
#     print("speaker id:", sp_id)
#     aud = aud*train_ds.max_wav_value 
#     display(Audio(aud, rate=config.sampling_rate))
#     already_displayed.add(sp_id)


# %%
# already_displayed = set()
# for d in val_ds:
#     mel, aud, _, sp_id = d

#     if sp_id in already_displayed:
#         continue
#     aud = aud*train_ds.max_wav_value 
#     print("speaker id:", sp_id)
#     display(Audio(aud, rate=config.sampling_rate))
#     already_displayed.add(sp_id)

# CNN_net.load_state_dict(torch.load('models/SINCNET/20251114_230216/CNN_net.pth'))
# DNN1_net.load_state_dict(torch.load('models/SINCNET/20251114_230216/DNN1_net.pth'))
# DNN2_net.load_state_dict(torch.load('models/SINCNET/20251114_230216/DNN2_net.pth'))


# %%
from IPython.display import Audio, display

with tqdm(total=config.epochs, desc="Processing Batches") as pbar:
    for epoch in range(config.epochs):
    # Used to calculate avg items/sec over epoch
        epoch_logs = []
        # Used to calculate avg loss over epoch
        num_iters = 0
        loss_sum=0.0
        err_sum=0.0
        
        CNN_net.train()
        DNN1_net.train()
        DNN2_net.train()
        for i, batch in enumerate(train_loader):
            #(mel, aud), y, _, _, sp_id, _ = batch_to_gpu(batch)
            mel, aud, _, sp_id = batch
            mel = to_gpu(mel).float()
            aud = to_gpu(aud).float()
            sp_id = to_gpu(sp_id).long()

            optimizer_CNN.zero_grad()
            optimizer_DNN1.zero_grad()
            optimizer_DNN2.zero_grad()

            pout=DNN2_net(DNN1_net(CNN_net(aud)))

            pred=torch.max(pout,dim=1)[1]
            loss = cost(pout, sp_id)
            err = torch.mean((pred!=sp_id).float())
            
            loss.backward()
            optimizer_CNN.step()
            optimizer_DNN1.step()
            optimizer_DNN2.step()
            
            loss_sum=loss_sum+loss.detach()
            err_sum=err_sum+err.detach()
            
            # print(
            #     "[Epoch %d/%d] [Batch %d/%d] [loss: %f] [error: %f]"
            #     % (epoch, config.epochs, i, len(train_loader), loss.item(), err.item())
            # )

            epoch_logs.append({
                "ds":"train",
                "loss": loss.item(),
                "error": err.item(),
            })
            
        CNN_net.eval()
        DNN1_net.eval()
        DNN2_net.eval()
        if epoch % config.epochs_per_checkpoint == 0: 
            for i, batch in enumerate(val_loader):
                #(mel, aud), y, _, _, sp_id, _ = batch_to_gpu(batch)
                mel, aud, _, sp_id = batch
                mel = to_gpu(mel).float()
                aud = to_gpu(aud).float()
                sp_id = to_gpu(sp_id).long()

                pout=DNN2_net(DNN1_net(CNN_net(aud)))

                pred=torch.max(pout,dim=1)[1]
                loss = cost(pout, sp_id)
                err = torch.mean((pred!=sp_id).float())
                
                # print(
                #     "[Validation] [Batch %d/%d] [loss: %f] [error: %f]"
                #     % (i, len(val_loader), loss.item(), err.item())
                # )
                
                epoch_logs.append({
                    "ds":"val",
                    "loss": loss.item(),
                    "error": err.item(),
                })
            
        epoch_df = pd.DataFrame(epoch_logs)
        run["epoch/train/loss"].append(epoch_df.loc[epoch_df["ds"] == "train", "loss"].mean())
        run["epoch/train/error"].append(epoch_df.loc[epoch_df["ds"] == "train", "error"].mean())

        if epoch % config.epochs_per_checkpoint == 0:
            run["epoch/val/loss"].append(epoch_df.loc[epoch_df["ds"] == "val", "loss"].mean())
            run["epoch/val/error"].append(epoch_df.loc[epoch_df["ds"] == "val", "error"].mean())
            pbar.set_postfix(
                train_loss=epoch_df.loc[epoch_df["ds"] == "train", "loss"].mean(),
                train_error=epoch_df.loc[epoch_df["ds"] == "train", "error"].mean(),
                val_loss=epoch_df.loc[epoch_df["ds"] == "val", "loss"].mean(),
                val_error=epoch_df.loc[epoch_df["ds"] == "val", "error"].mean()
                )

        pbar.update(1)


# %%
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

model_type = "SINCNET_GENDER"
os.makedirs("models/{}/{}".format(model_type, timestamp), exist_ok=True)

config_dict = {
    "config": config.__dict__,
    "model_config": {
        "CNN_arch": CNN_arch,
        "DNN1_arch": DNN1_arch,
        "DNN2_arch": DNN2_arch,
        "fs": fs,
        "n_mel_channels": config.n_mel_channels
    }
}

save_dict = {
    'CNN_net': CNN_net.state_dict(),
    'DNN1_net': DNN1_net.state_dict(),
    'DNN2_net': DNN2_net.state_dict(),
    'optimizer_CNN': optimizer_CNN.state_dict(),
    'optimizer_DNN1': optimizer_DNN1.state_dict(),
    'optimizer_DNN2': optimizer_DNN2.state_dict(),
    'epoch': config.epochs,
    'batch_size': config.batch_size, 
    'learning_rate': float(lr),
    'config': config_dict,
    'speaker_to_id_map': SPEAKER_TO_ID
    }


torch.save(save_dict, "models/{}/{}/checkpoint.pth".format(model_type, timestamp))
# torch.save(model_waveglow.state_dict(), "models/SINCNET/{}/waveglow.pth".format(timestamp))
# torch.save(audio_discriminator.state_dict(), "models/SINCNET/{}/audio_discriminator.pth".format(timestamp))

run["model/saved_model/checkpoint"].upload("models/{}/{}/checkpoint.pth".format(model_type, timestamp))
# run["model/saved_model/waveglow"].upload("models/SINCNET/{}/waveglow.pth".format(timestamp))
# run["model/saved_model/audio_dis"].upload("models/SINCNET/{}/audio_discriminator.pth".format(timestamp))





run.stop()

# %%



