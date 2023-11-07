#!/usr/bin/env python
# coding: utf-8

# In[45]:


import IPython
from scipy.io import wavfile
import scipy.signal
import numpy as np
#import matplotlib.pyplot as plt
import librosa
import time
from datetime import timedelta as td
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[46]:



def _stft(y, n_fft, hop_length, win_length):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y, hop_length, win_length):
    return librosa.istft(y, hop_length, win_length)


def _amp_to_db(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)


def _db_to_amp(x,):
    return librosa.core.db_to_amplitude(x, ref=1.0)


# In[47]:


def  SVDnoise(src_data):
    A=_stft(src_data,512,64,512)
    U,S,Vt=np.linalg.svd(A)
    Snew=np.zeros(np.shape(A))
    for i in range(len(S)):
        if i<10:
            Snew[i,i]=S[i]
    C=np.matmul(U,Snew)
    Anew=np.matmul(C,Vt)
    x=_istft(Anew,64,512)
    return x

    
    
    


# In[48]:


def estimate(x,winLen,overlap,fs,delta1,delta2):
    fraShift = round(winLen*(1-overlap));  # Frame shift for consecutive frames
    lambda_n=np.zeros(np.shape(x));
    periodograms=np.abs(x)**2;
    Xki2=periodograms.copy();
    Pki=np.zeros(np.shape(x));
    pki=np.zeros(np.shape(x));
    dki=np.zeros(np.shape(x));
    dkiSmo=dki.copy()
    win = np.hamming(winLen);        # Analysis window
   #win.append(np.zeros(len(x)-winlen))
    freRan = range(winLen//2+1);          # Frequency range to be considered
    freNum = winLen//2+1;      # Frequency number
    fraNum = np.fix((len(x)-winLen)/fraShift);   # Frame number
    alpha_n = (0.15*fs/fraShift-1)/(0.15*fs/fraShift+1)
    alpha_x = (0.2*fs/fraShift-1)/(0.2*fs/fraShift+1)
    #alpha_n=0.8
    #alpha_x=0.9
    
    if overlap > 0.6 : 
        thetaMean = np.array([0.7100 , 0.8804 , 0.4437 , 0.3636])
        Sigma = np.diag([0.0937 , 0.1382 , 0.0345 , 0.0082]);  
                                                    
    else   :
        thetaMean = np.array([0.6002 , 1.4655 , 0.2575 , 0.5043]);   
        Sigma = np.diag([0.0640 , 0.5406 , 0.0120 , 0.0192]);     
    SigmaInv = np.linalg.inv(Sigma);  
    #delta1=2;
    #delta2=20;
    for i in range(len(x.transpose())):
        if i==0:
            Pki[:,i] = Xki2[:,i];
            pki[:,i] = np.zeros(freNum);
            lambda_n[:,i] = Xki2[:,i]; 
            theta_nv = thetaMean[0];      # Normalized Variance         
            theta_ndv = thetaMean[1];     # Normalized Differential Variance
            theta_nav = thetaMean[2];     # Normalized Average Variance
            theta_mcr = thetaMean[3];     # Median Crossing        
        Pki[:,i] = alpha_x*Pki[:,i-1]+(1-alpha_x)*Xki2[:,i];   
        theta_nv = alpha_x*theta_nv+(1-alpha_x)*(Xki2[:,i]-Pki[:,i])**2/Pki[:,i]**2;    
        theta_ndv = alpha_x*theta_ndv+(1-alpha_x)*(Xki2[:,i]-periodograms[:,i-1])**2/Pki[:,i]**2;   
        theta_nav = alpha_x*theta_nav+(1-alpha_x)*((Xki2[:,i]+periodograms[:,i-1])/2-Pki[:,i])**2/Pki[:,i]**2;
        mcInd = 1*((((Xki2[:,i]-0.69*(Pki[:,i])>0))*1 - 1*((periodograms[:,i-1]-0.69*(Pki[:,i-1])>0))) != 0)
        theta_mcr = alpha_x*theta_mcr+(1-alpha_x)*mcInd;
        theta_nv1=theta_nv-thetaMean[0];
        theta_ndv1=theta_ndv-thetaMean[1];
        theta_nav1=theta_nav-thetaMean[2];
        theta_mcr1=theta_mcr-thetaMean[3];    
        theta = np.array([theta_nv1,theta_ndv1,theta_nav1,theta_mcr1]); 
        dki[:,i]= np.diag((((theta).transpose())@SigmaInv@(theta)));#####################       
        dkiSmo[:,i] = dki[:,i]
        dkiSmo12 = delta1*1*(dkiSmo[:,i].all()<=delta1) + delta2*1*(dkiSmo[:,i].all()>=delta2) + np.multiply(dkiSmo[:,i],1*(dkiSmo[:,i].all()>delta1 and dkiSmo[:,i].all()<delta2));
        pki[:,i] = np.multiply((Xki2[:,i]<9.2*lambda_n[:,i-1]),(dkiSmo12-delta1)/(delta2-delta1));
        pki[:,i] = pki[:,i]+np.multiply((dkiSmo[:,i]>delta1),(Xki2[:,i]>=9.2*lambda_n[:,i-1]));
        pki[:,i] = np.multiply(pki[:,i],(Pki[:,i]>lambda_n[:,i-1]));
        alphaSpp_n = alpha_n+(1-alpha_n)*pki[:,i];   
        lambda_n[:,i] = np.multiply(alphaSpp_n,lambda_n[:,i-1])+np.multiply((1-alphaSpp_n),Xki2[:,i]);
    return lambda_n   


# In[49]:


def filter(audio_clip_band_limited,delta,winlength,hoplength):
    ystft=_stft(audio_clip_band_limited,winlength,hoplength,winlength)
    phiy=np.zeros(np.shape(ystft))
    phin=np.zeros(np.shape(ystft))
    phiy[:,0]=(np.abs(ystft[:,0]))**2
    phin=estimate(ystft,winlength,0.75,src_rate,8,18)
    alpha=0.32
    for i in range (1,len(ystft.transpose())):
        phiy[:,i]=phiy[:,i-1]*(alpha)+(1-alpha)*(np.abs(ystft[:,i]))**2
    isnr=(phiy/phin)-1    
    H=1*(isnr>delta)
    xstft=ystft*H
    x=_istft(xstft,hoplength,winlength)
    return x
    


# In[50]:


wav_loc = "2019_12_10_10_35_28.wav"#input audio file
src_rate, src_data = wavfile.read(wav_loc)
src_data = src_data/32768
x=filter(src_data,0.2,512,64)
x=SVDnoise(x)


# In[51]:

#IPython.display.Audio(data=x, rate=src_rate)
wavfile.write("abc.wav",src_rate,x);#output audio file

# In[ ]:





# In[ ]:



