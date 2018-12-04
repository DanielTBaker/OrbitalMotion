"""
Fuctions used in simulating and analyzing random fields
Daniel Baker
Last Edited 07/26/2017 
"""


import numpy as np
import weight
import fitmod
from scipy.interpolate import interp1d
import gc
import sys

##Simulate a real gaussian random field with the given power spectrum
def gaus_sim(C,nf,nt,g1,g2,fft12,fft21):
    g1[:]=np.random.normal(0,1,(nf,nt))+1j*np.random.normal(0,1,(nf,nt))
    fft12()
    g2*=np.sqrt(C/2)
    fft21()

##Simulate a real gaussian random field with the given power spectrum
def gaus_sim2(C2,nf,nt):
	C=np.zeros((C2.shape[0],2*C2.shape[1]))
	idx=(np.linspace(0,nt-1,nt)*2).astype(int)
	C[:,idx]=C2
	gauss=np.random.normal(0,1,(nf,2*nt))+1j*np.random.normal(0,1,(nf,2*nt))
	FG=np.fft.fft(np.fft.fft(gauss,axis=0),axis=1)*np.sqrt(C/2)
	gauss=np.fft.ifft(np.fft.ifft(FG,axis=1),axis=0)
	return(gauss)

##Simulate a chisq field from a gaussian field with a given power
def chi_sim(C,nf,nt,tm,g1,g2,fft12,fft21):
	if tm<=nt-1:
		gaus_sim(C,nf,nt,g1,g2,fft12,fft21)
	else:
		gaus_sim2(C,nf,nt,g1,g2,fft12,fft21)*np.sqrt(2)
	return(np.abs(g1)**2)

def recon(F,CC,CT,CCR,CTR,freq,A,FW,FG,W,G,ds2,ds3,fftds23,fftds32,fft,fft2,fft3,nt):
	F_rev=np.empty(F.shape,dtype='complex')
	F_rev[:,0]=np.conjugate(F[:,0])
	F_rev[:,1:]=np.conjugate(F[:,::-1][:,:nt-1])
	FW[:]=2j*np.pi*freq*CC*F/CT
	#fft_FW*=fcutA
	fft3()
	#print(fft_dspec2.max())
	FG[:]=F_rev/CTR
	#fft_FG*=fcutArev
	fft2()
	ds2[:]*=G*nt
	#print(fft_dspec2.max())
	FW[:]=2j*np.pi*freq*CCR*F_rev/CTR
	#fft_FW*=fcutArev
	fft()
	#print(fft_W.max())
	FG[:]=F/CT
	#fft_FG*=fcutA
	fft2()
	#print(fft_G.max())
	ds2[:]+=G*W*nt
	fftds23()
	ds3*=2j*np.pi*freq*A/np.sqrt(nt)
	fftds32()
	ds2*=-np.sqrt(nt)

##Simulated lens recover with no lensing
def nolens_sim(CG,CC,CT,CCR,CTR,N,freq,nf,nt,mf,mt,A,g1,g2,fftg12,fftg21,ds1,ds2,ds3,fftds12,fftds23,fftds32,FW,FG,W,G,fft,fft2,fft3):
	##Simulate ChiSq field with Holes
	ds1[:]=chi_sim(CG,nf,nt,0,g1,g2,fftg12,fftg21)
	ds1+=np.sqrt(N)*np.random.normal(0,1,(nf,nt))
	ds1+=1-np.mean(ds1,axis=0)
	ds1*=mf[:,np.newaxis]*mt[np.newaxis,:]
	##Find Power Spectrum
	fftds12()
	fftds23()
	F=np.zeros(ds3.shape,dtype=complex)
	F[:]=ds3[:]/np.sqrt(nf*nt)
	##Recover Lens
	recon(F,CC,CT,CCR,CTR,freq,A,FW,FG,W,G,ds2,ds3,fftds23,fftds32,fft,fft2,fft3,nt)

##Simulations for noise determination (Average power after MF removal)
def noise_sim(CG,CC,CT,CCR,CTR,N,freq,nf,nt,mf,mt,A,g1,g2,fftg12,fftg21,ds1,ds2,ds3,fftds12,fftds23,fftds32,FW,FG,W,G,fft,fft2,fft3,MF):
	nolens_sim(CG,CC,CT,CCR,CTR,N,freq,nf,nt,mf,mt,A,g1,g2,fftg12,fftg21,ds1,ds2,ds3,fftds12,fftds23,fftds32,FW,FG,W,G,fft,fft2,fft3)
	ds2-=MF
	fftds23()
	FL=np.abs(ds3/np.sqrt(nt))**2
	return(FL)

##Lensing sim
def Lsim(CG,CC,CT,CCR,CTR,N,freq,nf,nt,mf,mt,A,g1,g2,fftg12,fftg21,ds1,ds2,ds3,fftds12,fftds23,fftds32,FW,FG,W,G,fft,fft2,fft3,t2):
	ds1[:]=chi_sim(CG,nf,nt,0,g1,g2,fftg12,fftg21)
	interp=interp1d(np.linspace(0,nt-1,nt),ds1,fill_value='extrapolate')
	ds1[:]=interp(t2)
	m=np.mean(ds1,axis=0)
	ds1+=np.sqrt(N)*np.random.normal(0,1,(nf,nt))-m+1
	ds1*=mf[:,np.newaxis]*mt[np.newaxis,:]
	fftds12()
	fftds23()
	F=np.zeros(ds3.shape,dtype=complex)
	F[:]=ds3[:]/np.sqrt(nf*nt)
	##Recover Lens
	recon(F,CC,CT,CCR,CTR,freq,A,FW,FG,W,G,ds2,ds3,fftds23,fftds32,fft,fft2,fft3,nt)
	

def NcovSim(CG,CC,CT,CCR,CTR,N,freq,nf,nt,mf,mt,A,g1,g2,fftg12,fftg21,ds1,ds2,ds3,fftds12,fftds23,fftds32,FW,FG,W,G,fft,fft2,fft3,MF,NL):
	nolens_sim(CG,CC,CT,CCR,CTR,N,freq,nf,nt,mf,mt,A,g1,g2,fftg12,fftg21,ds1,ds2,ds3,fftds12,fftds23,fftds32,FW,FG,W,G,fft,fft2,fft3)
	ds2-=MF
	L=weight.NseWt(ds2,ds3,fftds23,fftds32,NL)
	return(L)

def TransSim(CG,CC,CT,CCR,CTR,N,freq,nf,nt,mf,mt,A,g1,g2,fftg12,fftg21,ds1,ds2,ds3,fftds12,fftds23,fftds32,FW,FG,W,G,fft,fft2,fft3,t2,MF,NL):
	Lsim(CG,CC,CT,CCR,CTR,N,freq,nf,nt,mf,mt,A,g1,g2,fftg12,fftg21,ds1,ds2,ds3,fftds12,fftds23,fftds32,FW,FG,W,G,fft,fft2,fft3,t2)
	ds2-=MF
	L=weight.NseWt(ds2,ds3,fftds23,fftds32,NL)
	return(L)

##Simulations for noise determination (Average power after MF removal)
def comb_sim(CG,CC,CT,CCR,CTR,N,freq,nf,nt,mf,mt,fcutA,A,Norm=True):
	lens=nolens_sim(CG,CC,CT,CCR,CTR,N,freq,nf,nt,mf,mt,fcutA,A,Norm)
	FL=np.fft.rfft(lens,axis=1)/np.sqrt(nt)
	return(lens,np.abs(FL)**2)

##Average many recovered lenses with a single input lens
def TransSimAv(CG,CC,CT,CCR,CTR,N,freq,nf,nt,mf,mt,A,g1,g2,fftg12,fftg21,ds1,ds2,ds3,fftds12,fftds23,fftds32,FW,FG,W,G,fft,fft2,fft3,MF,NL,psip,n,rnk,i,ts,tfunc):
	L=np.zeros(nt)
	t2=np.linspace(0,nt-1,nt)+psip
	t2-=t2.min()
	##Loop over simulations
	for k in range(n):
		L+=TransSim(CG,CC,CT,CCR,CTR,N,freq,nf,nt,mf,mt,A,g1,g2,fftg12,fftg21,ds1,ds2,ds3,fftds12,fftds23,fftds32,FW,FG,W,G,fft,fft2,fft3,t2,MF,NL)
		if rnk==0:
			print(str(k+1)+' : '+str(i)+' : '+str(tfunc()-ts))
	return(L)
	
