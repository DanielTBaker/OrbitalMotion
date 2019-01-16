"""
Fuctions used in Fitting Routines (powerspectra or lensing)
Daniel Baker
Last Edited 11/30/2017 
"""

import numpy as np
from scipy.optimize import minimize,curve_fit
from scipy.stats import norm
from time import clock
import sys


##Functions used to determine regions of high noise for masking
def hist_fit(x,mu,s):
	y=np.zeros(x.shape[0]-1)
	for i in range(y.shape[0]):
		y[i]=norm.cdf((x[i+1]-mu)/s)-norm.cdf((x[i]-mu)/s)
	return(y)

##Bins data and fits to a gaussian
def rng_find(s):
	n,bins=np.histogram(s,bins=30)
	n=n.astype(float)/s.shape[0]
	popt,pcov=curve_fit(hist_fit,bins,n,p0=np.array([s.mean(),s.std()]))
	return(popt)


##Functions for converting from gaussian to chi-square with holes
##Convert from gaussian to chi using Cchi=Cgauss*Cgauss + delta(l)X(total gaussian power)
def gauss_to_chi(C,fft1,fft2,fftF,fftB):
	Ctemp=np.zeros(C.shape,dtype=complex)
	fft2[:]=np.copy(C[:])
	fft2[1:,:]=fft2[1:,:][::-1,:]
	fft2[:,1:]=fft2[:,1:][:,::-1]
	fftB()
	Ctemp[:]=np.copy(fft1)
	fft2[:]=np.copy(C[:])
	fftB()
	fft1[:]*=Ctemp
	fftF()
	fft2[0,0]+=(np.sum(C)/np.sqrt(C.shape[0]*C.shape[1]))**2
	return(np.real(fft2[:1+int(C.shape[0]/2),:]))

##Alter power spectrum to account for holes (C=C0*Cmask)
def pow_holes(C,IFCM,fft1,fft2,fft3,fftF12,fftF23,fftB32,fftB21):
	fft3[:]=C[:]
	fftB32()
	fftB21()
	fft1[:]*=IFCM
	fftF12()
	fftF23()
	fft3[:]=np.real(fft3)

##Power spectrum function for gaussian field
def pow_arr2D2(X,A=1e8,t01=.01,nt1=2,t02=.01,nt2=2,nf=2,f01=.01,f02=.01,B=0,A2=1,b01=0,b02=0):
	f=X[1]
	tau=X[0]
	temp=np.zeros(tau.shape)
	#temp[tau[:,1]<0,:]=(1./np.power(1+np.power((f[tau[:,1]<0,:]+B*np.sqrt(np.abs(tau[tau[:,1]<0,:])))/(f01+b01*np.abs(tau[tau[:,1]<0,:])),2),nf))*(A/np.power(1+np.power(tau[tau[:,1]<0,:]/t01,2),nt1))+(A/np.power(1+np.power(tau[tau[:,1]<0,:]/t02,2),nt2))*A2/np.power(1+np.power((f[tau[:,1]<0,:]-B*np.sqrt(np.abs(tau[tau[:,1]<0,:])))/(f02+b02*np.abs(tau[tau[:,1]<0,:])),2),nf)
	temp[tau[:,1]>0,:]=(1./np.power(1+np.power((f[tau[:,1]>0,:]-B*np.sqrt(tau[tau[:,1]>0,:]))/(f01+b01*np.abs(tau[tau[:,1]>0,:])),2),nf))*(A/np.power(1+np.power(tau[tau[:,1]>0,:]/t01,2),nt1))+(A/np.power(1+np.power(tau[tau[:,1]>0,:]/t02,2),nt2))*A2/np.power(1+np.power((f[tau[:,1]>0,:]+B*np.sqrt(tau[tau[:,1]>0,:]))/(f02+b02*np.abs(tau[tau[:,1]>0,:])),2),nf)
	return(temp)

##Model for chi-square power spectrum from gaussian with holes
def fit_func(X,IFCM,P,fftG1,fftG2,fftGF,fftGB,fftD1,fftD2,fftD3,fftDF12,fftDF23,fftDB32,fftDB21):
	CG=pow_arr2D2(X,*P[1:])
	CC=gauss_to_chi(CG,fftG1,fftG2,fftGF,fftGB)+P[0]
	CC[0,0]=fftG1.shape[0]*fftG1.shape[1]
	pow_holes(CC,IFCM,fftD1,fftD2,fftD3,fftDF12,fftDF23,fftDB32,fftDB21)


##Cost function to be minimized in fitting
def costfunc(P,C,X,IFCM,fftG1,fftG2,fftGF,fftGB,fftD1,fftD2,fftD3,fftDF12,fftDF23,fftDB32,fftDB21,tau_skip=1):
	fit_func(X,IFCM,P,fftG1,fftG2,fftGF,fftGB,fftD1,fftD2,fftD3,fftDF12,fftDF23,fftDB32,fftDB21)
	temp=np.log(fftD3[:,1:])+(C/fftD3[:,1:])
	return(np.real(np.sum(temp[tau_skip:,:])))

def costfuncGlob(P,C,X,IFCM,fftG1,fftG2,fftGF,fftGB,fftD1,fftD2,fftD3,fftDF12,fftDF23,fftDB32,fftDB21):
	CG=pow_arr2D2(X,*P[1:])
	CC=gauss_to_chi(CG,fftG1,fftG2,fftGF,fftGB)+P[0]
	CC[0,1:]=0
	CC[0,0]=fftG1.shape[0]*fftG1.shape[1]
	temp=0
	for i in range(len(C)):
		pow_holes(CC,IFCM[i],fftD1,fftD2,fftD3,fftDF12,fftDF23,fftDB32,fftDB21)
		temp+=np.sum((np.log(fftD3[:,1:])+(C[i]/fftD3[:,1:]))[1:,:])
	print(clock(),P,temp)
	return(temp)
def costfuncHR(P,C,X,IFCM,nf,nt,CA,IFCRES,n):
	C2=fit_funcHR(X,IFCM,nf,nt,P,CA,IFCRES,n)[:,1:]
	temp=np.log(C2)+(C/C2)
	return(np.sum(temp[1:,:]))

##Find end points of gaps and data
def bnd_find(sigma,nt):
	starts=np.array([],dtype=int)
	stops=np.array([],dtype=int)
	for i in range(nt-1):
		if sigma[i]==0 and sigma[i+1]>0:
		    starts=np.append(starts,i+1)
		if sigma[i]>0 and sigma[i+1]==0:
		    stops=np.append(stops,i)
	if sigma[0]!=0:
		starts=np.append(np.array([0]),starts)
	if sigma[nt-1]!=0:
		stops=np.append(stops,nt-1)
	return(starts.astype(int),stops.astype(int))

##Determine Masks in time and frequency
def mask(dspec,lt=50,lf=20):
	mt=np.ones(dspec.shape[1])
	mf=np.ones(dspec.shape[0])
	m=np.mean(dspec[:,np.std(dspec,axis=0)>0],axis=1)
	mf[m<m.mean()-m.std()]=0
	m=np.mean(dspec[mf==1,:],axis=0)
	mt[m<m[np.std(dspec,axis=0)>0].mean()-m[np.std(dspec,axis=0)>0].std()]=0
	starts,stops=bnd_find(mf,mf.shape[0])
	for i in range(starts.shape[0]):
		if stops[i]-starts[i]<2*lf:
			mf[starts[i]-1:stops[i]+1]=0
	starts,stops=bnd_find(mt,mt.shape[0])
	for i in range(starts.shape[0]):
		if stops[i]-starts[i]<2*lt:
			mt[starts[i]-1:stops[i]+1]=0
	starts,stops=bnd_find(mf,mf.shape[0])
	x=np.linspace(0,lf-1,lf)
	for i in range(stops.shape[0]):
		mf[starts[i]:starts[i]+lf]*=(1-np.cos(np.pi*(x/lf)))/2
		mf[stops[i]-lf+1:stops[i]+1]*=(1+np.cos(np.pi*(x/lf)))/2
	starts,stops=bnd_find(mt,mt.shape[0])
	x=np.linspace(0,lt-1,lt)
	for i in range(stops.shape[0]):
		mt[starts[i]:starts[i]+lt]*=(1-np.cos(np.pi*(x/lt)))/2
		mt[stops[i]-lt+1:stops[i]+1]*=(1+np.cos(np.pi*(x/lt)))/2
	return(mf,mt)

##Singular Value Decomposition Normalization
def svd(chi,nf,nt):
	U,sig,V=np.linalg.svd(chi)
	SIG=np.zeros((nf,nt))
	SIG[0,0]=sig[0]
	temp=np.matmul(U,np.matmul(SIG,V))
	return(chi/temp)

##Improved Masking and Normalization Code
def NormMask(dspec,lf,lt):
	##Cut out very noisy bins
	idx_t=np.linspace(0,dspec.shape[1]-1,dspec.shape[1],dtype=int)
	idx_f=np.linspace(0,dspec.shape[0]-1,dspec.shape[0],dtype=int)
	dspec2=dspec[:,np.mean(dspec,axis=0)>np.std(dspec,axis=0)/2]
	idx_t=idx_t[np.mean(dspec,axis=0)>np.std(dspec,axis=0)/2]
	idx_f=idx_f[np.mean(dspec2,axis=1)>np.std(dspec2,axis=1)/2]
	dspec2=dspec2[np.mean(dspec2,axis=1)>np.std(dspec2,axis=1)/2,:]

	##SVD
	u,s,w = np.linalg.svd(dspec2)
	s[2:] = 0.0
	S = np.zeros([len(u), len(w)], np.complex128)
	S[:len(s), :len(s)] = np.diag(s)
	model = np.dot(np.dot(u, S), w)

	##Cut out bins the behave poorly under SVD
	idx_t=idx_t[np.std(dspec2/model.real,0)<2]
	idx_f=idx_f[np.std(dspec2/model.real,1)<2]
	dspec3=dspec2[np.std(dspec2/model.real,1)<2,:]
	dspec3=dspec3[:,np.std(dspec2/model.real,0)<2]

	##SVD on good bins only
	u,s,w = np.linalg.svd(dspec3)
	s[1:] = 0.0
	S = np.zeros([len(u), len(w)], np.complex128)
	S[:len(s), :len(s)] = np.diag(s)
	model = np.dot(np.dot(u, S), w)

	##Form full SVD (set to 1 in gaps/bad bins)
	svd_gaps=np.ones(dspec.shape)
	for i in range(idx_f.shape[0]):
		svd_gaps[idx_f[i],idx_t]=model[i,:]

	##Setup 1d masks
	mt=np.zeros(dspec.shape[1])
	mt[idx_t]=1
	mf=np.zeros(dspec.shape[0])
	mf[idx_f]=1   
	##Remove short unmasked sections
	starts,stops=bnd_find(mf,mf.shape[0])
	for i in range(starts.shape[0]):
		if stops[i]-starts[i]<2*lf+2:
			mf[starts[i]:stops[i]+1]=0
	starts,stops=bnd_find(mt,mt.shape[0])
	for i in range(starts.shape[0]):
		if stops[i]-starts[i]<2*lt+2:
			mt[starts[i]:stops[i]+1]=0
	return(mt,mf,svd_gaps)

##Apply transfer function to input lens
def FTR(FL,FTS,FTC,N,nMF,nT,scal):
	FTR=np.sum(((np.real(FL)/scal)*(FTS.T)).T,axis=0)+np.sum(((np.imag(FL)/scal)*(FTC.T)).T,axis=0)
	NTRT=(np.sum((np.real(FL)/scal)**2)+np.sum((np.imag(FL)/scal)**2))*N/nT
	NTRMF=(np.sum((np.real(FL)/scal)**2)+np.sum((np.imag(FL)/scal)**2))*N/nMF
	return(FTR,NTRT,NTRMF)
