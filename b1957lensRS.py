"""
Lens Recovery Code -Generalized-
Daniel Baker
Last Edited 01/04/2018  

Arguments:
-f Prefix for dspec files (same for all obersations of interest)
-d Specify specific dataset (dspec filename= fd.npy)
-T Transpose dspec after loading
-bt Number of time bins
-bf Number of frequency channels
-l Smoothing length for mask
-nof Skip fitting
"""

import pyfftw
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import fitmod
import argparse
import weight
from scipy.optimize import minimize,curve_fit
import sys
from scipy.integrate import simps
from scipy.ndimage.filters import convolve1d
import os
import re
import pickle as pkl

##Command Line Arguments
parser = argparse.ArgumentParser(description='Lens Recovery Code for B1957')
parser.add_argument('-d', type=int, help='Observation Date')
parser.add_argument('-nof', dest='fit', action='store_false', help='Skip Fitting')
parser.add_argument('-f',type=str,help='fname prefix')
parser.add_argument('-T',dest='T',action='store_true',help='Transpose dspec')
parser.add_argument('-bt',type=float,default=np.inf,help='Number of Time Bins')
parser.add_argument('-bf',type=float,default=np.inf,help='Number of Frequncy Bins')
parser.add_argument('-lt', type=int,default = 50,help='Mask Smoothing Length in Time')
parser.add_argument('-lf', type=int,default = 20,help='Mask Smoothing Length in Frequency')
parser.add_argument('-th', type=int,default = 8,help='Number of Threads')
parser.add_argument('-G',dest='G',action='store_true',help='Use Global Fiducial C')
parser.add_argument('-al',type=float,default=0,help='Global Frequency Scaling')

parser.set_defaults(T=False)
parser.set_defaults(G=False)
parser.set_defaults(fit=True)
args=parser.parse_args()
fname=args.f+str(args.d)
sz=np.load(fname+'dspec.npy').shape
if args.T:
	nf=np.min((sz[1],args.bf)).astype(int)
	nt=np.min((sz[0],args.bt)).astype(int)
else:
	nf=np.min((sz[0],args.bf)).astype(int)
	nt=np.min((sz[1],args.bt)).astype(int)

if args.al==0:
	dr='./AL0/'
	if not os.path.isdir(dr):
		os.makedirs(dr)
else:
	dr='./AL%s/' % format(args.al,'.3e')
	if not os.path.isdir(dr):
		os.makedirs(dr)

print('Parsed',flush=True)

nthread=args.th

print('Start FFT Setup',flush=True)
##Check if wisdom exists to speed the ffts
try:
	pyfftw.import_wisdom(pkl.load(open('pyfftwwis-1-%s.pkl' % args.th,'rb')))
except:
	print('No Wisdom',flush=True)
##Initialize ffts
fft_G1= pyfftw.empty_aligned((nf,nt), dtype='complex128')
fft_G2= pyfftw.empty_aligned((nf,nt), dtype='complex128')
fft_object_GF = pyfftw.FFTW(fft_G1,fft_G2, axes=(0,1), direction='FFTW_FORWARD',threads=nthread)
fft_object_GB = pyfftw.FFTW(fft_G2,fft_G1, axes=(1,0), direction='FFTW_BACKWARD',threads=nthread)
fft_object_GBA0 = pyfftw.FFTW(fft_G2,fft_G1, axes=(0,), direction='FFTW_BACKWARD',threads=nthread)
fft_dspec1 = pyfftw.empty_aligned((nf,nt),dtype='float64')
fft_dspec2 = pyfftw.empty_aligned((int(nf/2)+1,nt),dtype='complex128')
fft_dspec3 = pyfftw.empty_aligned((int(nf/2)+1,nt),dtype="complex128")
fft_object_dspecF12=pyfftw.FFTW(fft_dspec1,fft_dspec2, axes=(0,), direction='FFTW_FORWARD',threads=nthread)
fft_object_dspecF23=pyfftw.FFTW(fft_dspec2,fft_dspec3, axes=(1,), direction='FFTW_FORWARD',threads=nthread)
fft_object_dspecB32=pyfftw.FFTW(fft_dspec3,fft_dspec2, axes=(1,), direction='FFTW_BACKWARD',threads=nthread)
fft_object_dspecB21=pyfftw.FFTW(fft_dspec2,fft_dspec1, axes=(0,), direction='FFTW_BACKWARD',threads=nthread)
pkl.dump(pyfftw.export_wisdom(),open('pyfftwwis-1-%s.pkl' % args.th,'wb'))

print('FFT Setup Complete',flush=True)

C_data=np.zeros(fft_dspec3.shape)
F=np.zeros(fft_dspec3.shape,dtype=complex)


##Import Dynamic Spectrum
if args.T:
	dspec=(np.load(fname+'dspec.npy')[:nt,:nf]).T
else:
	dspec=(np.load(fname+'dspec.npy')[:nf,:nt])

##variable arrays for fitting
freq=np.fft.fftfreq(nt)
tau=np.fft.fftfreq(nf)

farr=np.ones((tau.shape[0],freq.shape[0]))*freq
tarr=np.ones((tau.shape[0],freq.shape[0]))*tau[:,np.newaxis]
X=(tarr,farr)

##Determine mask for missing time bins and reweight dspec (via svd)) to account for variations in gain
print('Find Mask',flush=True)
mt,mf,dspec=fitmod.NormMask(dspec,args.lf,args.lt)
t=np.linspace(1,args.lf,args.lf)
start,stop=fitmod.bnd_find(mf,mf.shape[0])
for i in range(start.shape[0]):
    mf[start[i]:start[i]+args.lf]=(1-np.cos(np.pi*t/(args.lf+1)))/2
    mf[stop[i]-args.lf+1:stop[i]+1]=(np.cos(np.pi*(t)/(args.lf+1))+1)/2
t=np.linspace(1,args.lt,args.lt)
start,stop=fitmod.bnd_find(mt,mt.shape[0])
for i in range(start.shape[0]):
    mt[start[i]:start[i]+args.lt]=(1-np.cos(np.pi*t/(args.lt+1)))/2
    mt[stop[i]-args.lt+1:stop[i]+1]=(np.cos(np.pi*(t)/(args.lt+1))+1)/2


np.save(dr+fname+'MaskT.npy',mt)
np.save(dr+fname+'MaskF.npy',mf)

print('Find IFCM',flush=True)
fft_G1[:]=mf[:,np.newaxis]*mt[np.newaxis,:]
##Correlation function of mask for use in conversions
fft_object_GF()
fft_G2*=np.conjugate(fft_G2)/(nf*nt)
fft_object_GB()
IFCM=np.copy(np.real(fft_G1))

##Find Power Spectrum
print('Caclulate Power',flush=True)
fft_dspec1[:]=np.copy(dspec)*mf[:,np.newaxis]*mt[np.newaxis,:]
fft_object_dspecF12()
fft_object_dspecF23()
C_data[:]=np.copy(np.abs(fft_dspec3[:]/np.sqrt(nf*nt))**2)

fft_dspec1[:]=np.copy(dspec)
fft_object_dspecF12()
F[:]=np.copy(fft_dspec2)/np.sqrt(nf)

##Try to load global fit file if it exits or use best guess
print('Fit Start',flush=True)
tau2=np.fft.rfftfreq(nf)
try:
	popt2=np.load(fname+'Fit.npy')
	globf=True
	meth='Nelder-Mead'
	CG=fitmod.pow_arr2D2(X,*popt2[1:])
	CC=np.zeros((1+int(nf/2),nt))
	CC[:]=fitmod.gauss_to_chi(CG,fft_G1,fft_G2,fft_object_GF,fft_object_GB)+popt2[0]
	CC[0,1:]=0
	CC[0,0]=nf*nt
	fitmod.pow_holes(CC,IFCM,fft_dspec1,fft_dspec2,fft_dspec3,fft_object_dspecF12,fft_object_dspecF23,fft_object_dspecB32,fft_object_dspecB21)
	CT=np.zeros(CC.shape)
	CT[:]=fft_dspec3[:]
	popt2[0]*=(C_data/CT)[np.abs(tau2)>3*tau2.max()/4,:][:,np.abs(freq)>3*freq.max()/4].mean()
	print('Parameters Loaded',flush=True)
except:
	Ng=C_data[1:,np.abs(freq)>.25].mean()/(mf.mean()*mt.mean())
	Ngscal=Ng*((mf.mean()*mt.mean()))
	Ag=np.sqrt(C_data[1:10,1:10].mean()-Ngscal)*100
	fmaxp=np.zeros(C_data.shape[0]-1)
	fmaxn=np.zeros(C_data.shape[0]-1)
	for i in range(int(nf/2)):
	    fmaxp[i]=freq[freq>0][C_data[i+1,freq>0]==C_data[i+1,freq>0].max()]
	    fmaxn[i]=freq[freq<0][C_data[i+1,freq<0]==C_data[i+1,freq<0].max()]
	def rtfit(x,A):
	    return(A*np.sqrt(x))
	tau2=np.fft.rfftfreq(nf)[1:]
	popt,pcov=curve_fit(rtfit,tau2[tau2<.2],(np.abs(fmaxn[tau2<.2])+fmaxp[tau2<.2])/2)
	Bg=popt[0]

	CMXp=np.zeros(int(nf/2))
	CMXn=np.zeros(int(nf/2))
	for i in range(int(nf/2)):
	    CMXp[i]=C_data[i+1,freq==fmaxp[i]]-Ngscal
	    CMXn[i]=C_data[i+1,freq==fmaxn[i]]-Ngscal
	A2g=(CMXn/CMXp)[rtfit(tau2,*popt)>freq[5]].mean()
	def specfit(x,A,B):
	    return(1/np.power(1+np.power(x/A,2),B))
	popt,pcov=curve_fit(specfit,tau2[1000:],(CMXp/CMXp[:20].mean())[1000:],sigma=np.sqrt((CMXp/CMXp[:20].mean())[1000:]),p0=np.array([.5,2]),maxfev=10000)
	t01g=popt[0]/2
	nt1g=popt[1]
	popt,pcov=curve_fit(specfit,tau2[1000:],(CMXn/CMXn[:20].mean())[1000:],sigma=np.sqrt((CMXn/CMXn[:20].mean())[1000:]),p0=np.array([.5,2]),maxfev=10000)
	t02g=popt[0]/2
	nt2g=popt[1]
	temp=np.mean(C_data[500:600,1:],axis=0)-Ngscal
	temp/=(temp[1:10].mean())
	ftemp=freq[1:][temp>0]
	popt,pcov=curve_fit(specfit,ftemp,temp[temp>0],sigma=np.sqrt(temp[temp>0]),p0=np.array([.5,2]))
	nfg=popt[1]
	f01g=popt[0]/2
	f02g=popt[0]/2
	b01g=0
	b02g=0
	popt2=np.array([Ng,Ag,t01g,nt1g,t02g,nt2g,nfg,f01g,f02g,Bg,A2g,b01g,b02g])
	temp=fitmod.pow_arr2D2(X,*popt2[1:])
	temp2=fitmod.gauss_to_chi(temp,fft_G1,fft_G2,fft_object_GF,fft_object_GB)+popt2[0]
	temp2[0,:]=0
	temp2[:,0]=0
	temp2[0,0]=nf*nt
	fitmod.pow_holes(temp2,IFCM,fft_dspec1,fft_dspec2,fft_dspec3,fft_object_dspecF12,fft_object_dspecF23,fft_object_dspecB32,fft_object_dspecB21)
	temp3=np.copy(fft_dspec3[:])
	Ag*=np.sqrt((C_data[1000:2000,1:20].mean()-Ngscal)/(temp3[1000:2000,1:20].mean()-Ngscal))
	popt2=np.array([Ng,Ag,t01g,nt1g,t02g,nt2g,nfg,f01g,f02g,Bg,A2g,b01g,b02g])
	meth='TNC'
	globf=False

##Check if Fitting required
if args.fit:
	##Fit and export
	bnds=((0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(None,None),(None,None))
	res=minimize(fitmod.costfunc,popt2,args=(C_data[:,1:],X,IFCM,fft_G1,fft_G2,fft_object_GF,fft_object_GB,fft_dspec1,fft_dspec2,fft_dspec3,fft_object_dspecF12,fft_object_dspecF23,fft_object_dspecB32,fft_object_dspecB21,1),bounds=bnds,method=meth)
	popt=res.x
	np.save(fname+'Fit.npy',popt)
	x=np.linspace(-1e-3,1e-3,7)
	x+=1
	y=np.zeros((popt.shape[0],7))
	for i in range(popt.shape[0]):
		ptemp=np.copy(popt)
		for k in range(7):
			ptemp[i]=popt[i]*x[k]
			y[i,k]=fitmod.costfunc(ptemp,C_data[:,1:],X,IFCM,fft_G1,fft_G2,fft_object_GF,fft_object_GB,fft_dspec1,fft_dspec2,fft_dspec3,fft_object_dspecF12,fft_object_dspecF23,fft_object_dspecB32,fft_object_dspecB21)
	np.save(fname+'FitShape.npy',y)
else:
	##Use fit file
	popt=np.load(fname+'Fit.npy')

##Exit if no global fit
if not globf:
	sys.exit('No Global Fit')
N=popt[0]
popt2=np.load(args.f+'GlobFit.npy')
print('Theoretical Powers',flush=True)
##Theoretical Powers
##Note that these are the powers WITHOUT the effect of gaps as we wish to compare them locally
fft_G1[:]=(np.ones((nt,nf))*mf).T
##Correlation function of mask for use in conversions
fft_object_GF()
fft_G2*=np.conjugate(fft_G2)/(nf*nt)
fft_object_GB()
IFCM2=np.copy(np.real(fft_G1))

CG=fitmod.pow_arr2D2(X,*popt[1:])
CCL=np.copy(fitmod.gauss_to_chi(CG,fft_G1,fft_G2,fft_object_GF,fft_object_GB))

if args.G:
	temp=np.fft.fftshift(CCL,axes=1)
	sigscal=np.sqrt(simps(simps(temp,dx=1/nt,axis=1),dx=1/nf,axis=0))
	CCL/=sigscal**2
	#sigscal=np.sqrt(CC[1,1])
	N/=sigscal**2
	F/=sigscal
	C_data/=sigscal**2
	fit2=np.load(fname+'Fit2.npy')
	CG=np.zeros(CG.shape)
	X=(tarr,farr/(1.+args.al))
	CG=fitmod.pow_arr2D2(X,*fit2[1:])/sigscal
	CC=np.copy(fitmod.gauss_to_chi(CG,fft_G1,fft_G2,fft_object_GF,fft_object_GB))
else:
	print('Daily Fit Only',flush=True)
#fft_dspec3[:]=CC
#fft_object_dspecB32()
#fft_object_dspecB21()
#fft_dspec1*=IFCM2
#fft_object_dspecF12()
#fft_object_dspecF23()
#CC2=np.copy(np.real(fft_dspec3))
#fft_dspec3[:]=CC+N
#fft_object_dspecB32()
#fft_object_dspecB21()
#fft_dspec1*=IFCM2
#fft_object_dspecF12()
#fft_object_dspecF23()
#CCN=np.copy(np.real(fft_dspec3))


nw=args.lt
CP=np.fft.ifftshift(np.gradient(np.fft.fftshift(CC,axes=1),freq[1],axis=1),axes=1)
FK0=(CC/np.power(CC+N,2))*(1+(freq*CP/CC))
FK0[CC==0]=0
K0=np.fft.ifft(FK0,axis=1)
K2=np.roll(K0,nw,axis=1)[:,:2*nw+1]
temp=FK0*CC*(1+(freq*CP/CC))
temp[CC==0]=0
A=1/(simps(np.fft.fftshift(temp,axes=1),dx=freq[1])*2*np.pi*nt)


F2=np.zeros(F[1:,:].shape)
for i in range(F2.shape[0]):
	F2[i,:]=convolve1d(np.real(F[i+1,:]),np.real(K2[i+1,:]))+convolve1d(np.imag(F[i+1,:]),np.imag(K2[i+1,:]))+1j*(convolve1d(np.real(F[i+1,:]),np.imag(K2[i+1,:]))-convolve1d(np.imag(F[i+1,:]),np.real(K2[i+1,:])))

L=np.real(F[1:,:]*F2*(A[1:][:,np.newaxis]))

np.save('%s%sLens.npy' % (dr,fname),L)
np.save('%s%sA.npy' % (dr,fname),A)
np.save('%s%sK2.npy' % (dr,fname),K2)

fft_dspec3[:]=CC+N
fft_object_dspecB32()
fft_object_dspecB21()
fft_dspec1*=IFCM
fft_object_dspecF12()
fft_object_dspecF23()
for i in range(3):
	if i==0:
		plt.loglog(freq,np.mean(C_data[1:1000,:],axis=0),'r',label='Data f >0')
		plt.loglog(-freq,np.mean(C_data[1:1000,:],axis=0),'g',label='Data f >0')
		plt.loglog(freq,np.mean(fft_dspec3[1:1000,:],axis=0),'b',label='Fit f >0')
		plt.loglog(-freq,np.mean(fft_dspec3[1:1000,:],axis=0),'y',label='Fit f >0')
	else:
		plt.loglog(freq,np.mean(C_data[1000*i:1000*(i+1),:],axis=0),'r')
		plt.loglog(-freq,np.mean(C_data[1000*i:1000*(i+1),:],axis=0),'g')
		plt.loglog(freq,np.mean(fft_dspec3[1000*i:1000*(i+1),:],axis=0),'b')
		plt.loglog(-freq,np.mean(fft_dspec3[1000*i:1000*(i+1),:],axis=0),'y')
plt.legend(loc=0)
plt.savefig('%s%sPowComp.png' % (dr,fname))
plt.clf()
plt.imshow(dspec*mf[:,np.newaxis]*mt[np.newaxis,:])
plt.savefig('%s%sdspec.png' % (dr,fname))

