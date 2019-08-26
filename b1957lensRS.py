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
parser.add_argument('-samp',type=int,help='MCMC sample Number')

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


np.save('%s%sMaskT.npy' %(dr,fname),mt)
np.save('%s%sMaskF.npy' %(dr,fname),mf)

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

MCMC=np.load('%sSamples.npz' %args.f)
date_idx=np.argwhere(MCMC['dates']==args.d)[0]
glob_fit=np.zeros(13)
glob_fit[np.array([0,1,2,3,4,5,10])]=MCMC['samps'][:,args.samp,date_idx*7:(date_idx+1)*7].mean(0)
glob_fit[np.array([6,7,8,9,11,12])]=MCMC['samps'][:,args.samp,-6:].mean(0)
CG=fitmod.pow_arr2D2(X,*glob_fit[1:])
CC=np.copy(fitmod.gauss_to_chi(CG,fft_G1,fft_G2,fft_object_GF,fft_object_GB))
N=glob_fit[0]

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

np.save('%s%sLens_%s.npy' % (dr,fname,args.samp),L)
np.save('%s%sA_%s.npy' % (dr,fname,args.samp),A)
np.save('%s%sK2_%s.npy' % (dr,fname,args.samp),K2)

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
plt.savefig('%s%sPowComp_%s.png' % (dr,fname,args.samp))
plt.clf()
plt.imshow(dspec*mf[:,np.newaxis]*mt[np.newaxis,:])
plt.savefig('%s%sdspec.png' % (dr,fname))