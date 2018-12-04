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

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import fitmod
import argparse
import weight
from scipy.optimize import minimize,curve_fit
import sys
import pyfftw
from scipy.integrate import simps
from scipy.ndimage.filters import convolve1d
import os
import re
import pickle as pkl
from scipy.interpolate import interp1d 

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
nf=np.min((sz[0],args.bf)).astype(int)
nt=np.min((sz[1],args.bt)).astype(int)

if args.al==0:
	dr='./AL0/'
else:
	dr='./AL%s/' % format(args.al,'.3e')
	if not os.path.isdir(dr):
		os.makedirs(dr)

print('Parsed')
sys.stdout.flush()

def svd(chi,nf,nt):
	U,sig,V=np.linalg.svd(chi)
	SIG=np.zeros((nf,nt))
	SIG[0,0]=sig[0]
	temp=np.matmul(U,np.matmul(SIG,V))
	return(chi/temp)

nthread=args.th

print('Start FFT Setup')
sys.stdout.flush()
try:
	pyfftw.import_wisdom(pkl.load(open('pyfftwwis-1-%s.pkl' % args.th,'rb')))
except:
	print('No Wisdom')
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

print('FFT Setup Complete')
sys.stdout.flush()

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
tarr=np.ones((tau.shape[0],freq.shape[0])).T*tau
tarr=tarr.T
X=(tarr,farr)

##Determine mask for missing time bins and reweight dspec to account for variations in gain
print('Find Mask')
mf,mt,dspec=fitmod.NormMask(dspec,args.lf,args.lt)
t=np.linspace(1,args.lf,args.lf)
start,stop=fitmod.bnd_find(mf,mf.shape[0])
for i in range(start.shape[0]):
    mf[start[i]:start[i]+args.lf]=(1-np.cos(np.pi*t/args.lf))/2
    mf[stop[i]-args.lf+1:stop[i]+1]=(np.cos(np.pi*(t-1)/args.lf)+1)/2
t=np.linspace(1,args.lt,args.lt)
start,stop=fitmod.bnd_find(mt,mt.shape[0])
for i in range(start.shape[0]):
    mt[start[i]:start[i]+args.lt]=(1-np.cos(np.pi*t/args.lt))/2
    mt[stop[i]-args.lt+1:stop[i]+1]=(np.cos(np.pi*(t-1)/args.lt)+1)/2


np.save(dr+fname+'MaskT.npy',mt)
np.save(dr+fname+'MaskF.npy',mf)

print('Find IFCM')
sys.stdout.flush()
fft_G1[:]=(np.ones((nt,nf))*mf).T*mt
##Correlation function of mask for use in conversions
fft_object_GF()
fft_G2*=np.conjugate(fft_G2)/(nf*nt)
fft_object_GB()
IFCM=np.copy(np.real(fft_G1))

##Find Power Spectrum
print('Caclulate Power')
sys.stdout.flush()
fft_dspec1[:]=np.copy(dspec)*mf[:,np.newaxis]*mt[np.newaxis,:]
fft_object_dspecF12()
fft_object_dspecF23()
C_data[:]=np.copy(np.abs(fft_dspec3[:]/np.sqrt(nf*nt))**2)

fits=np.array([re.findall('b1957-[0-9]+Fit.npy',f) for f in os.listdir('./') if re.search('[0-9]+Fit.npy',f)])[:,0]
CG=np.zeros(farr.shape)
X=(tarr,farr/(1.+args.al))
for fit in fits:
	popt2=np.load(fit)
	CGT=fitmod.pow_arr2D2(X,*popt2[1:])
	CCT=np.fft.fftshift(np.copy(fitmod.gauss_to_chi(CGT,fft_G1,fft_G2,fft_object_GF,fft_object_GB)),axes=1)
	sigscal=np.sqrt(simps(simps(CCT,dx=1/nt,axis=1),dx=1/nf,axis=0))
	CG+=CGT/sigscal
CG/=fits.shape[0]*np.sqrt(1+args.al)

def scal_fit(P,CD,CG,freq,IFCM,fft_G1,fft_G2,fftF,fftB,fft1,fft2,fft3,fftF12,fftF23,fftB32,fftB21):
	inter=interp1d(np.fft.fftshift(freq),np.fft.fftshift(CG,axes=1),fill_value='extrapolate')
	CG2=np.fft.ifftshift(P[0]*inter(freq*P[1]),axes=1)
	CC=fitmod.gauss_to_chi(CG2,fft_G1,fft_G2,fftF,fftB)+P[3]
	fitmod.pow_holes(CC,IFCM,fft1,fft2,fft3,fftF12,fftF23,fftB32,fftB21)
	value=np.log(fft3[31:,1:])+(CD/fft3[31:,1:])
	return(np.real(np.sum(value)))

N=np.load(fname+'Fit.npy')[0]
res=minimize(scal_fit,np.array([1,1,N]),args=(C_data,CG,freq,IFCM,fft_G1,fft_G2,fft_object_GF,fft_object_GB,fft_dspec1,fft_dspec2,fft_dspec3,fft_object_dspecF12,fft_object_dspecF23,fft_object_dspecB32,fft_object_dspecB21),bounds=((0,None),(0,None)),method='Nelder-Mead')

def pltmake(C_data,CC_G,date,freq):
    plt.figure()
    plt.subplot(121)
    plt.loglog(freq,np.mean(C_data[1:1000,:],axis=0),'r')
    plt.loglog(freq,np.mean(CC_G[1:1000,:],axis=0),'k')
    plt.loglog(freq,np.mean(C_data[1000:2000,:],axis=0),'r')
    plt.loglog(freq,np.mean(CC_G[1000:2000,:],axis=0),'k')
    plt.loglog(freq,np.mean(C_data[2000:3000,:],axis=0),'r')
    plt.loglog(freq,np.mean(CC_G[2000:3000,:],axis=0),'k')
    plt.title(r'Fit Comparison 2014/06/%s $f_D$>0' %date)
    plt.ylabel('C (arbitrons)')
    plt.xlabel(r'$f_D$ (mHz)')
    plt.subplot(122)
    plt.loglog(-freq,np.mean(C_data[1:1000,:],axis=0),'r',label='Data')
    plt.loglog(-freq,np.mean(CC_G[1:1000,:],axis=0),'k',label='Daily Fit')
    plt.loglog(-freq,np.mean(C_data[1000:2000,:],axis=0),'r')
    plt.loglog(-freq,np.mean(CC_G[1000:2000,:],axis=0),'k')
    plt.loglog(-freq,np.mean(C_data[2000:3000,:],axis=0),'r')
    plt.loglog(-freq,np.mean(CC_G[2000:3000,:],axis=0),'k')
    plt.title(r'Fit Comparison 2014/06/%s $f_D$<0' %date)
    plt.ylabel('C (arbitrons)')
    plt.xlabel(r'$f_D$ (mHz)')
    plt.legend(loc=0)

np.save(fname+'Fit2.npy',res.x)

P=res.x
inter=interp1d(np.fft.fftshift(freq),np.fft.fftshift(CG,axes=1),fill_value='extrapolate')
CG2=np.fft.ifftshift(P[0]*inter(freq*P[1]),axes=1)
CC=fitmod.gauss_to_chi(CG2,fft_G1,fft_G2,fft_object_GF,fft_object_GB)+P[3]
fitmod.pow_holes(CC,IFCM,fft_dspec1,fft_dspec2,fft_dspec3,fft_object_dspecF12,fft_object_dspecF23,fft_object_dspecB32,fft_object_dspecB21)
C_Fit=np.copy(fft_dspec3)
pltmake(C_data,CC_G,str(args.d),freq)
plt.savefig('%s%sPowComp2.png' % (dr,fname))


