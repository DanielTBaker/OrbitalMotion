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

import numpy as np
import fitmod
import argparse
import weight
from scipy.optimize import minimize,curve_fit
import sys
import pyfftw
import os
import re
import pickle as pkl
from scipy.integrate import simps


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
parser.add_argument('-km', dest='km', action='store_true',help='Known Mask')
parser.add_argument('-th', type=int,default = 8,help='Number of Threads')
parser.add_argument('-al',type=float,default=0,help='Global Frequency Scaling')

parser.set_defaults(km=False)
parser.set_defaults(T=False)
parser.set_defaults(fit=True)
args=parser.parse_args()
fname=args.f+str(args.d)
sz=np.load(fname+'dspec.npy').shape
nf=np.min((sz[0],args.bf)).astype(int)
nt=np.min((sz[1],args.bt)).astype(int)

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
fft_FW = pyfftw.empty_aligned((int(nf/2)+1,nt),dtype='complex128')
fft_FG = pyfftw.empty_aligned((int(nf/2)+1,nt),dtype='complex128')
fft_W = pyfftw.empty_aligned((int(nf/2)+1,nt),dtype='complex128')
fft_G = pyfftw.empty_aligned((int(nf/2)+1,nt),dtype='complex128')
fft_object_IFW = pyfftw.FFTW(fft_FW,fft_W, axes=(1,), direction='FFTW_BACKWARD',threads=nthread)
fft_object_IFW2 = pyfftw.FFTW(fft_FG,fft_G, axes=(1,), direction='FFTW_BACKWARD',threads=nthread)
fft_object_IFW3 = pyfftw.FFTW(fft_FW,fft_dspec2, axes=(1,), direction='FFTW_BACKWARD',threads=nthread)
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


np.save('AL0/'+fname+'MaskT.npy',mt)
np.save('AL0/'+fname+'MaskF.npy',mf)

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

fft_dspec1[:]=np.copy(dspec)*mf[:,np.newaxis]
fft_object_dspecF12()
F[:]=np.copy(fft_dspec2)/np.sqrt(nf)

##Try to load global fit file if it exits or use best guess
print(str(4)+'Fit Start')
sys.stdout.flush()
try:
	popt2=np.load(args.f+'GlobFit.npy')
	globf=True
	meth='Nelder-Mead'
except:
	Ng=C_data[1:,np.abs(freq)>.25].mean()/(mf.mean()*nt.mean())
	Ngscal=Ng*((mf.mean()*nt.mean()))
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

	CMXp=np.zeros(nf/2)
	CMXn=np.zeros(nf/2)
	for i in range(int(nf/2)):
		CMXp[i]=C_data[i+1,freq==fmaxp[i]]-Ngscal
		CMXn[i]=C_data[i+1,freq==fmaxn[i]]-Ngscal
	A2g=(CMXn/CMXp)[rtfit(tau2,*popt)>freq[5]].mean()
	def specfit(x,A,B):
		return(1/np.power(1+np.power(x/A,2),B))
	popt,pcov=curve_fit(specfit,tau2[1:],(CMXp/CMXp[:20].mean())[1:],sigma=np.sqrt((CMXp/CMXp[:20].mean())[1:]),p0=np.array([.5,2]),maxfev=10000)
	t01g=popt[0]/2
	nt1g=popt[1]
	popt,pcov=curve_fit(specfit,tau2[1:],(CMXn/CMXn[:20].mean())[1:],sigma=np.sqrt((CMXn/CMXn[:20].mean())[1:]),p0=np.array([.5,2]),maxfev=10000)
	t02g=popt[0]/2
	nt2g=popt[1]
	temp=np.mean(C_data[1:10,1:],axis=0)-Ngscal
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
	temp2=fitmod.gauss_to_chi(temp,nf,nt)
	temp3=fitmod.pow_holes(temp2+popt2[0],IFCM)
	Ag*=np.sqrt((C_data[2:100,1:20].mean()-Ngscal)/(temp3[2:100,1:20].mean()-Ngscal))
	popt2=np.array([Ng,Ag,t01g,nt1g,t02g,nt2g,nfg,f01g,f02g,Bg,A2g,b01g,b02g])
	meth='Nelder-Mead'
	globf=False

##Check if Fitting required
if args.fit:
	##Fit and export
	bnds=((0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(None,None),(None,None))
	res=minimize(fitmod.costfunc,popt2,args=(C_data[:,1:],X,IFCM,fft_G1,fft_G2,fft_object_GF,fft_object_GB,fft_dspec1,fft_dspec2,fft_dspec3,fft_object_dspecF12,fft_object_dspecF23,fft_object_dspecB32,fft_object_dspecB21),bounds=bnds,method=meth)
	popt=res.x
	np.save(fname+'Fit.npy',popt)
else:
	##Use fit file
	popt=np.load(fname+'Fit.npy')
##Exit if no global fit
if not globf:
	sys.exit('No Global Fit')
N=popt[0]

##Theoretical Powers
##Note that these are the powers WITHOUT the effect of gaps as we wish to compare them locally
CG=fitmod.pow_arr2D2(X,*popt[1:])
CC=np.copy(fitmod.gauss_to_chi(CG,fft_G1,fft_G2,fft_object_GF,fft_object_GB))
temp=np.fft.fftshift(CC)
sigscal=np.sqrt(simps(simps(temp,dx=1/nt,axis=1),dx=1/nf,axis=0))
#sigscal=np.sqrt(CC[1,1])

N/=sigscal**2
##Theoretical Powers
fits=np.array([re.findall('b1957-[0-9]+Fit.npy',f) for f in os.listdir('./') if re.search('[0-9]+Fit.npy',f)])[:,0]
CG=np.zeros(CG.shape)
X=(tarr,farr)
for fit in fits:
	popt2=np.load(fit)
	CG+=fitmod.pow_arr2D2(X,*popt2[1:])
CC=np.copy(fitmod.gauss_to_chi(CG,fft_G1,fft_G2,fft_object_GF,fft_object_GB))
temp=np.fft.fftshift(CC)
sigscal=np.sqrt(simps(simps(temp,dx=1/nt,axis=1),dx=1/nf,axis=0))
CG=np.zeros(CG.shape)
X=(tarr,farr/(1.+args.al))
for fit in fits:
	popt2=np.load(fit)
	CG+=fitmod.pow_arr2D2(X,*popt2[1:])
CG/=sigscal*((1+args.al))

CC=np.zeros((1+int(nf/2),nt))
CC[:]=fitmod.gauss_to_chi(CG,fft_G1,fft_G2,fft_object_GF,fft_object_GB)+N
CC[0,1:]=0
CC[0,0]=nf*nt
fitmod.pow_holes(CC,IFCM,fft_dspec1,fft_dspec2,fft_dspec3,fft_object_dspecF12,fft_object_dspecF23,fft_object_dspecB32,fft_object_dspecB21)
CT=np.zeros(CC.shape)
CT[:]=fft_dspec3[:]
CC[:]=fitmod.gauss_to_chi(CG,fft_G1,fft_G2,fft_object_GF,fft_object_GB)
CC[0,1:]=0
CC[0,0]=nf*nt
CCR=np.copy(CC)
CCR[:,1:]=CC[:,1:][:,::-1]
CTR=np.copy(CT)
CTR[:,1:]=CT[:,1:][:,::-1]

##Cutoff frequencies introudced for each tau
fcutA=np.ones(CC.shape)
for i in range(CC.shape[0]):
    if CC[i,freq<0].max()>N/2:
        flow=freq[freq<0][CC[i,freq<0]>N/2].min()
    else:
        flow=0
    if CC[i,freq>0].max()>N/2:
        fhigh=freq[freq>0][CC[i,freq>0]>N/2].max()
    else:
        fhigh=0
    fcutA[i,np.abs(freq)>max(np.abs(flow),fhigh)]=0
fcutArev=np.copy(fcutA)
fcutArev[:,1:]=fcutA[::,1:][:,::-1]

##Determine A(L)
AL=weight.AL_find2(CC,CT,CCR,CTR,freq,fcutA,fcutArev,fft_dspec2,fft_dspec3,fft_object_dspecF23,fft_object_dspecB32)
np.save(fname+'A.npy',AL)


##Recover Lens (Split estimator integral into two convolutions)
F_rev=np.empty(F.shape,dtype='complex')
F_rev[:,0]=np.conjugate(F[:,0])
F_rev[:,1:]=np.conjugate(F[:,::-1][:,:nt-1])
fft_FW[:]=2j*np.pi*freq*CC*F/CT
#fft_FW*=fcutA
fft_object_IFW3()
#print(fft_dspec2.max())
fft_FG[:]=F_rev/CTR
#fft_FG*=fcutArev
fft_object_IFW2()
fft_dspec2[:]*=fft_G*nt
#print(fft_dspec2.max())
fft_FW[:]=2j*np.pi*freq*CCR*F_rev/CTR
#fft_FW*=fcutArev
fft_object_IFW()
#print(fft_W.max())
fft_FG[:]=F/CT
#fft_FG*=fcutA
fft_object_IFW2()
#print(fft_G.max())
fft_dspec2[:]+=fft_G*fft_W*nt
fft_object_dspecF23()
fft_dspec3*=2j*np.pi*freq*AL/np.sqrt(nt)
fft_object_dspecB32()
fft_dspec2*=-np.sqrt(nt)

np.save(fname+'Lens.npy',np.real(fft_dspec2))


