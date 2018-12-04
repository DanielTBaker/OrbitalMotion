"""
Simulated unlensed fields to determine mean field due to gaps
Daniel Baker
Last Edited 01/04/2018 

Arguments:
-f Prefix for dspec files (same for all obersations of interest)
-d Specify specific dataset (dspec filename= fd.npy)
-n number of simulations to run

Requires that Mask and estimator weights be known.
"""

import numpy as np
import argparse
import simmod
import fitmod
import weight
from mpi4py import MPI
import pyfftw
import sys
from scipy.ndimage.filters import convolve1d
from scipy.integrate import simps,cumtrapz
from scipy.interpolate import interp1d
import os
import re
import pickle as pkl

##Initialize Multiprocessing
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

ts=MPI.Wtime()


##Command Line Arguments
parser = argparse.ArgumentParser(description='Lens Recovery Code for B1957')
parser.add_argument('-d', type=int, help='Observation Date')
parser.add_argument('-n', type=int, help='Number of Simulations')
parser.add_argument('-f',type=str,help='fname prefix')
parser.add_argument('-th', type=int,default = 8,help='Number of Threads')
parser.add_argument('-nf', type=int,default = 6000,help='Number of Channels')
parser.add_argument('-nt', type=int,default = 2108,help='Number of Time Bins')
parser.add_argument('-G',dest='G',action='store_true',help='Use Global Fiducial C')
parser.add_argument('-al',type=float,default=0,help='Global Frequency Scaling')
parser.add_argument('-N', dest='N',action='store_true',help='Use Noise Weights')

parser.set_defaults(G=False)
parser.set_defaults(N=False)
args=parser.parse_args()
fname=args.f+str(args.d)
n=args.n

##Import Fit Parameters
popt=np.load(fname+'Fit.npy')
N=popt[0]
popt2=np.load(args.f+'GlobFit.npy')

nt=args.nt
nf=args.nf

sys.stdout.flush()
nthread=args.th

try:
	pyfftw.import_wisdom(pkl.load(open('pyfftwwis-%s-%s.pkl' % (size,args.th),'rb')))
except:
	if rank==0:
		print('No Wisdom')
fft_G1= pyfftw.empty_aligned((nf,nt), dtype='complex128')
fft_G2= pyfftw.empty_aligned((nf,nt), dtype='complex128')
fft_object_GF = pyfftw.FFTW(fft_G1,fft_G2, axes=(0,1), direction='FFTW_FORWARD',threads=nthread)
fft_object_GB = pyfftw.FFTW(fft_G2,fft_G1, axes=(1,0), direction='FFTW_BACKWARD',threads=nthread)
fft_dspec1 = pyfftw.empty_aligned((nf,nt),dtype='float64')
fft_dspec2 = pyfftw.empty_aligned((int(nf/2)+1,nt),dtype='complex128')
fft_dspec3 = pyfftw.empty_aligned((int(nf/2)+1,nt),dtype="complex128")
fft_object_dspecF12=pyfftw.FFTW(fft_dspec1,fft_dspec2, axes=(0,), direction='FFTW_FORWARD',threads=nthread)
fft_object_dspecF23=pyfftw.FFTW(fft_dspec2,fft_dspec3, axes=(1,), direction='FFTW_FORWARD',threads=nthread)
fft_object_dspecB32=pyfftw.FFTW(fft_dspec3,fft_dspec2, axes=(1,), direction='FFTW_BACKWARD',threads=nthread)
fft_object_dspecB21=pyfftw.FFTW(fft_dspec2,fft_dspec1, axes=(0,), direction='FFTW_BACKWARD',threads=nthread)
pkl.dump(pyfftw.export_wisdom(),open('pyfftwwis-%s-%s.pkl' % (size,args.th),'wb'))

if args.al==0:
	dr='./AL0/'
else:
	dr='./AL%s/' % format(args.al,'.3e')

##Import mask
mf=np.load(dr+fname+'MaskF.npy')
mt=np.load(dr+fname+'MaskT.npy')
mt0=np.copy(mt)
mt[mt>0]==1


##Divide simulations between processors 
delta=int(np.mod(n,size))
k=int((n-delta)/size)
lengs=np.append(np.ones(delta)*(k+1),np.ones(size-delta)*k).astype(int)
disp=(np.cumsum(lengs)-lengs)[rank]

##variable arrays for Theoretical Powers
freq=np.fft.fftfreq(nt)
tau=np.fft.fftfreq(nf)

farr=np.ones((nf,nt))*freq
tarr=np.ones((nf,nt)).T*tau
tarr=tarr.T
X=(tarr,farr)

CG=fitmod.pow_arr2D2(X,*popt[1:])
if args.G:
	CC=np.copy(fitmod.gauss_to_chi(CG,fft_G1,fft_G2,fft_object_GF,fft_object_GB))
	temp=np.fft.fftshift(CC)
	sigscal=np.sqrt(simps(simps(temp,dx=1/nt,axis=1),dx=1/nf,axis=0))
	#sigscal=np.sqrt(CC[1,1])
	N/=sigscal**2
	##Theoretical Powers
	fit2=np.load(fname+'Fit2.npy')
	CG=np.zeros(CG.shape)
	X=(tarr,farr/(1.+args.al))
	CG=fitmod.pow_arr2D2(X,*fit2[1:])/sigscal


##Import Estimator Variables
A=np.load('%s%sA.npy' % (dr,fname))
K2=np.load('%s%sK2.npy' % (dr,fname))
F2=np.zeros(fft_dspec2[1:,:].shape,dtype=complex)
Lt=np.zeros(F2.shape)
L=np.zeros((n,Lt.shape[1]))
MF=np.load('%s%sMF.npy' % (dr,fname))
Scal=np.load('%s%sScal.npy' % (dr,fname))
phi=np.load('%sphi.npy' % fname)	

avg=np.zeros(nt)
for i in range(nt):
	if i>20 and i<mt.shape[0]-20:
		avg[i]=np.mean(mt0[i-20:i+21])
MF2=np.mean(MF[:,avg==1],axis=1)
if args.N:
	L2=np.zeros((n,Lt.shape[1]))
	NSE=np.load('%s%sNoise.npy' % (dr,fname))
	NSE2=np.mean(NSE[:,avg==1],axis=1)
	S=-.1/np.mean((np.sum((Scal-MF2[:,np.newaxis])[30:,:]/(NSE2[30:,np.newaxis]),axis=0)/np.sum(1/NSE2[30:]))[avg==1])
else:
	SA=-.1/(np.mean(Scal-MF2[:,np.newaxis],axis=0))
	S=SA[avg==1].mean()

t=np.linspace(0,nt-1,nt)
KS=.021
KC=.035
alpha=0
phi=np.load('%sphi.npy' % fname)
Lpp=(1+alpha)*(KS*np.sin(phi)+KC*np.cos(phi))+alpha
t2=t+np.concatenate((np.zeros(1),cumtrapz(Lpp)))
t2-=t2.min()
t2=np.mod(t2,nt)

mt[mt>0]==1
for k in range(lengs[int(rank)]):
	fft_dspec1[:]=simmod.chi_sim(CG,nf,nt,0,fft_G1,fft_G2,fft_object_GF,fft_object_GB)
	fft_object_dspecF12()
	fft_object_dspecF23()
	fft_dspec3/=np.sqrt(fft_dspec1.shape[0]*fft_dspec1.shape[1])
	fft_dspec3[0,1:]=0
	fft_dspec3[1:,0]=0
	fft_dspec3[0,0]*=np.sqrt(CG.shape[0]*CG.shape[1])/np.abs(fft_dspec3[0,0])
	fft_object_dspecB32()
	fft_object_dspecB21()
	fft_dspec1*=np.sqrt(fft_dspec1.shape[0]*fft_dspec1.shape[1])
	unlns=np.copy(fft_dspec1)
	interp=interp1d(t,unlns,fill_value='extrapolate')
	fft_dspec1[:]=interp(t2)
	fft_dspec1[:]+=np.sqrt(N)*np.random.normal(0,1,(nf,nt))
	fft_dspec1*=mf[:,np.newaxis]
	fft_dspec1[:,mt==0]=0
	fft_object_dspecF12()
	fft_dspec2/=np.sqrt(nf)
	for i in range(F2.shape[0]):
		F2[i,:]=convolve1d(np.real(fft_dspec2[i+1,:]),np.real(K2[i+1,:]))+convolve1d(np.imag(fft_dspec2[i+1,:]),np.imag(K2[i+1,:]))+1j*(convolve1d(np.real(fft_dspec2[i+1,:]),np.imag(K2[i+1,:]))-convolve1d(np.imag(fft_dspec2[i+1,:]),np.real(K2[i+1,:])))
	Lt=np.real(fft_dspec2[1:,:]*F2*(A[1:][:,np.newaxis])-MF2[:,np.newaxis])
	if args.N:
		L2[disp+k,:]=(np.sum(Lt[30:,:]/(NSE2[30:,np.newaxis]),axis=0)/np.sum(1/NSE2[30:]))*S
		L[disp+k,:]=np.mean(Lt,axis=0)*S
	else:
		L[disp+k,:]=np.mean(Lt,axis=0)*S
	if rank==0:
		print(k+1,MPI.Wtime()-ts)
		sys.stdout.flush()
		

recvbuff=None 
if rank==0:
	recvbuff=np.zeros(L.shape)
comm.Barrier()
comm.Reduce(L,recvbuff,root=0,op=MPI.SUM)
if args.N:
	recvbuff2=None 
	if rank==0:
		recvbuff2=np.zeros(L2.shape)
	comm.Barrier()
	comm.Reduce(L2,recvbuff2,root=0,op=MPI.SUM)
if rank==0:
	np.save('%s%sCov.npy' % (dr,fname),recvbuff)
	np.save('%s%sInpt.npy' % (dr,fname),Lpp)
	np.save('%s%sDspecSim.npy' % (dr,fname),fft_dspec1)
	if args.N:
		np.save('%s%sCovN.npy' % (dr,fname),recvbuff2)
	



