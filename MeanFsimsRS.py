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

import pyfftw
import numpy as np
import argparse
import simmod
import fitmod
import weight
from mpi4py import MPI
import sys
from scipy.ndimage.filters import convolve1d
from scipy.integrate import simps
import os
import re
import pickle as pkl

##Initialize Multiprocessing
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

ts=MPI.Wtime()

if rank==0:
	print(size)
##Command Line Arguments
parser = argparse.ArgumentParser(description='Lens Recovery Code for B1957')
parser.add_argument('-d', type=int, help='Observation Date')
parser.add_argument('-n', type=int, help='Number of Simulations')
parser.add_argument('-f',type=str,help='fname prefix')
parser.add_argument('-th', type=int,default = 8,help='Number of Threads')
parser.add_argument('-nf', type=int,default = 6000,help='Number of Channels')
parser.add_argument('-nt', type=int,default = 2108,help='Number of Time Bins')
parser.add_argument('-al',type=float,default=0,help='Global Frequency Scaling')
parser.add_argument('-samp',type=int,help='MCMC sample Number')

args=parser.parse_args()
fname=args.f+str(args.d)
n=args.n

if args.al==0:
	dr='./AL0/'
else:
	dr='./AL%s/' % format(args.al,'.3e')

nt=args.nt
nf=args.nf

print(str(1)+'Setup FFTW')
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
##Import mask
mf=np.load(dr+fname+'MaskF.npy')
mt=np.load(dr+fname+'MaskT.npy')

mt[mt>0]==1
mf[mf>0]==1

##Divide simulations between processors 
delta=int(np.mod(n,size))
k=int((n-delta)/size)
lengs=np.append(np.ones(delta)*(k+1),np.ones(size-delta)*k).astype(int)
disp=np.cumsum(lengs)-lengs[0]

##variable arrays for Theoretical Powers
freq=np.fft.fftfreq(nt)
tau=np.fft.fftfreq(nf)

farr=np.ones((nf,nt))*freq
tarr=np.ones((nf,nt)).T*tau
tarr=tarr.T
X=(tarr,farr)

MCMC=np.load('%sSamples.npz' %args.f)
date_idx=np.argwhere(MCMC['dates']=='%sdspec.npy' %args.d)[0]
glob_fit=np.zeros(13)
glob_fit[np.array([0,1,2,3,4,5,10])]=MCMC['samps'][:,args.samp,date_idx*7:(date_idx+1)*7].mean(0)
glob_fit[np.array([6,7,8,9,11,12])]=MCMC['samps'][:,args.samp,-6:].mean(0)
CG=fitmod.pow_arr2D2(X,*glob_fit[1:])
CC=np.copy(fitmod.gauss_to_chi(CG,fft_G1,fft_G2,fft_object_GF,fft_object_GB))
N=glob_fit[0]

##Import Estimator Variables
A=np.load('%s%s/%sA.npy' % (dr,args.samp,fname))
K2=np.load('%s%s/%sK2.npy' % (dr,args.samp,fname))
F2=np.zeros(fft_dspec2[1:,:].shape,dtype=complex)
L=np.zeros(F2.shape)
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
	fft_dspec1[:]+=np.sqrt(N)*np.random.normal(0,1,(nf,nt))
	fft_dspec1*=mf[:,np.newaxis]
	fft_dspec1*=mt
	fft_object_dspecF12()
	fft_dspec2/=np.sqrt(nf)
	for i in range(F2.shape[0]):
		F2[i,:]=convolve1d(np.real(fft_dspec2[i+1,:]),np.real(K2[i+1,:]))+convolve1d(np.imag(fft_dspec2[i+1,:]),np.imag(K2[i+1,:]))+1j*(convolve1d(np.real(fft_dspec2[i+1,:]),np.imag(K2[i+1,:]))-convolve1d(np.imag(fft_dspec2[i+1,:]),np.real(K2[i+1,:])))
	L+=np.real(fft_dspec2[1:,:]*F2*(A[1:][:,np.newaxis]))
	if rank==0:
		print(k+1,MPI.Wtime()-ts)
		sys.stdout.flush()
		

recvbuff=None 
if rank==0:
	recvbuff=np.zeros(L.shape)
comm.Barrier()
comm.Reduce(L,recvbuff,root=0,op=MPI.SUM)
if rank==0:
	recvbuff/=n
	np.save('%s%s/%sMF.npy' % (dr,args.samp,fname),recvbuff)
	
	



