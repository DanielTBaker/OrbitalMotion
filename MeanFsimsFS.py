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
from scipy.integrate import simps
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
parser.add_argument('-al',type=float,default=0,help='Global Frequency Scaling')

args=parser.parse_args()
fname=args.f+str(args.d)
n=args.n

##Import Fit Parameters
popt=np.load(fname+'Fit.npy')
N=popt[0]
popt2=np.load(args.f+'GlobFit.npy')

##Determine A(L)
A=np.load(fname+'A.npy')

nt=A.shape[1]
nf=(A.shape[0]-1)*2

print(str(1)+'Setup FFTW')
sys.stdout.flush()
nthread=args.th

fft_G1= pyfftw.empty_aligned((nf,nt), dtype='complex64')
fft_G2= pyfftw.empty_aligned((nf,nt), dtype='complex64')
fft_object_GF = pyfftw.FFTW(fft_G1,fft_G2, axes=(0,1), direction='FFTW_FORWARD',threads=nthread)
fft_object_GB = pyfftw.FFTW(fft_G2,fft_G1, axes=(1,0), direction='FFTW_BACKWARD',threads=nthread)
fft_dspec1 = pyfftw.empty_aligned((nf,nt),dtype='float32')
fft_dspec2 = pyfftw.empty_aligned((int(nf/2)+1,nt),dtype='complex64')
fft_dspec3 = pyfftw.empty_aligned((int(nf/2)+1,nt),dtype="complex64")
fft_object_dspecF12=pyfftw.FFTW(fft_dspec1,fft_dspec2, axes=(0,), direction='FFTW_FORWARD',threads=nthread)
fft_object_dspecF23=pyfftw.FFTW(fft_dspec2,fft_dspec3, axes=(1,), direction='FFTW_FORWARD',threads=nthread)
fft_object_dspecB32=pyfftw.FFTW(fft_dspec3,fft_dspec2, axes=(1,), direction='FFTW_BACKWARD',threads=nthread)
fft_object_dspecB21=pyfftw.FFTW(fft_dspec2,fft_dspec1, axes=(0,), direction='FFTW_BACKWARD',threads=nthread)
fft_FW = pyfftw.empty_aligned((int(nf/2)+1,nt),dtype='complex64')
fft_FG = pyfftw.empty_aligned((int(nf/2)+1,nt),dtype='complex64')
fft_W = pyfftw.empty_aligned((int(nf/2)+1,nt),dtype='complex64')
fft_G = pyfftw.empty_aligned((int(nf/2)+1,nt),dtype='complex64')
fft_object_IFW = pyfftw.FFTW(fft_FW,fft_W, axes=(1,), direction='FFTW_BACKWARD',threads=nthread)
fft_object_IFW2 = pyfftw.FFTW(fft_FG,fft_G, axes=(1,), direction='FFTW_BACKWARD',threads=nthread)
fft_object_IFW3 = pyfftw.FFTW(fft_FW,fft_dspec2, axes=(1,), direction='FFTW_BACKWARD',threads=nthread)


##Import mask
mf=np.load('AL0/'+fname+'MaskF.npy')
mt=np.load('AL0/'+fname+'MaskT.npy')


##Correlation function of mask for use in conversions
fft_G1[:]=(np.ones((nt,nf))*mf).T*mt
fft_object_GF()
fft_G2*=np.conjugate(fft_G2)/(nf*nt)
fft_object_GB()
IFCM=np.copy(np.real(fft_G1))

##Divide simulations between processors 
delta=int(np.mod(n,size))
k=int((n-delta)/size)
lengs=np.append(np.ones(delta)*(k+1),np.ones(size-delta)*k).astype(int)
lns=np.zeros((1+int(nf/2),nt))

##variable arrays for Theoretical Powers
freq=np.fft.fftfreq(nt)
tau=np.fft.fftfreq(nf)

farr=np.ones((nf,nt))*freq
tarr=np.ones((nf,nt)).T*tau
tarr=tarr.T
X=(tarr,farr)

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

##Theoretical Powers
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

##Loop over simulations
for i in range(lengs[int(rank)]):
	simmod.nolens_sim(CG,CC,CT,CCR,CTR,N,freq,nf,nt,mf,mt,A,fft_G1,fft_G2,fft_object_GF,fft_object_GB,fft_dspec1,fft_dspec2,fft_dspec3,fft_object_dspecF12,fft_object_dspecF23,fft_object_dspecB32,fft_FW,fft_FG,fft_W,fft_G,fft_object_IFW,fft_object_IFW2,fft_object_IFW3)
	lns+=np.real(fft_dspec2)[:]
	if rank==0:
		print(i+1,MPI.Wtime()-ts)
		sys.stdout.flush()
		

recvbuff=None 
if rank==0:
	recvbuff=np.zeros(lns.shape)
comm.Barrier()
comm.Reduce(lns,recvbuff,root=0,op=MPI.SUM)

lnsM=np.zeros(lns.shape)
lnsS=np.zeros(lns.shape)
if rank==0:
	lnsM=recvbuff/n
	np.save(fname+'MeanF.npy',lnsM)

##Convergence Test
comm.Barrier()
comm.Bcast(lnsM,root=0)
locS=((lns-lnsM)**2)/size
comm.Barrier()
comm.Reduce(locS,lnsS,root=0,op=MPI.SUM)
if rank==0:
	Ms=lnsM**2
	print(lnsS.mean()/(size*Ms.mean()),lnsS.mean(),Ms.mean(),lnsS.std(),Ms.std())
	
	



