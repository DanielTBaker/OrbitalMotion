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
from scipy.integrate import simps
from scipy.interpolate import interp1d

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
parser.add_argument('-nt', type=int,default = 2110,help='Number of Time Bins')

args=parser.parse_args()
fname=args.f+str(args.d)
n=args.n

##Import Fit Parameters
popt=np.load(fname+'Fit.npy')
N=popt[0]
popt2=np.load(args.f+'GlobFit.npy')

nt=args.nt
nf=args.nf

print(str(1)+'Setup FFTW')
sys.stdout.flush()
nthread=args.th

fft_G1= pyfftw.empty_aligned((nf,nt), dtype='complex64')
fft_G2= pyfftw.empty_aligned((nf,nt), dtype='complex64')
fft_object_GF = pyfftw.FFTW(fft_G1,fft_G2, axes=(0,1), direction='FFTW_FORWARD',threads=nthread)
fft_object_GB = pyfftw.FFTW(fft_G2,fft_G1, axes=(1,0), direction='FFTW_BACKWARD',threads=nthread)



##Import mask
mf=np.load(fname+'MaskF.npy')


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

##Theoretical Powers
CG=fitmod.pow_arr2D2(X,*popt2[1:])

##Import Estimator Variables
EstVar=np.load(fname+'RSArray.npy')
K2=EstVar[0,:]
Ctp2=EstVar[1,:]
Ct2=EstVar[2,:]
IFtemp2=EstVar[3,:]
tint=EstVar[4,:]
A=1/simps(np.power(IFtemp2,2))
#A=1/simps(tint*K2*Ctp2,tint)
O=simps(K2*Ct2,tint)+(K2.max()*N)

##Loop over simulations
Lest=np.zeros((lengs[int(rank)],nt))
t0=np.linspace(0,nt-1,nt)
t1=np.linspace(0,nt,nt+1)
t2=t0+5*np.sin(np.pi*t0/(2*nt))
t2-=t2.min()
t2=np.mod(t2,nt)

for i in range(lengs[int(rank)]):
	temp=simmod.chi_sim(CG,nf,nt,0,fft_G1,fft_G2,fft_object_GF,fft_object_GB)
	interp=interp1d(t1,np.concatenate((temp,temp[:,0][:,np.newaxis]),axis=1))
	dspec=interp(t2)+np.sqrt(N)*np.random.normal(0,1,(nf,nt))
	dspec2=convolve1d(dspec,K2,mode='constant')
	Lest[i,:]=np.mean(((dspec*dspec2-O)*A)[mf>0,:],axis=0)
	if rank==0:
		print(i+1,MPI.Wtime()-ts)
		sys.stdout.flush()
		

recvbuff=None 
if rank==0:
	recvbuff=np.empty((n,nt))
comm.Gatherv(Lest,[recvbuff,lengs*nt,disp*nt,MPI.DOUBLE],root=0)
if rank==0:
	np.save('%sLensSims.npy' % fname,recvbuff)
	
	



