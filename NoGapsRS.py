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
from scipy.interpolate import interp1d

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
parser.add_argument('-G',dest='G',action='store_true',help='Use Global Fiducial C')
parser.add_argument('-al',type=float,default=0,help='Global Frequency Scaling')
parser.add_argument('-lt', type=int,default = 50,help='Mask Smoothing Length in Time')
parser.add_argument('-lf', type=int,default = 20,help='Mask Smoothing Length in Frequency')

parser.set_defaults(G=False)

args=parser.parse_args()
fname=args.f+str(args.d)
n=args.n

if args.al==0:
	dr='./AL0/'
else:
	dr='./AL%s/' % format(args.al,'.3e')

##Import Fit Parameters
popt=np.load(fname+'Fit.npy')
N=popt[0]
popt2=np.load(args.f+'GlobFit.npy')

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
fft_object_GBA0 = pyfftw.FFTW(fft_G2,fft_G1, axes=(0,), direction='FFTW_BACKWARD',threads=nthread)
fft_dspec1 = pyfftw.empty_aligned((nf,nt),dtype='float64')
fft_dspec2 = pyfftw.empty_aligned((int(nf/2)+1,nt),dtype='complex128')
fft_dspec3 = pyfftw.empty_aligned((int(nf/2)+1,nt),dtype="complex128")
fft_object_dspecF12=pyfftw.FFTW(fft_dspec1,fft_dspec2, axes=(0,), direction='FFTW_FORWARD',threads=nthread)
fft_object_dspecF23=pyfftw.FFTW(fft_dspec2,fft_dspec3, axes=(1,), direction='FFTW_FORWARD',threads=nthread)
fft_object_dspecB32=pyfftw.FFTW(fft_dspec3,fft_dspec2, axes=(1,), direction='FFTW_BACKWARD',threads=nthread)
fft_object_dspecB21=pyfftw.FFTW(fft_dspec2,fft_dspec1, axes=(0,), direction='FFTW_BACKWARD',threads=nthread)
pkl.dump(pyfftw.export_wisdom(),open('pyfftwwis-%s-%s.pkl' % (size,args.th),'wb'))

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

CG=fitmod.pow_arr2D2(X,*popt[1:])
if args.G:
	CC=np.copy(fitmod.gauss_to_chi(CG,fft_G1,fft_G2,fft_object_GF,fft_object_GB))
	temp=np.fft.fftshift(CC)
	sigscal=np.sqrt(simps(simps(temp,dx=1/nt,axis=1),dx=1/nf,axis=0))
	#sigscal=np.sqrt(CC[1,1])

	N/=sigscal**2
	X=(tarr,farr*(1+args.al))
	popt2[1]*=(1+args.al)**(3/2)
	##Theoretical Powers
	fits=np.array([re.findall('b1957-[0-9]+Fit.npy',f) for f in os.listdir('./') if re.search('[0-9]+Fit.npy',f)])[:,0]
	CG=np.zeros(CG.shape)
	X=(tarr,farr*(1.+args.al))
	for fit in fits:
		popt2=np.load(fit)
		popt2[1]*=(1+args.al)**(3/2)
		CG+=fitmod.pow_arr2D2(X,*popt2[1:])
	CG/=fits.shape[0]
	CC=np.copy(fitmod.gauss_to_chi(CG,fft_G1,fft_G2,fft_object_GF,fft_object_GB))
	temp=np.fft.fftshift(CC)
	sigscal=np.sqrt(simps(simps(temp,dx=1/nt,axis=1),dx=1/nf,axis=0))
	CG/=sigscal

##Import Estimator Variables
fft_dspec3[:]=CC
fft_object_dspecB32()
fft_object_dspecB21()
fft_object_dspecF12()
fft_object_dspecF23()
CC2=np.copy(np.real(fft_dspec3))
fft_dspec3[:]=CC+N
fft_object_dspecB32()
fft_object_dspecB21()
fft_object_dspecF12()
fft_object_dspecF23()
CCN=np.copy(np.real(fft_dspec3))


nw=args.lt
CP=np.fft.ifftshift(np.gradient(np.fft.fftshift(CC2,axes=1),freq[1],axis=1),axes=1)
FK0=(CC2/np.power(CCN,2))*(1+(freq*CP/CC2))
FK0[CC2==0]=0
K0=np.fft.ifft(FK0,axis=1)
K2=np.roll(K0,nw,axis=1)[:,:2*nw+1]
temp=FK0*CC2*(1+(freq*CP/CC2))
temp[CC2==0]=0
A=1/(simps(np.fft.fftshift(temp,axes=1),dx=freq[1])*2*np.pi*nt)
F2=np.zeros(fft_dspec2[1:,:].shape,dtype=complex)
L=np.zeros(F2.shape)
for k in range(lengs[int(rank)]):
	fft_dspec1[:]=simmod.chi_sim(CG,nf,nt,0,fft_G1,fft_G2,fft_object_GF,fft_object_GB)+np.sqrt(N)*np.random.normal(0,1,(nf,nt))
	fft_object_dspecF12()
	fft_dspec2/=np.sqrt(nf)
	for i in range(F2.shape[0]):
		F2[i,:]=convolve1d(np.real(fft_dspec2[i+1,:]),np.real(K2[i+1,:]))+convolve1d(np.imag(fft_dspec2[i+1,:]),np.imag(K2[i+1,:]))+1j*(convolve1d(np.real(fft_dspec2[i+1,:]),np.imag(K2[i+1,:]))-convolve1d(np.imag(fft_dspec2[i+1,:]),np.real(K2[i+1,:])))
	L+=np.real(fft_dspec2[1:,:]*F2*(A[1:][:,np.newaxis]))
	if rank==0:
		print(k+1,MPI.Wtime()-ts)
		sys.stdout.flush()
		

MF=np.zeros(L.shape)
comm.Barrier()
comm.Allreduce(L,MF,op=MPI.SUM)
MF/=n

t=np.linspace(0,nt-1,nt)
t2=.99*t
for k in range(lengs[int(rank)]):
	interp=interp1d(t,simmod.chi_sim(CG,nf,nt,0,fft_G1,fft_G2,fft_object_GF,fft_object_GB))
	fft_dspec1[:]=interp(t2)
	fft_dspec1[:]+=np.sqrt(N)*np.random.normal(0,1,(nf,nt))
	fft_object_dspecF12()
	fft_dspec2/=np.sqrt(nf)
	for i in range(F2.shape[0]):
		F2[i,:]=convolve1d(np.real(fft_dspec2[i+1,:]),np.real(K2[i+1,:]))+convolve1d(np.imag(fft_dspec2[i+1,:]),np.imag(K2[i+1,:]))+1j*(convolve1d(np.real(fft_dspec2[i+1,:]),np.imag(K2[i+1,:]))-convolve1d(np.imag(fft_dspec2[i+1,:]),np.real(K2[i+1,:])))
	L+=np.real(fft_dspec2[1:,:]*F2*(A[1:][:,np.newaxis]))
	if rank==0:
		print(k+1,MPI.Wtime()-ts)
		sys.stdout.flush()
		

Scal=np.zeros(L.shape)
comm.Barrier()
comm.Allreduce(L,Scal,op=MPI.SUM)
Scal/=n

MF2=np.mean(MF[:,20:-20],axis=1)

SA=-.01/np.mean(Scal-MF2[:,np.newaxis],axis=0)
S=SA[20:-20].mean()

t=np.linspace(0,nt-1,nt)
KS=.021
KC=.035
alpha=-.107
Lpp=(1+alpha)*(KS*np.sin(phi)+KC*np.cos(phi))+alpha
t2=t+np.concatenate((np.zeros(1),cumtrapz(Lpp)))
t2-=t2.min()

mt[mt>0]==1
for k in range(lengs[int(rank)]):
	interp=interp1d(t,simmod.chi_sim(CG,nf,nt,0,fft_G1,fft_G2,fft_object_GF,fft_object_GB))
	fft_dspec1[:]=interp(t2)
	fft_dspec1[:]+=np.sqrt(N)*np.random.normal(0,1,(nf,nt))
	fft_dspec1*=mf[:,np.newaxis]
	fft_dspec1[:,mt==0]=0
	fft_object_dspecF12()
	fft_dspec2/=np.sqrt(nf)
	for i in range(F2.shape[0]):
		F2[i,:]=convolve1d(np.real(fft_dspec2[i+1,:]),np.real(K2[i+1,:]))+convolve1d(np.imag(fft_dspec2[i+1,:]),np.imag(K2[i+1,:]))+1j*(convolve1d(np.real(fft_dspec2[i+1,:]),np.imag(K2[i+1,:]))-convolve1d(np.imag(fft_dspec2[i+1,:]),np.real(K2[i+1,:])))
	Lt=np.real(fft_dspec2[1:,:]*F2*(A[1:][:,np.newaxis])-MF2[:,np.newaxis])
	L[disp+k,:]=np.mean(Lt,axis=0)*S
	if rank==0:
		print(k+1,MPI.Wtime()-ts)
		sys.stdout.flush()
		

recvbuff=None 
if rank==0:
	recvbuff=np.zeros(L.shape)
comm.Barrier()
comm.Reduce(L,recvbuff,root=0,op=MPI.SUM)
	
if rank==0:
	np.save('%s%sCovNG.npy' % (dr,fname),recvbuff)


