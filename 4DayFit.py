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
from mpi4py import MPI
import numpy as np
import fitmod
import argparse
import weight
from scipy.optimize import minimize,curve_fit
import sys
import os
import re
import pickle as pkl

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

ts=MPI.Wtime()

##Command Line Arguments
parser = argparse.ArgumentParser(description='Lens Recovery Code for B1957')
parser.add_argument('-f',type=str,help='fname prefix')
parser.add_argument('-T',dest='T',action='store_true',help='Transpose dspec')
parser.add_argument('-bt',type=float,default=np.inf,help='Number of Time Bins')
parser.add_argument('-bf',type=float,default=np.inf,help='Number of Frequncy Bins')
parser.add_argument('-lt', type=int,default = 50,help='Mask Smoothing Length in Time')
parser.add_argument('-lf', type=int,default = 20,help='Mask Smoothing Length in Frequency')
parser.add_argument('-th', type=int,default = 8,help='Number of Threads')

parser.set_defaults(T=False)
parser.set_defaults(G=False)
parser.set_defaults(fit=True)
args=parser.parse_args()

dates=np.array([re.findall('[0-9][0-9]dspec.npy',f) for f in os.listdir('./') if re.search('[0-9]+dspec.npy',f)])[:,0]



fname=args.f+dates[rank][:2]
sz=np.load(fname+'dspec.npy').shape
nf=np.min((sz[0],args.bf)).astype(int)
nt=np.min((sz[1],args.bt)).astype(int)


print('Parsed')
sys.stdout.flush()

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
mt, mf, svd_gaps = fitmod.NormMask(dspec,args.lf,args.lt)
msk_sharp=mt[np.newaxis,:]*mf[:,np.newaxis]
msk_smooth=np.copy(msk_sharp)
t=np.linspace(1,args.lf,args.lf)
start,stop=fitmod.bnd_find(mf,mf.shape[0])
for i in range(start.shape[0]):
	msk_smooth[start[i]:start[i]+args.lf,:]*=(1-np.cos(np.pi*t[:,np.newaxis]/(args.lf+1)))/2
	msk_smooth[stop[i]-args.lf+1:stop[i]+1,:]*=(np.cos(np.pi*(t[:,np.newaxis])/(args.lf+1))+1)/2
t=np.linspace(1,args.lt,args.lt)
start,stop=fitmod.bnd_find(mt,mt.shape[0])
for i in range(start.shape[0]):
	msk_smooth[:,start[i]:start[i]+args.lt]*=(1-np.cos(np.pi*t[np.newaxis,:]/(args.lt+1)))/2
	msk_smooth[:,stop[i]-args.lt+1:stop[i]+1]*=(np.cos(np.pi*(t[np.newaxis,:])/(args.lt+1))+1)/2

print('Find IFCM')
sys.stdout.flush()
fft_G1[:]=msk_smooth*svd_gaps
##Correlation function of mask for use in conversions
fft_object_GF()
fft_G2*=np.conjugate(fft_G2)/(nf*nt)
fft_object_GB()
IFCM=np.copy(np.real(fft_G1))

##Find Power Spectrum
print('Caclulate Power')
sys.stdout.flush()
fft_dspec1[:]=dspec*msk_smooth
fft_object_dspecF12()
fft_object_dspecF23()
C_data[:]=np.copy(np.abs(fft_dspec3[:]/np.sqrt(nf*nt))**2)

def parallel_function_caller(x,stopp,rank,C_data,X,IFCM,fft_G1,fft_G2,fft_object_GF,fft_object_GB,fft_dspec1,fft_dspec2,fft_dspec3,fft_object_dspecF12,fft_object_dspecF23,fft_object_dspecB32,fft_object_dspecB21,tskip):
	stopp[0]=comm.bcast(stopp[0], root=0)
	summ=0
	if stopp[0]==0:
		#your function here in parallel
		x=comm.bcast(x, root=0)
		P=np.zeros(13)
		P[np.array([0,1,2,3,4,5,10])]=x[rank*7:(rank+1)*7]
		P[np.array([6,7,8,9,11,12])]=x[-6:]
		summ1=fitmod.costfunc(P,C_data[:,1:],X,IFCM,fft_G1,fft_G2,fft_object_GF,fft_object_GB,fft_dspec1,fft_dspec2,fft_dspec3,fft_object_dspecF12,fft_object_dspecF23,fft_object_dspecB32,fft_object_dspecB21,tskip)
		summ=comm.reduce(summ1,op=MPI.SUM, root=0)
	return(summ)

if rank == 0 :
	stop=[0]
	x = np.zeros(size*7+6)
	avg=np.zeros(6)
	for i in range(dates.shape[0]):
		fit=np.load(args.f+dates[i][:2]+'Fit.npy')
		avg+=fit[np.array([6,7,8,9,11,12])]
		x[i*7:(i+1)*7]=fit[np.array([0,1,2,3,4,5,10])]
	avg/=dates.shape[0]
	x[-6:]=avg
	res = minimize(parallel_function_caller,x, args=(stop,rank,C_data,X,IFCM,fft_G1,fft_G2,fft_object_GF,fft_object_GB,fft_dspec1,fft_dspec2,fft_dspec3,fft_object_dspecF12,fft_object_dspecF23,fft_object_dspecB32,fft_object_dspecB21,1))
	stop=[1]
	parallel_function_caller(x,stop,rank,C_data,X,IFCM,fft_G1,fft_G2,fft_object_GF,fft_object_GB,fft_dspec1,fft_dspec2,fft_dspec3,fft_object_dspecF12,fft_object_dspecF23,fft_object_dspecB32,fft_object_dspecB21,31)
if rank==1:
	runs=0
if rank>0:
	stop=[0]
	x=np.zeros(size*7+6)
	while stop[0]==0:
		parallel_function_caller(x,stop,rank,C_data,X,IFCM,fft_G1,fft_G2,fft_object_GF,fft_object_GB,fft_dspec1,fft_dspec2,fft_dspec3,fft_object_dspecF12,fft_object_dspecF23,fft_object_dspecB32,fft_object_dspecB21,1)
		if rank==1:
			runs+=1
			if np.mod(runs,100)==0:
				rt=MPI.Wtime()-ts
				print('First %s Evaluations in %s seconds' %(runs,rt))	
if rank==0:
	PF=res.x
else:
	PF=np.zeros(x.shape)
comm.Bcast(PF,root=0)
P=np.zeros(13)
P[np.array([0,1,2,3,4,5,10])]=PF[rank*7:(rank+1)*7]
P[np.array([6,7,8,9,11,12])]=PF[-6:]
np.save(fname+'Fit2.npy',P)
