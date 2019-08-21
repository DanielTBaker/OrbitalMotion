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
import sys
import os
import re
import pickle as pkl
from emcee.utils import MPIPool
import numpy as np
import emcee
import sys
from mpi4py import MPI

run_start=MPI.Wtime()

##Command Line Arguments
parser = argparse.ArgumentParser(description='Lens Recovery Code for B1957')
parser.add_argument('-f',type=str,help='fname prefix')
parser.add_argument('-T',dest='T',action='store_true',help='Transpose dspec')
parser.add_argument('-bt',type=float,default=np.inf,help='Number of Time Bins')
parser.add_argument('-bf',type=float,default=np.inf,help='Number of Frequncy Bins')
parser.add_argument('-lt', type=int,default = 50,help='Mask Smoothing Length in Time')
parser.add_argument('-lf', type=int,default = 20,help='Mask Smoothing Length in Frequency')
parser.add_argument('-th', type=int,default = 8,help='Number of Threads')
parser.add_argument('-ns', type=int,default = 50,help='Number of Steps')

parser.set_defaults(T=False)
parser.set_defaults(G=False)
parser.set_defaults(fit=True)
args=parser.parse_args()


load=os.path.isfile('%sSamples.npz' %args.f)
if load:
	samples_old=np.load('%sSamples.npz' %args.f)['samps']
	names_old=np.load('%sSamples.npz' %args.f)['names']
	dates=np.load('%sSamples.npz' %args.f)['dates']
else:
	samples_old=np.zeros((0,0,0))
	dates=np.array([re.findall('[0-9][0-9]dspec.npy',f) for f in os.listdir('./') if re.search('[0-9]+dspec.npy',f)])[:,0]

nt=500
nf=500
ns=1

freq=np.fft.fftfreq(nt)
tau=np.fft.fftfreq(nf)

fname=args.f+dates[0][:2]
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
fft_G1 = pyfftw.empty_aligned((nf,nt), dtype='complex128')
fft_G2 = pyfftw.empty_aligned((nf,nt), dtype='complex128')
fft_object_GF = pyfftw.FFTW(fft_G1,fft_G2, axes=(0,1), direction='FFTW_FORWARD',threads=nthread)
fft_object_GB = pyfftw.FFTW(fft_G2,fft_G1, axes=(1,0), direction='FFTW_BACKWARD',threads=nthread)
fft_object_GBA0 = pyfftw.FFTW(fft_G2,fft_G1, axes=(0,), direction='FFTW_BACKWARD',threads=nthread)
fft_dspec1 = pyfftw.empty_aligned((nf,nt),dtype='float64')
fft_dspec2 = pyfftw.empty_aligned((int(nf/2)+1,nt),dtype='complex128')
fft_dspec3 = pyfftw.empty_aligned((int(nf/2)+1,nt),dtype="complex128")
fft_object_dspecF12 = pyfftw.FFTW(fft_dspec1,fft_dspec2, axes=(0,), direction='FFTW_FORWARD',threads=nthread)
fft_object_dspecF23 = pyfftw.FFTW(fft_dspec2,fft_dspec3, axes=(1,), direction='FFTW_FORWARD',threads=nthread)
fft_object_dspecB32 = pyfftw.FFTW(fft_dspec3,fft_dspec2, axes=(1,), direction='FFTW_BACKWARD',threads=nthread)
fft_object_dspecB21 = pyfftw.FFTW(fft_dspec2,fft_dspec1, axes=(0,), direction='FFTW_BACKWARD',threads=nthread)
pkl.dump(pyfftw.export_wisdom(),open('pyfftwwis-1-%s.pkl' % args.th,'wb'))

def lnprior(theta,lwrs,uprs):
    if (theta-lwrs).min()<0:
        return(-np.inf)
    if (theta-uprs).max()>0:
        return(-np.inf)
    return(0.0)
   
def lnprob(theta,C,X,IFCM,lwrs,uprs):
	global fft_G1,fft_G2,fft_dspec1,fft_dspec2,fft_dspec3
	lp=lnprior(theta,lwrs,uprs)
	if not np.isfinite(lp):
		return -np.inf
	lp2=0
	P=np.zeros(13)
	for i in range(len(C)):
		P[np.array([0,1,2,3,4,5,10])]=theta[i*7:(i+1)*7]
		P[np.array([6,7,8,9,11,12])]=theta[-6:]
		lp2-=fitmod.costfunc(P,C[i],X,IFCM[i],fft_G1,fft_G2,fft_object_GF,fft_object_GB,fft_dspec1,fft_dspec2,fft_dspec3,fft_object_dspecF12,fft_object_dspecF23,fft_object_dspecB32,fft_object_dspecB21)
	return(lp+lp2)

ndim, nwalkers = 34, 70

init=np.zeros(ndim)
names=np.empty(34,dtype='<U3') 
for i in range(dates.shape[0]):
	fname=args.f+dates[i][:2]
	P=np.load(fname+'Fit2.npy')
	init[i*7:(i+1)*7]=P[np.array([0,1,2,3,4,5,10])]
	names[i*7:(i+1)*7]=np.array(['N','A','t01','nt1','t02','nt2','A2'])
init[-6:]=P[np.array([6,7,8,9,11,12])]
names[-6:]=np.array(['nf','f01','f02','B','b01','b02'])

if load:
	pos=samples_old[:,-1,:]
else:
	pos=[(np.random.normal(0,1,ndim)*init/100)+init for i in range(nwalkers)]


print('pre-pool')
pool = MPIPool(loadbalance=True)
if not pool.is_master():
	pool.wait()
	sys.exit(0)
print('post pool')
print(names)

##variable arrays for fitting
freq=np.fft.fftfreq(nt)
tau=np.fft.fftfreq(nf)

farr=np.ones((tau.shape[0],freq.shape[0]))*freq
tarr=np.ones((tau.shape[0],freq.shape[0])).T*tau
tarr=tarr.T
X=(tarr,farr)

C_list, IFCM_list = list(), list()
C_data=np.zeros(fft_dspec3.shape)

for i in range(dates.shape[0]):
	fname=args.f+dates[i][:2]
	if args.T:
		dspec=(np.load(fname+'dspec.npy')[:nt,:nf]).T
	else:
		dspec=(np.load(fname+'dspec.npy')[:nf,:nt])
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
	fft_G1[:]=msk_smooth
	##Correlation function of mask for use in conversions
	fft_object_GF()
	fft_G2*=np.conjugate(fft_G2)/(nf*nt)
	fft_object_GB()
	IFCM=np.copy(np.real(fft_G1))
	fft_dspec1[:]=dspec*msk_smooth/svd_gaps
	fft_object_dspecF12()
	fft_object_dspecF23()
	C_data[:]=np.copy(np.abs(fft_dspec3[:]/np.sqrt(nf*nt))**2)
	C_list.append(C_data[:,1:])
	IFCM_list.append(IFCM)

lwrs=np.zeros(34)
lwrs[-2:]=np.array([-np.inf,-np.inf])
uprs=np.ones(34)*np.inf

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool,args=(C_list,X,IFCM_list,lwrs,uprs))
niters=args.ns

for i, result in enumerate(sampler.sample(pos, iterations=niters)):
	if (i+1) % (args.ns//10) == 0:
		samples=sampler.chain
		if load:
			samples=np.concatenate((samples_old,samples),axis=1)
		np.savez('%sSamples.npz' %args.f,samps=samples[:,:i+1+samples_old.shape[1],:],dates=dates,names=names)
		print("Step %s of %s finished at %s" %(i+1,niters,MPI.Wtime()-run_start),flush=True)

samples=sampler.chain
if load and np.all(dates==dates_old):
	samples=np.concatenate((samples_old,samples),axis=1)

for i in range(4):
	for j in range(7):
		plt.figure(figsize=(8,8))
		for k in range(nwalkers):
			plt.plot(samples[k,:,i*7+j])
		plt.title('%s (%s) : %s +- %s (%s)' %(names[i*7+j],i,samples[:,:,i*7+j].mean(),samples[:,:,i*7+j].std(),init[i*7+j]))
		plt.axhline(init[i*7+j])
		plt.savefig('%s%s%s.png' %(args.f,dates[i][:2],names[i*7+j]))
		plt.close('all')
for i in range(6):
	plt.figure(figsize=(8,8))
	for k in range(nwalkers):
		plt.plot(samples[k,:,-6+i])
	plt.title('%s : %s +- %s (%s)' %(names[-6+i],samples[:,:,-6+i].mean(),samples[:,:,-6+i].std(),init[-6+i]))
	plt.axhline(init[-6+i])
	plt.savefig('%s%s.png' %(args.f,names[-6+i]))
	plt.close('all')
np.savez('%sSamples.npz' %args.f,samps=samples,dates=dates,names=names)
print(dates)
pool.close()
