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
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

data=np.load('b1957-201406Samples.npz')

samps=data['samps']
dates=data['dates']
names=data['names']

with PdfPages('Plots/MCMC_chains.pdf') as pdf:
	for i in range(dates.shape[0]):
		for k in range(7):
			plt.figure()
			plt.xlabel('Step')
			plt.ylabel('Value')
			plt.title('2014/06/%s %s' %(dates[i][:2], names[i*7+k]))
			for chain in range(samps.shape[0]):
				plt.plot(samps[chain,:,i*7+k])
			pdf.savefig()
			plt.close()

	for k in range(6):
			plt.figure()
			plt.xlabel('Step')
			plt.ylabel('Value')
			plt.title('Global %s' %(names[28+k]))
			for chain in range(samps.shape[0]):
				plt.plot(samps[chain,:,i*7+k])
			pdf.savefig()
			plt.close()



nthread=20

for D in dates:
    dspec=np.load('b1957-201406%sdspec.npy' %D[:2])

    sz=dspec.shape
    nf=sz[0]
    nt=sz[1]    

    print('Start FFT Setup',flush=True)
    ##Check if wisdom exists to speed the ffts
    try:
        pyfftw.import_wisdom(pkl.load(open('pyfftwwis-1-20.pkl','rb')))
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
    pkl.dump(pyfftw.export_wisdom(),open('pyfftwwis-1-%s.pkl' % 20,'wb'))

    freq=np.fft.fftfreq(nt)
    tau=np.fft.fftfreq(nf)

    farr=np.ones((tau.shape[0],freq.shape[0]))*freq
    tarr=np.ones((tau.shape[0],freq.shape[0]))*tau[:,np.newaxis]
    X=(tarr,farr)

    mt,mf,dspec_svd=fitmod.NormMask(dspec,10,20)
    dspec/=dspec_svd
    t=np.linspace(1,10,10)
    start,stop=fitmod.bnd_find(mf,mf.shape[0])
    for i in range(start.shape[0]):
        mf[start[i]:start[i]+10]=(1-np.cos(np.pi*t/(10+1)))/2
        mf[stop[i]-10+1:stop[i]+1]=(np.cos(np.pi*(t)/(10+1))+1)/2
    t=np.linspace(1,20,20)
    start,stop=fitmod.bnd_find(mt,mt.shape[0])
    for i in range(start.shape[0]):
        mt[start[i]:start[i]+20]=(1-np.cos(np.pi*t/(20+1)))/2
        mt[stop[i]-20+1:stop[i]+1]=(np.cos(np.pi*(t)/(20+1))+1)/2

    print('Find IFCM',flush=True)
    fft_G1[:]=mf[:,np.newaxis]*mt[np.newaxis,:]
    ##Correlation function of mask for use in conversions
    fft_object_GF()
    fft_G2*=np.conjugate(fft_G2)/(nf*nt)
    fft_object_GB()
    IFCM=np.copy(np.real(fft_G1))

    fft_dspec1[:]=np.copy(dspec)*mf[:,np.newaxis]*mt[np.newaxis,:]
    fft_object_dspecF12()
    fft_object_dspecF23()
    C_data[:]=np.copy(np.abs(fft_dspec3[:]/np.sqrt(nf*nt))**2)

    date_idx=int(np.argwhere(dates == D)[0])
    glob_fit=np.zeros(13)
    plt.figure(0)
    glob_fit[np.array([0,1,2,3,4,5,10])]=samps[:,-1,date_idx*7:(date_idx+1)*7].mean(0)
    glob_fit[np.array([6,7,8,9,11,12])]=samps[:,-1,-6:].mean(0)
    CG=fitmod.pow_arr2D2(X,*glob_fit[1:])
    CC=np.copy(fitmod.gauss_to_chi(CG,fft_G1,fft_G2,fft_object_GF,fft_object_GB))
    N=glob_fit[0]

    fft_dspec3[:]=CC+N
    fft_object_dspecB32()
    fft_object_dspecB21()
    fft_dspec1*=IFCM
    fft_object_dspecF12()
    fft_object_dspecF23()
    plt.subplot(2,2,date_idx+1)
    plt.title('2016/04/%s' %D[:2])
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
    plt.figure(1)
    glob_fit[np.array([0,1,2,3,4,5,10])]=samps[:,0,date_idx*7:(date_idx+1)*7].mean(0)
    glob_fit[np.array([6,7,8,9,11,12])]=samps[:,0,-6:].mean(0)
    CG=fitmod.pow_arr2D2(X,*glob_fit[1:])
    CC=np.copy(fitmod.gauss_to_chi(CG,fft_G1,fft_G2,fft_object_GF,fft_object_GB))
    N=glob_fit[0]

    fft_dspec3[:]=CC+N
    fft_object_dspecB32()
    fft_object_dspecB21()
    fft_dspec1*=IFCM
    fft_object_dspecF12()
    fft_object_dspecF23()
    plt.subplot(2,2,date_idx+1)
    plt.title('2016/04/%s' %D[:2])
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
    plt.legend()

plt.figure(0)
plt.savefig('./Plots/last_samp.png')
plt.figure(1)
plt.savefig('./Plots/first_samp.png')