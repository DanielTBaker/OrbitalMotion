"""
Fuctions used in determining ans using weights for estimator
Daniel Baker
Last Edited 07/13/2017 
"""

import numpy as np

##Convolution script
def Fconv(A1,A2):
    if len(A1.shape)==1:
        return(np.fft.fft(np.fft.ifft(A1)*np.fft.ifft(A2)))
    else:
        if len(A2.shape)==2:
            return(np.fft.fft(np.fft.ifft(A1,axis=1)*np.fft.ifft(A2,axis=1),axis=1))
        else:
            return(np.fft.fft(np.fft.ifft(A1,axis=1)*np.fft.ifft(A2),axis=1))

##Convert A(l)->A(L-l)
def array_revshift(A,idx,A5):
    A2=np.fft.fftshift(A,axes=1)
    Arev=A2[:,::-1]
    A3=np.zeros(A.shape)
    if idx<0:
        A3[:,:A.shape[1]-abs(idx)+1]=Arev[:,abs(idx)-1:]
    else:
        A3[:,idx]=A5
        A3[:,idx+1:]=Arev[:,:A.shape[1]-1-idx]
    return(np.fft.ifftshift(A3,axes=1))


##Find weighting function from Power Spectrum and Frequency Cutoff
def AL_find(CU,CT,freq,fcutA):
	AL=np.zeros(CU.shape)

	for i in range(freq.shape[0]):
		L=freq[i]
		idx=int(L/freq[1])
		CULl=array_revshift(CU,idx,CU[:,CU.shape[1]/2])
		CTLl=array_revshift(CT,idx,CT[:,CU.shape[1]/2])
		fcutALl=array_revshift(fcutA,idx,fcutA[:,CU.shape[1]/2])
		f=np.power(2*np.pi,2)*L*(freq*CU+(L-freq)*CULl)
		F=f/(CT*CTLl)
		F[:,0]=0
		temp=F*f
		temp[fcutA==0]=0
		temp[fcutALl==0]=0
		temp2=np.sum(temp,axis=1)
		AL[:,i]=np.power(2*np.pi*L,2)/(freq[1]*temp2)
		AL[temp2==0,i]=0

	AL[:,0]=0
	AL[AL==float('inf')]=0
	return(AL)

##Find weighting function from Power Spectrum and Frequency Cutoff
#def AL_find2(CU,CT,freq,fcutA):
#	temp1=freq*CU*fcutA
#	temp1[:,0]=0
#	temp2=fcutA/CT
#	temp2[:,0]=0

#	temp3=np.power(2*np.pi,2)*np.fft.fft(np.fft.ifft(np.power(temp1,2)*temp2,axis=1)*np.fft.ifft(temp2,axis=1)+np.fft.ifft(temp1*temp2,axis=1)**2,axis=1)
#	AL=freq/temp3
#	AL[:,0]=0
#	AL[AL==float('inf')]=0
#	return(AL)

def AL_find2(CU,CT,CUR,CTR,freq,fcutA,fcutArev,fft1,fft2,fftF,fftB):
	fft2[:]=np.power(freq*CU,2)*(fcutA/CT)
	fftB()
	temp1=np.zeros(fft1.shape,dtype=complex)
	temp1[:]=fft1[:]
	fft2[:]=fcutArev/CTR
	fftB()
	temp1*=fft1[:]
	fft2[:]=np.power(freq*CUR,2)*(fcutArev/CTR)
	fftB()
	temp2=np.zeros(fft1.shape,dtype=complex)
	temp2[:]=fft1[:]
	fft2[:]=fcutA/CT
	fftB()
	temp2*=fft1[:]
	temp1+=temp2
	fft2[:]=(freq*CU)*(fcutA/CT)
	fftB()
	temp2[:]=fft1[:]
	fft2[:]=(freq*CUR)*(fcutArev/CTR)
	fftB()
	fft1*=2*temp2
	fft1+=temp1
	fftF()
	#temp1=np.fft.ifft(np.power(freq*CU,2)*(fcutA/CT),axis=1)
	#temp2=np.fft.ifft(np.power(freq*CUR,2)*(fcutArev/CTR),axis=1)
	#temp3=np.fft.ifft(fcutArev/CTR,axis=1)
	#temp4=np.fft.ifft(fcutA/CT,axis=1)
	#temp5=np.fft.ifft((freq*CU)*(fcutA/CT),axis=1)
	#temp6=np.fft.ifft((freq*CUR)*(fcutArev/CTR),axis=1)

	temp1[:]=1./((4*np.pi**2)*fft2)
	temp1*=fcutA
	return(np.nan_to_num(np.real(temp1)))

##Determine cutoff frequency where C<N
def cutoff(C,N,freq):
	fcut=np.zeros(C.shape[0])
	for i in range(C.shape[0]):
		fcut[i]=min(.5,np.abs(2*freq[np.abs(C[i,:]-N)==np.abs(C[i,:]-N).min()][0])+freq[1])
	fcut[fcut<freq[1]]=freq[1]
	##Freq Cutoff Array
	fcutA=np.ones(C.shape)
	for i in range(C.shape[0]):
		fcutA[i,np.abs(freq)>=fcut[i]]=0
	return(fcut,fcutA)

##Inverse Variance Weighting of lenses (discarding tau=0)
def NseWt(ds2,ds3,fftds23,fftds32,N):
	s=np.std(N,axis=1)
	fftds23()
	ds3/=np.sqrt(ds2.shape[1])
	N[N<1e-10]=0
	temp=ds3[s>0,:]/N[s>0,:]
	temp[N[s>0,:]==0]=0
	FL2=np.sum(temp,axis=0)
	temp=1/N[s>0,:]
	temp[N[s>0,:]==0]=0
	temp2=np.sum(temp,axis=0)
	FL2/=temp2
	FL2[temp2==0]=0
	L2=np.fft.ifft(FL2)*np.sqrt(ds2.shape[1])
	return(np.real(L2))


