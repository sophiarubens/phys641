import numpy as np
from matplotlib import pyplot as plt
from sigpyproc.readers import FilReader
from astropy.time import Time

part1plots=False

def multiply_columnwise(A,b):
    '''for each column in matrix A, multiply elementwise by vector b'''
    # assert(A.shape[0]==b.shape[0]), "A must have the same number of elements per column (i.e. rows) as b has entries"
    return (A.T*b).T

# PART 1
# 1. plot the source spectrum over the CHIME frequency band
npts=5002
alpha=-0.6
chimelo=400e6 # frequencies in Hz
chimehi=800e6
nu=np.linspace(chimelo,chimehi,npts) # vector of CHIME observing frequencies
S400=8.684 # spectral flux at 400 Jy
S0=S400/(chimelo**(alpha)) # if I describe my power law as S(nu) = S0*nu**alpha, S0 is the normalization factor
S=S0*nu**alpha
if part1plots:
    plt.figure()
    plt.plot(nu/1e6,S) # plot the frequencies in MHz to prioritize legibility
    plt.title('spectrum of calibrator source 3C 129 over the CHIME frequency band')
    plt.xlabel('frequency (MHz)')
    plt.ylabel('spectral flux (Jy)')
    plt.savefig('calspectheo.png')
    plt.show()

# 2. import and inspect the calibrator data
def importfil(fname,verbose=False):
    file=FilReader(fname)
    fheader=file.header
    if verbose: # verbosity flag controls header printing. good for intuition and groundwork; less so for minimizing terminal clutter during subsequent debugging
        print(fheader)
    t0=fheader.tstart # reference/ start time of observations
    dt=fheader.tsamp # time step between observations
    ns=fheader.nsamples # number of samples
    phases=np.arange(0,ns) # index phase measurements
    times=t0+phases*dt # vector of times is a linear function of the phase index
    nchan=fheader.nchans # number of frequency channels
    fch1=fheader.fch1 # frequency of initial channel
    df=fheader.foff # frequency offset of channels
    nsamp=fheader.nsamples # number of samples per channel = number of time steps of observation
    channels=np.arange(0,nchan) # index channel numbers
    freqs=fch1+channels*df # frequency vector is a linear function of the channel number
    fileblock=file.read_block(0,ns) # data is accessible by reading a block
    filedata=fileblock.data
    filespec=np.mean(filedata,axis=1) # average over phase to get the data as a function of frequency
    return file,fheader,times,t0,dt, ns,nchan,fch1,df,nsamp,channels,freqs,fileblock,filedata,filespec

# below: all the return values described above, but prefaced with "cal," for "calibrator"
cal,    calheader, caltimes, calt0, caldt, calns, calnchan, calfch1, caldf, calnsamp, calchannels, calfreqs, calblock,  caldata,  calspec=importfil('calibrator_source.fil',verbose=False)
_, calmask=cal.clean_rfi(method='mad',threshold=2)
cal,    calheader, caltimes, calt0, caldt, calns, calnchan, calfch1, caldf, calnsamp, calchannels, calfreqs, calblock,  caldata,  calspec=importfil('calibrator_source_masked.fil',verbose=False)

# amplitude will be too high where the cal spec has values near zero ... overwrite using a mask
caldata=cal.read_block(0,calheader.nsamples,calheader.fch1,calheader.nchans)
caldataarray = caldata.data
cal.compute_stats()
freqmask=cal.chan_stats.mean==0
calmasked=np.where(~freqmask,cal.chan_stats.mean,np.nan)

if part1plots:
    plt.figure(figsize=(10,5))
    calfreqs=calheader.chan_freqs
    plt.plot(calfreqs, calmasked) # calmasked
    plt.xlabel('frequency (MHz)')
    plt.ylabel('mean ADU over all '+str(calns)+' phase measurements')
    plt.title('average spectrum of calibrator source 3C 129')
    plt.savefig('calspecraw.png')
    plt.show()

# 3. convert ADC counts to Jy using a transfer function
theospec_datares=S0*(calfreqs*1e6)**alpha # theoretical spectrum sampled at the same frequencies as the CHIME calibrator spectrum (i.e. "data resolution")
calspeczeros=np.nonzero(calspec==0)
calspec[calspeczeros]=np.inf # set the entries in calspec w/ zeros to have infinity there to prevent division errors
transfer=theospec_datares/calmasked
transfer[calspeczeros]=np.nan # we don't want the transfer function to have zeros in the problematic places; nans are more intuitive

if part1plots:
    plt.figure(figsize=(10,5))
    plt.plot(calfreqs,transfer)
    plt.xlabel('frequency (MHz)')
    plt.ylabel('transfer amplitude (Jy/ADU)')
    plt.title('transfer function between the theoretical spectrum and CHIME calibrator observation')
    plt.tight_layout()
    plt.savefig('transfer.png')
    plt.show()

calt0obj=Time(calt0,format='mjd') # astropy.time.Time object version of the MJD time t0
calt0iso=calt0obj.iso # ISO format is more intuitively human-readable: YYYY-MM-DD HH:MM:SS.sss
caltimes0=np.arange(0,calns)*caldt # vector of times in the calibration vector
transferred_caldata=multiply_columnwise(caldata.data,transfer)
ceil=99
hires=500
if part1plots:
    plt.figure(figsize=(10,5))
    plt.imshow(transferred_caldata,extent=[caltimes0[0],caltimes0[-1],calfreqs[-1],calfreqs[0]],aspect=1e-2,vmin=np.nanpercentile(transferred_caldata,0),vmax=np.nanpercentile(transferred_caldata,ceil)) # HERE AND IN OTHER WATERFALL PLOT IMSHOWS: extent=[left,right,bottom,top]; vmin and vmax set using the 0th and 100th percentiles of the data being imshown //,vmin=np.nanpercentile(transferred_caldata,0),vmax=np.nanpercentile(transferred_caldata,100)
    plt.xlabel('time (s) after '+calt0iso)
    plt.ylabel('frequency (MHz)')
    plt.title('Calibrator source 3C 129 waterfall plot')
    cbar=plt.colorbar()
    cbar.set_label('Mean flux density (Jy)')
    plt.tight_layout()
    plt.savefig('cal3waterfall.png',dpi=hires)
    plt.show()

# 4. apply the transfer function to the blank sky 
# below: similar to when I imported the calibrator data, but this time with "bsk," for "blank sky"
bsk,    bskheader, bsktimes, bskt0, bskdt, bskns, bsknchan, bskfch1, bskdf, bsknsamp, bskchannels, bskfreqs, bskblock,  bskdata,  bskspec=importfil('blank_sky.fil')
_, bskmask=bsk.clean_rfi(method='mad',threshold=2)
bsk,    bskheader, bsktimes, bskt0, bskdt, bskns, bsknchan, bskfch1, bskdf, bsknsamp, bskchannels, bskfreqs, bskblock,  bskdata,  bskspec=importfil('blank_sky_masked.fil')

bskmasked=bskspec*transfer
bskmasked[np.nonzero(bskmasked==0)]=np.nan
if part1plots:
    plt.figure(figsize=(10,5))
    plt.plot(bskfreqs,bskmasked) # apply the calibration transfer function when plotting the average spectrum
    plt.xlabel('frequency (MHz)')
    plt.ylabel('Mean flux density (Jy), averaged over all '+str(bsknsamp)+' phase measurements')
    plt.title('average spectrum of the blank sky')
    plt.savefig('bskspec.png')
    plt.show()

bskt0obj=Time(bskt0,format='mjd') # same time conversion as for the calibrator
bskt0iso=bskt0obj.iso 
bsktimes0=np.arange(0,bskns)*bskdt
transferred_bskdata=multiply_columnwise(bskdata,transfer)
if part1plots:
    plt.figure(figsize=(10,5))
    plt.imshow(transferred_bskdata,extent=[bsktimes0[0],bsktimes0[-1],bskfreqs[-1],bskfreqs[0]],aspect=1e-2,vmin=np.nanpercentile(transferred_bskdata,0),vmax=np.nanpercentile(transferred_bskdata,ceil)) # extent=[left,right,bottom,top] ## ,vmin=np.nanpercentile(transferred_bskdata,0),vmax=np.nanpercentile(transferred_bskdata,100)
    plt.xlabel('time (s) after '+bskt0iso)
    plt.ylabel('frequency (MHz)')
    plt.title('transfer function–aware waterfall plot of the blank sky')
    cbar=plt.colorbar()
    cbar.set_label('Mean flux density (Jy)')
    plt.tight_layout()
    plt.savefig('bskwaterfall.png',dpi=hires)
    plt.show()

# 5. conduct statistical analysis on a per-channel basis
normedbskdata=0*bskdata # create a holder of the same size as the phase-and-frequency–indexed data set
channelmeans=0*bskfreqs
channelstds=0*bskfreqs
for i,channel in enumerate(bskdata): # look at the stats one channel at a time
    mean=np.mean(channel)
    channelmeans[i]=mean
    std=np.std(channel)
    channelstds[i]=std
    if std==0: # hard-code protection against division-by-zero errors
        normedbskdata[i,:]=np.nan
    else:
        normedbskdata[i,:]=(channel-mean)/std

bskscalednormed=multiply_columnwise(normedbskdata,transfer)
mod=15
if part1plots:
    plt.figure(figsize=(10,5))
    plt.imshow(bskscalednormed,extent=[bsktimes0[0],bsktimes0[-1],bskfreqs[-1],bskfreqs[0]],aspect=1e-2,norm='log') #,vmin=np.nanpercentile(bskscalednormed,mod),vmax=np.nanpercentile(bskscalednormed,100-mod)) # extent=[left,right,bottom,top]
    plt.xlabel('time (s) after '+bskt0iso)
    plt.ylabel('frequency (MHz)')
    plt.title('Normalized and transfer function–aware waterfall plot of the blank sky')
    cbar=plt.colorbar()
    cbar.set_label('Mean flux density (Jy)')
    plt.tight_layout()
    plt.savefig('bsknormedwaterfall.png',dpi=hires)
    plt.show()

bskscalednormedspec=np.sum(bskscalednormed,axis=1)
if part1plots:
    nchan=bskscalednormed.shape[0]
    for i in range(nchan): # keep looping over channels until I find one that gives a suitable histogram
        chan=bskscalednormed[i,:] 
        if (np.isnan(chan).any()):
            pass
        else:
            print('suitable chan = ',chan)
            plt.figure()
            plt.hist(chan,bins=50) #,bins='auto'
            plt.xlabel('time bin')
            plt.ylabel('number of time steps in the channel of interest')
            plt.title('channel '+str(i)+' histogram')
            plt.savefig('histogram.png')
            plt.show()
            break

#PART 2
# 1. SNR math
S400=0.15
alpha=-1.5
S0=S400*(400e6)**(-alpha) # S=S0*nu**alpha; S400=S0*(400e6)**(-1.5) -> S0=S400*(400e6)**1.5
# print('S0',S0)
Spulsar=S0*(calfreqs*1e6)**alpha # the units work out if I use the Hz version (not MHz)
plt.figure()
plt.plot(calfreqs,Spulsar)
plt.title('check theo pulsar power law')
plt.xlabel('frequency (MHz)')
plt.ylabel('spectral flux density (Jy)')
plt.show()
calmaskchanmask=calmask.chan_mask
bskmaskchanmask=bskmask.chan_mask
Spulsar[calmaskchanmask]=np.nan # CHECK!! this might be flipped or the wrong thing
Spulsar[bskmaskchanmask]=np.nan

def SNR(Spulsar,N,Ssys):
    return Spulsar*np.sqrt(N)/Ssys

SNR1pulse=SNR(Spulsar,1,bskmasked)
snr1mean=np.nanmean(SNR1pulse)
snr1max=np.nanmax(SNR1pulse)
snr1min=np.nanmin(SNR1pulse)

plottheospec=False
if plottheospec:
    plt.figure()
    plt.plot(calfreqs,SNR1pulse)
    plt.xlabel('frequency (MHz)')
    plt.ylabel('dimensionless, unitless SNR proportionality')
    plt.title('RHS of SNR proportionality for N=1 and the POWER LAW pulsar spectrum')
    plt.savefig('one_pulse_theo_snr.png')
    plt.show()

meanSNR1pulse=np.nanmean(SNR1pulse)
N_from_mean=(2./snr1mean)**2 # factor_by_which_you_need_to_increase_snr=2./meanSNR1 = sqrt(N_required) -> N_required = (2./meanSNR1)**2
N_from_max=(2./snr1max)**2
N_from_min=(2./snr1min)**2
print('To achieve SNR~2, you observe x pulses / fold the data x times')
print('Based on the mean one-pulse SNR,',np.round(snr1mean,4),', x=',round(N_from_mean))
print('Based on the max  one-pulse SNR,',np.round(snr1max,4),', x=',round(N_from_max))
print('Based on the min  one-pulse SNR,',np.round(snr1min,4),', x=',round(N_from_min))

# 2. load and normalize pulsar data
# pfil=FilReader('pulsardata.fil')
# _, pmask=pfil.clean_rfi(method='mad',threshold=3.) ## PREVIOUSLY, I WAS NOT SPECIFYING mask_value
# pfil=FilReader('pulsardata_masked.fil') # use the RFI-flagged version of the data
# pns=pfil.header.nsamples # number of samples
# pfileblock=pfil.read_block(0,pns) # data is accessible by reading a block
# pfiledata=pfileblock.data
# pulsarspec_adu=np.mean(pfiledata,axis=1) # average over phase to get the spectral data as a function of frequency
# pspecaduzero=np.nonzero(pulsarspec_adu==0)
# pulsarspec_adu[pspecaduzero]=np.nan

# # transfer[np.nonzero(transfer==0)]=np.nan # try this to see if it helps remove the weird drops down to zero in the #2 SNR plot that show up even though I did RFI flagging

pfil=FilReader('pulsardata.fil')
_, pmask=pfil.clean_rfi(method='mad',threshold=3.,mask_value=0) ## PREVIOUSLY, I WAS NOT SPECIFYING mask_value
pfil=FilReader('pulsardata_masked.fil') # use the RFI-flagged version of the data
phead=pfil.header
pt0=phead.tstart # reference/ start time of observations
pdt=phead.tsamp # time step between observations
pns=phead.nsamples # number of samples
pphases=np.arange(0,pns) # index phase measurements
ptimes=pt0+pphases*pdt # vector of times is a linear function of the phase index
pnchan=phead.nchans # number of frequency channels
pfch1=phead.fch1 # frequency of initial channel
pdf=phead.foff # frequency offset of channels
pnsamp=phead.nsamples # number of samples per channel = number of time steps of observation
pchannels=np.arange(0,pnchan) # index channel numbers
pfreqs=pfch1+pchannels*pdf # frequency vector is a linear function of the channel number, in MHz
pfileblock=pfil.read_block(0,pns) # data is accessible by reading a block
pfiledata=pfileblock.data
pulsarspec_adu=np.mean(pfiledata,axis=1) # average over phase to get the spectral data as a function of frequency
pspecaduzero=np.nonzero(pulsarspec_adu==0)
pulsarspec_adu[pspecaduzero]=np.nan

transfer[np.nonzero(transfer==0)]=np.nan # try this to see if it helps remove the weird drops down to zero in the #2 SNR plot that show up even though I did RFI flagging

checkfreqs=False
if checkfreqs:
    print("pfreqs==calfreqs?",(pfreqs==calfreqs).all())
    print('calfreqs==bskfreqs?',(calfreqs==bskfreqs).all()) # since this prints out True, there's no need to worry about different spectra originally prepared at the calibrator / blank sky frequencies having different frequency values for the same array indices ... the sort of less hacky way to do this would have been to check that the f0, df, and nchan were all the same

################
pfil.downsample(tfactor=32)
pfil32=FilReader('pulsardata_masked_f1_t32.fil') # off_Fil_32 = FilReader("/home/jovyan/work/phys641data/Data/blank_sky_masked_f1_t32.fil") 
p32block=pfil32.read_block(0,pfil32.header.nsamples, pfil32.header.fch1, pfil32.header.nchans) # off_data_32 = off_Fil_32.read_block(0, off_Fil_32.header.nsamples, off_Fil_32.header.fch1, off_Fil_32.header.nchans)
pnormed=p32block.normalise()  # normalize data within each channel
################

# pnormed=pfileblock.normalise()
print('done normalizing')
pnormeddata=pnormed.data

plt.figure(figsize=(10,5))
plt.imshow(pnormeddata,extent=[bsktimes0[0],bsktimes0[-1],bskfreqs[-1],bskfreqs[0]],aspect=1e-2,vmin=np.percentile(pnormeddata,1),vmax=np.percentile(pnormeddata,99))
plt.xlabel('time (s) after '+calt0iso)
plt.ylabel('frequency (MHz)')
plt.title('Normalized pulsar data')
cbar=plt.colorbar()
cbar.set_label('Flux (ADU)')
plt.tight_layout()
plt.show()

# 3. DM transform
fterm=1/(400.**2)-1/(800.**2) # GHz; tau = kDM*DM*fterm so deltatau = kDM*deltaDM*fterm -> deltaDM = deltatau/(kDM*fterm)
kDM=4148.8 # MHz**2 pc**{-1} cm**3 s
ctrdm=30 # dmt_transform searches a range of dms symmetric about the specified center, according to the source code, so I'll provide the center of this region and use this center to figure out the number of steps required to get my desired spacing (set by the limit of the instrument) dm_arr = dm + np.linspace(-dm, dm, dmsteps)
deltaDM=pdt/(kDM*fterm) # pc cm**{-3} **desired spacing for the dm search
print('deltaDM=',deltaDM)
DMrange=2*ctrdm # want to search 10 to 50, but dmt_transform searches 0 to 2*dm, and I'd rather have symmetric regions outside the most likely one (because I don't know if an abnormally low or high dm is more likely), so I call with 30 as the center and not 25 or anything else
ndmsteps=8
dm_t=pnormed.dmt_transform(ctrdm,dmsteps=ndmsteps) # massively scaled down version that runs quickly (useful for figuring out the rest of the steps)
# ndmsteps=int(DMrange/deltaDM)
# print('number of DM steps to take at the minimum sensible spacing =',ndmsteps)
# dm_t=pfileblock.dmt_transform(ctrdm,dmsteps=ndmsteps) # full call that takes 40ish minutes to run
# ^^ initial guess for dm = median of the range 10 to 50 that is apparently characteristic of pulsars

pt0obj=Time(pt0,format='mjd')
pt0iso=pt0obj.iso
ptimes0=np.arange(pns)*pdt # like caltimes0=np.arange(0,calns)*caldt
plt.figure(figsize=(10,5))
plt.imshow(dm_t.data,extent=[ptimes0[0],ptimes0[-1],2*ctrdm,0],aspect=0.1) # L,R,B,T
cbar=plt.colorbar()
cbar.set_label('S/N')
plt.xlabel('time (s) after '+pt0iso)
plt.ylabel('DM ')
plt.title('DM-time grid')
plt.show()

# 4. search for periodic signal in Fourier space
dm_f=np.fft.rfft(dm_t.data).real # we don't care about the imag part bc the rfft of a real-valued array is purely real (no risk of losing info here)
pfreqs=np.fft.fftfreq(pns,d=pdt) # as many frequencies as number of time samples, from a time array with spacing pdt
plt.figure(figsize=(10,5))
plt.imshow(dm_f,extent=[pfreqs[0],pfreqs[-1],2*ctrdm,0],aspect=1e-3,norm='log',vmax=np.percentile(dm_f,99))
cbar=plt.colorbar()
cbar.set_label('S/N')
plt.xlabel('frequency (Hz), calculated from time (s) after '+pt0iso)
plt.ylabel('DM')
plt.title('DM-frequency grid')
plt.show()
bestloc=np.unravel_index(dm_f.argmin(), dm_f.shape)
print('maximal SNR is',dm_f[bestloc])
best_freq=pfreqs[bestloc[0]]
print('SNR-maximizing freq is',best_freq)
best_dm=bestloc[1]*deltaDM
print('SNR-maximizing DM is',best_dm)
assert(1==0)

# 5. fold data to find pulse
pfolded=pfil.fold(best_freq,best_dm,nints=1,nbands=1024,nbins=int(0.5//pdt)) # positional args are ordered as period,dm
# off_data_32_folded = off_Fil_32.fold(0.5,2,nints=1,nbands=1024,nbins=int(0.5//off_Fil_32.header.tsamp))
# plt.figure()
# plt.imshow(off_data_32_folded.data[0,:,:],aspect='auto',interpolation='nearest')
# plt.colorbar()
# plt.xlabel('Time [samples]')
# plt.ylabel('Freq. [channel]')
# plt.show()

# Tldr upshot of a 208 discussion: internally, dmt_transform creates dm_arr = dm + np.linspace(-dm, dm, dmsteps), so manipulate your Δdm accordingly