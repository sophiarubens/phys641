import numpy as np
from matplotlib import pyplot as plt
import sigpyproc
from sigpyproc.readers import FilReader
from astropy.time import Time

def multiply_columnwise(A,b):
    '''for each column in matrix A, multiply elementwise by vector b'''
    # assert(A.shape[0]==b.shape[0]), "A must have the same number of elements per column (i.e. rows) as b has entries"
    return (A.T*b).T

# 1. plot the source spectrum over the CHIME frequency band
npts=5002
alpha=-0.6
chimelo=400e6 # frequencies in Hz
chimehi=800e6
nu=np.linspace(chimelo,chimehi,npts) # vector of CHIME observing frequencies
S400=8.684 # spectral flux at 400 Jy
S0=S400/(chimelo**(alpha)) # if I describe my power law as S(nu) = S0*nu**alpha, S0 is the normalization factor
S=S0*nu**alpha
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

# amplitude will be too high where the cal spec has values near zero ... overwrite using a mask
caldata=cal.read_block(0,calheader.nsamples,calheader.fch1,calheader.nchans)
caldataarray = caldata.data
cal.compute_stats()
freqmask=cal.chan_stats.mean==0
calmasked=np.where(freqmask,cal.chan_stats.mean,np.nan)

plt.figure()
calfreqs=calheader.chan_freqs
plt.plot(calfreqs, cal.chan_stats.mean) # not calmasked yet
plt.xlabel('frequency (MHz)')
plt.ylabel('mean ADU over all '+str(calns)+' phase measurements')
plt.title('average spectrum of calibrator source 3C 129')
plt.savefig('calspecraw.png')
plt.show()


# 3. convert ADC counts to Jy using a transfer function
# print('calfreqs=',calfreqs)
theospec_datares=S0*(calfreqs*1e6)**alpha # theoretical spectrum sampled at the same frequencies as the CHIME calibrator spectrum (i.e. "data resolution")
# print('min and max of theospec_datares:',np.min(theospec_datares),np.max(theospec_datares))
# print('min and max of calspec:',np.min(calspec),np.max(calspec))
calspeczeros=np.nonzero(calspec==0)
calspec[calspeczeros]=np.inf # set the entries in calspec w/ zeros to have infinity there to prevent division errors
transfer=theospec_datares/calspec
transfer[calspeczeros]=np.nan # we don't want the transfer function to have zeros in the problematic places; nans are more intuitive

plt.figure()
plt.plot(calfreqs,transfer)
plt.xlabel('frequency (MHz)')
plt.ylabel('transfer amplitude (Jy/ADU)')
plt.title('transfer function between the theoretical spectrum and CHIME calibrator observation')
plt.tight_layout()
plt.savefig('transfer.png')
plt.show()

plt.figure(figsize=(10,5))
calt0obj=Time(calt0,format='mjd') # astropy.time.Time object version of the MJD time t0
calt0iso=calt0obj.iso # ISO format is more intuitively human-readable: YYYY-MM-DD HH:MM:SS.sss
caltimes0=np.arange(0,calns)*caldt # vector of times in the calibration vector
transferred_caldata=multiply_columnwise(caldata.data,transfer)
floor=1
ceil=99 # too high: 99.999; 99.9 marginally better
plt.imshow(transferred_caldata,extent=[caltimes0[0],caltimes0[-1],calfreqs[-1],calfreqs[0]],aspect=1e-2,vmin=np.nanpercentile(transferred_caldata,0),vmax=np.nanpercentile(transferred_caldata,ceil)) # HERE AND IN OTHER WATERFALL PLOT IMSHOWS: extent=[left,right,bottom,top]; vmin and vmax set using the 0th and 100th percentiles of the data being imshown //,vmin=np.nanpercentile(transferred_caldata,0),vmax=np.nanpercentile(transferred_caldata,100)
plt.xlabel('time (s) after '+calt0iso)
plt.ylabel('frequency (MHz)')
plt.title('Calibrator source 3C 129 waterfall plot')
cbar=plt.colorbar()
cbar.set_label('flux density (Jy)')
plt.tight_layout()
plt.savefig('cal3waterfall.png')
plt.show()

# 4. apply the transfer function to the blank sky 
# below: similar to when I imported the calibrator data, but this time with "bsk," for "blank sky"
bsk,    bskheader, bsktimes, bskt0, bskdt, bskns, bsknchan, bskfch1, bskdf, bsknsamp, bskchannels, bskfreqs, bskblock,  bskdata,  bskspec=importfil('blank_sky.fil')

plt.figure()
plt.plot(bskfreqs,bskspec*transfer) # apply the calibration transfer function when plotting the average spectrum
plt.xlabel('frequency (MHz)')
plt.ylabel('flux density (Jy), averaged over all '+str(bsknsamp)+' phase measurements')
plt.title('average spectrum of the blank sky')
plt.savefig('bskspec.png')
plt.show()

plt.figure(figsize=(10,5))
bskt0obj=Time(bskt0,format='mjd') # same time conversion as for the calibrator
bskt0iso=bskt0obj.iso 
bsktimes0=np.arange(0,bskns)*bskdt
transferred_bskdata=multiply_columnwise(bskdata,transfer)
plt.imshow(transferred_bskdata,extent=[bsktimes0[0],bsktimes0[-1],bskfreqs[-1],bskfreqs[0]],aspect=1e-2,vmin=np.nanpercentile(transferred_bskdata,0),vmax=np.nanpercentile(transferred_bskdata,ceil)) # extent=[left,right,bottom,top] ## ,vmin=np.nanpercentile(transferred_bskdata,0),vmax=np.nanpercentile(transferred_bskdata,100)
plt.xlabel('time (s) after '+bskt0iso)
plt.ylabel('frequency (MHz)')
plt.title('transfer function–aware waterfall plot of the blank sky')
cbar=plt.colorbar()
cbar.set_label('flux density (Jy)')
plt.tight_layout()
plt.savefig('bskwaterfall.png')
plt.show()

# 5. conduct statistical analysis on a per-channel basis
normedbskdata=0*bskdata # create a holder of the same size as the phase-and-frequency–indexed data set
for i,channel in enumerate(bskdata): # look at the stats one channel at a time
    mean=np.mean(channel)
    std=np.std(channel)
    if std==0: # hard-code protection against division-by-zero errors
        normedbskdata[i,:]=np.nan
    else:
        normedbskdata[i,:]=(channel-mean)/std

plt.figure(figsize=(10,5))
bskscalednormed=multiply_columnwise(normedbskdata,transfer)
mod=0.5
plt.imshow(bskscalednormed,extent=[bsktimes0[0],bsktimes0[-1],bskfreqs[-1],bskfreqs[0]],aspect=1e-2,vmin=np.nanpercentile(bskscalednormed,0.1),vmax=np.nanpercentile(bskscalednormed,99.9)) # extent=[left,right,bottom,top]
plt.xlabel('time (s) after '+bskt0iso)
plt.ylabel('frequency (MHz)')
plt.title('Normalized and transfer function–aware waterfall plot of the blank sky')
cbar=plt.colorbar()
cbar.set_label('flux density (Jy)')
plt.tight_layout()
plt.savefig('bsknormedwaterfall.png')
plt.show()