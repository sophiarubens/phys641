import numpy as np
from matplotlib import pyplot as plt
from sigpyproc.readers import FilReader
from astropy.time import Time

# it seems like pulsar spectral indices are never steeper than about -4 ... from that paper

data3=FilReader('data_261215947.fil')
_, data3mask=data3.clean_rfi(method='mad',threshold=3.)
data3_masked=FilReader('data_261215947_masked.fil') # use the RFI-flagged version of the data

data3head=data3.header
data3_t0=data3head.tstart # reference/ start time of observations
data3_dt=data3head.tsamp # time step between observations
data3_ns=data3head.nsamples # number of samples
data3_phases=np.arange(0,data3_ns) # index phase measurements
data3_times0=data3_phases*data3_dt
data3_times=data3_t0+data3_times0 # vector of times is a linear function of the phase index
data3_nchan=data3head.nchans # number of frequency channels
data3_fch1=data3head.fch1 # frequency of initial channel
data3_df=data3head.foff # frequency offset of channels
data3_nsamp=data3head.nsamples # number of samples per channel = number of time steps of observation
data3_channels=np.arange(0,data3_nchan) # index channel numbers
data3_freqs=data3_fch1+data3_channels*data3_df # frequency vector is a linear function of the channel number, in MHz
data3_maskedblock=data3_masked.read_block(0,data3_ns) # data is accessible by reading a block
data3_filedata=data3_maskedblock.data

tfac=16 # use slightly less downsampling than in the example ... sacrifice a bit of evaluation speed for more finely sampled results
data3_masked.downsample(tfactor=tfac) # downsample by a factor of tfactor
data3_masked_downsampled=FilReader('data_261215947_masked_f1_t'+str(tfac)+'.fil')
data3mdhead=data3_masked_downsampled.header # md is short for _masked_downsampled
data3md_dt=data3mdhead.tsamp
data3md_ns=data3mdhead.nsamples 
data3md_phases=np.arange(0,data3md_ns) 
data3md_times0=data3_phases*data3_dt
data3md_times=data3_t0+data3_times0 
data3_downsampled_block=data3_masked_downsampled.read_block(0,data3md_ns)
data3_masked_downsampled_normalized=data3_downsampled_block.normalise()  # normalize data within each channel
pnormeddata=data3_masked_downsampled_normalized.data

data3_t0obj=Time(data3_t0,format='mjd')
data3_t0iso=data3_t0obj.iso
hires=500

plt.figure(figsize=(10,5))
plt.imshow(pnormeddata,extent=[data3_times0[0],data3_times0[-1],data3_freqs[-1],data3_freqs[0]],aspect=1e-2,vmin=np.percentile(pnormeddata,1),vmax=np.percentile(pnormeddata,99))
plt.xlabel('time (s) after '+data3_t0iso)
plt.ylabel('frequency (MHz)')
plt.title('Normalized part 3 data for ID 261215947')
cbar=plt.colorbar()
cbar.set_label('Flux (ADU)')
plt.tight_layout()
plt.savefig('normed_261215947_data.png',dpi=hires)
plt.show()

# setup for matched filter loops
spectral_index=-1.4 # mean finding from doi:10.1093/mnras/stt257

start_time_lo=data3_t0 # do NOT work with zero-based times; use the actual values
start_time_hi=data3_times[-1]

period_lo=1.4e-3
log_period_lo=np.log(period_lo)
period_hi=8.5
log_period_hi=np.log(period_hi)
n_periods_to_test=100
periods=np.logspace(period_lo,period_hi,n_periods_to_test) # https://www.cv.nrao.edu/~sransom/web/Ch6.html ; logspace needs powers of ten

