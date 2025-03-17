import numpy as np
from matplotlib import pyplot as plt
from sigpyproc.readers import FilReader
from astropy.time import Time

# it seems like pulsar spectral indices are never steeper than about -4 ... from that paper

# id_of_interest=261215947 # me
id_of_interest=260374104 # test w/ a random classmate's data ... formalize this later with a loop over all or something
data3=FilReader('data_'+str(id_of_interest)+'.fil')
_, data3mask=data3.clean_rfi(method='mad',threshold=3.)
data3_masked=FilReader('data_'+str(id_of_interest)+'_masked.fil') # use the RFI-flagged version of the data

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
data3_masked_downsampled=FilReader('data_'+str(id_of_interest)+'_masked_f1_t'+str(tfac)+'.fil')
data3mdhead=data3_masked_downsampled.header # md is short for _masked_downsampled
data3md_dt=data3mdhead.tsamp
data3md_ns=data3mdhead.nsamples 
data3md_phases=np.arange(0,data3md_ns) 
data3md_times0=data3md_phases*data3md_dt
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
plt.title('Normalized part 3 data for ID '+str(id_of_interest))
cbar=plt.colorbar()
cbar.set_label('Flux (ADU)')
plt.tight_layout()
plt.savefig('normed_'+str(id_of_interest)+'_data.png',dpi=hires)
plt.show()

# setup for matched filter loops
spectral_index=-1.4 # mean finding from doi:10.1093/mnras/stt257

fterm=1/(400.**2)-1/(800.**2) # GHz; tau = kDM*DM*fterm so deltatau = kDM*deltaDM*fterm -> deltaDM = deltatau/(kDM*fterm)
kDM=4148.8 # MHz**2 pc**{-1} cm**3 s
deltaDM=data3md_dt/(kDM*fterm) # pc cm**{-3} **desired spacing for the dm search
dm_lo=-50
dm_hi=0
DMrange=dm_hi-dm_lo 
n_dms=int(DMrange/deltaDM)
print('n_dms=',n_dms)
dm_vec=np.linspace(dm_lo,dm_hi,n_dms) # use galactic DM magnitudes, but negative
dm_t_array=np.zeros((n_dms,data3md_ns)) # dmt_transform-like array, to populate manually
print('dm_t_array.shape=',dm_t_array.shape)

for i,dm in enumerate(dm_vec): # consider the DM candidates
    dedisp_array = data3_masked_downsampled_normalized.dedisperse(dm) # get 2D array dedispersed by the candidate amount
    dm_t_array[i,:] = dedisp_array.get_tim().data # sum over frequency channels to get the dedispersed data for that time and DM

bestloc=np.unravel_index((np.abs(dm_t_array)).argmax(), dm_t_array.shape)
best_start_time=data3md_times0[bestloc[1]]
print('SNR-maximizing start time is',best_start_time)
best_dm=dm_vec[bestloc[0]]
print('SNR-maximizing DM is',best_dm)

plt.figure(figsize=(10,5))
plt.imshow(dm_t_array,aspect=1e-1,extent=[data3md_times0[0],data3md_times0[-1],dm_vec[-1],dm_vec[0]]) # extent=[L,R,B,T]
plt.colorbar()
plt.scatter(best_start_time,best_dm)
plt.xlabel('time (s) after '+str(data3_t0iso))
plt.ylabel('DM (pc cm^{-3})')
plt.title('DM-time grid for '+str(id_of_interest))
plt.savefig('dm_t_array_'+str(id_of_interest)+'.png',dpi=hires)
plt.show() # since I only see one maximum, it's not worth searching in Fourier space ... doing a Fourier analysis would merely reveal that the zero-frequency component dominates (possibly with some ring-down)

# dm_f_array=np.abs(np.fft.rfft(dm_t_array))
# data3_freqs=np.fft.fftfreq(data3md_ns,d=data3md_dt) # as many frequencies as number of time samples, from a time array with the DOWNSAMPLED spacing
# plt.figure(figsize=(10,5))
# plt.imshow(dm_f_array)
# plt.colorbar()
# plt.xlabel('frequency (Hz), referenced to 1/time after time='+str(data3_t0iso))
# plt.ylabel('DM (pc cm^{-3})')
# plt.title('DM-frequency grid for '+str(id_of_interest))
# plt.show() # expect to see only the zero-freq component b/c we only see one instance of an alien signal...

#############
period_lo=np.max((best_start_time,10.-best_start_time)) # lower bound on period: longest stretch in the data before or after a negative-DM signal (whether it comes before or after the signal we observe isn't important)
log_period_lo=1. # the period can't be shorter than 10 s
log_period_hi=2.
n_periods_to_test=5
periods_to_test=np.logspace(log_period_lo,log_period_hi,n_periods_to_test)

log_width_lo=log_period_lo-3 # start w/ a lower bound of 3 OoM smaller than the shortest considered period, trying to acknowledge "The measured width of pulsar profiles is typically less than 10 per cent of the pulse period." https://doi.org/10.1111/j.1365-2966.2009.15926.x 
log_width_hi=log_period_hi-1 # in line with the statement in the paper referenced above
n_widths_to_test=100
widths_to_test=np.logspace(log_width_lo,log_width_hi,n_widths_to_test)

for i,test_period in enumerate(periods_to_test):
    n_fold_bins=int(test_period//data3md_dt)
    data3_folded_3d=data3_masked.fold(test_period,best_dm,nints=1,nbands=1024,nbins=n_fold_bins)
    data3_folded_2d=data3_folded_3d.data[0] # store the one integration we specified in 2d format
    data3_folded_1d=np.nanmean(data3_folded_2d,axis=0) # sum over frequency channels
    downsampled_folded_times=np.linspace(0,test_period,n_fold_bins)

    #
    plt.figure()
    plt.plot(downsampled_folded_times,data3_folded_1d)
    plt.xlabel('time (s)')
    plt.ylabel('S/N')
    plt.title('Frequency-averaged, folded pulse profile for period{:7.2f}'.format(test_period))
    plt.savefig('folded_1d.png',dpi=hires)
    plt.show()
    #