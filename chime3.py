import numpy as np
from matplotlib import pyplot as plt
from sigpyproc.readers import FilReader
from astropy.time import Time

# it seems like pulsar spectral indices are never steeper than about -4 ... from that paper

srubensid=261215947
data3=FilReader('data_'+str(srubensid)+'.fil')
_, data3mask=data3.clean_rfi(method='mad',threshold=3.)
data3_masked=FilReader('data_'+str(srubensid)+'_masked.fil') # use the RFI-flagged version of the data

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
data3_masked_downsampled=FilReader('data_'+str(srubensid)+'_masked_f1_t'+str(tfac)+'.fil')
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
plt.title('Normalized part 3 data for ID '+str(srubensid))
cbar=plt.colorbar()
cbar.set_label('Flux (ADU)')
plt.tight_layout()
plt.savefig('normed_'+str(srubensid)+'_data.png',dpi=hires)
plt.show()

# setup for matched filter loops
spectral_index=-1.4 # mean finding from doi:10.1093/mnras/stt257

start_time_lo=data3_t0 # do NOT work with zero-based times; use the actual values
start_time_hi=data3_times[-1]
n_start_times_to_test=100
start_times_to_test=np.linspace(start_time_lo,start_time_hi,n_start_times_to_test)

period_lo=1.4e-3
log_period_lo=np.log(period_lo)
period_hi=8.5
log_period_hi=np.log(period_hi)
n_periods_to_test=100 # use logspace b/c large dynamic range
periods_to_test=np.logspace(period_lo,period_hi,n_periods_to_test) # https://www.cv.nrao.edu/~sransom/web/Ch6.html ; logspace needs powers of ten

log_width_lo=log_period_lo-3 # start w/ a lower bound of 3 OoM smaller than the shortest considered period, trying to acknowledge "The measured width of pulsar profiles is typically less than 10 per cent of the pulse period." https://doi.org/10.1111/j.1365-2966.2009.15926.x 
log_width_hi=log_period_hi-1 # in line with the statement in the paper referenced above
n_widths_to_test=100
widths_to_test=np.logspace(log_width_lo,log_width_hi,n_widths_to_test)

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

for i,dm in enumerate(dm_vec): # consider the DM candidates
    dedisp_array = data3_masked_downsampled_normalized.dedisperse(dm) # get 2D array dedispersed by the candidate amount
    dm_t_array[i,:] = dedisp_array.get_tim().data # sum over frequency channels to get the dedispersed data for that time and DM

plt.figure(figsize=(10,5))
plt.imshow(dm_t_array,aspect=1e-1,extent=[data3md_times0[0],data3md_times0[-1],dm_vec[-1],dm_vec[0]]) # extent=[L,R,B,T]
plt.colorbar()
plt.xlabel('time (s) after '+str(data3_t0iso))
plt.ylabel('DM (pc cm^{-3})')
plt.title('DM-time grid for '+str(srubensid))
plt.show() # since I only see one maximum, it's not worth searching in Fourier space ... doing a Fourier analysis would merely reveal that the zero-frequency component dominates (possibly with some ring-down)

dm_f_array=np.abs(np.fft.rfft(dm_t_array))
data3_freqs=np.fft.fftfreq(data3md_ns,d=data3md_dt) # as many frequencies as number of time samples, from a time array with the DOWNSAMPLED spacing

plt.figure(figsize=(10,5))
plt.imshow(dm_f_array)
plt.colorbar()
plt.xlabel('frequency (Hz), referenced to 1/time after time='+str(data3_t0iso))
plt.ylabel('DM (pc cm^{-3})')
plt.title('DM-frequency grid for '+str(srubensid))
plt.show() # expect to see only the zero-freq component b/c we only see one instance of an alien signal...

for i,start_time in enumerate(start_times_to_test):
    for j,period in enumerate(periods_to_test):
        for k,width in enumerate(widths_to_test):
            pass


# ###
# # 4. search for periodic signal in Fourier space
# dm_f=np.abs(np.fft.rfft(dm_t.data)) # we don't care about the imag part b/c the rfft of a real-valued array is purely real (no risk of losing info here)
# downsampled_nsamp=p_masked_downsampled.header.nsamples # number of samples in the downsampled data
# downsampled_tsamp=p_masked_downsampled.header.tsamp # length of a sample in the downsampled data
# pfreqs=np.fft.rfftfreq(downsampled_nsamp,d=downsampled_tsamp) # as many frequencies as number of time samples, from a time array with spacing pdt_downsampled

# chan_cutoff=60
# dm_f_slice=np.abs(dm_f)[:,:chan_cutoff]
# fundamental_loc=np.unravel_index(dm_f_slice.argmax(), dm_f_slice.shape) # SNR-maximizing indices
# direct_fundamental_freq=pfreqs[fundamental_loc[1]]
# direct_fundamental_period=1./direct_fundamental_freq

# bestloc=np.unravel_index((np.abs(dm_f)).argmax(), dm_f.shape)
# print('maximal SNR is',dm_f[bestloc])
# best_freq=pfreqs[bestloc[1]]/2. # argmax yields the first harmonic ... divide by two to get the fundamental
# best_period=1./best_freq
# print('SNR-maximizing period is',best_period)
# best_dm=dm_t.dms[bestloc[0]]
# print('SNR-maximizing DM is',best_dm)
# ########################################################

# # TRY THE WEIGHTED AVERAGE TO IDENTIFY THE BEST FREQ AND DM
# abs_dm_f=np.abs(dm_f)
# columnwise_weights=np.sum(abs_dm_f,axis=1)
# possible_dms=np.arange(0,2*ctrdm,deltaDM)[:-1]
# weighted_best_dm=np.average(possible_dms,weights=columnwise_weights)
# print('WEIGHTED best_dm=',weighted_best_dm)

# plt.figure(figsize=(10,5))
# plt.imshow(dm_f,extent=[pfreqs[0],pfreqs[-1],2*ctrdm,0],aspect=0.5) #,norm='log',vmax=np.percentile(dm_f,99))
# cbar=plt.colorbar()
# cbar.set_label('S/N')
# plt.xlabel('frequency (Hz), calculated from time (s) after '+pt0iso)
# plt.ylabel('DM')
# plt.title('DM-frequency grid')
# plt.savefig('dm_f_grid.png',dpi=hires)
# plt.show()

# # 5. fold data to find pulse
# n_folded_bins=int(best_period//pdt_downsampled)
# p_folded_3d=p_masked.fold(best_period,weighted_best_dm,nints=1,nbands=1024,nbins=n_folded_bins) # weighted averaging for dm but still the ad hoc /2 in freq to get the fundamental

# p_folded_2d=p_folded_3d.data[0] # store the one integration we specified in 2d format
# p_folded_1d=np.nanmean(p_folded_2d,axis=0) # sum over frequency channels
# downsampled_folded_times=np.linspace(0,best_period,n_folded_bins)

# plt.figure(figsize=(10,5))
# plt.imshow(p_folded_2d,aspect=5e-4,extent=[downsampled_folded_times[0],downsampled_folded_times[-1],bskfreqs[-1],bskfreqs[0]]) # L,R,B,T
# cbar=plt.colorbar()
# cbar.set_label('S/N')
# plt.xlabel('time (s)')
# plt.ylabel('freq (Hz)')
# plt.title('Folded pulse profile waterfall')
# plt.savefig('folded_2d.png',dpi=hires)
# plt.show()

# plt.figure()
# plt.plot(downsampled_folded_times,p_folded_1d)
# plt.xlabel('time (s)')
# plt.ylabel('S/N')
# plt.title('Frequency-averaged, folded pulse profile')
# plt.savefig('folded_1d.png',dpi=hires)
# plt.show()
# ###