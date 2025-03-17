import numpy as np
from matplotlib import pyplot as plt
from sigpyproc.readers import FilReader
from astropy.time import Time

id_of_interest=261215947 # me
# id_of_interest=260374104 # test w/ a random classmate's data ... formalize this later with a loop over all or something
data3=FilReader('data_'+str(id_of_interest)+'.fil')
_, data3mask=data3.clean_rfi(method='mad',threshold=3.)
data3_masked=FilReader('data_'+str(id_of_interest)+'_masked.fil') # use the RFI-flagged version of the data

##### UNDERSTAND THE SINGLE-PULSE NOISE VIA SNR ANALYSIS -> NUMBER OF FOLDS REQUIRED FOR A CERTAIN SNR 
# spectral_index=-1.4 # mean finding from [doi:10.1093/mnras/stt257]
# pulsar indices are never steeper than about -4 [https://doi.org/10.1093/mnras/stx2476], and physical intuition says they should all have nonpositive spectral indices
# unclear which quantities to use in the SNR RHS,, can't hurt to start with 
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

data3_t0obj=Time(data3_t0,format='mjd')
data3_t0iso=data3_t0obj.iso
hires=500

### RE-CREATE THE TRANSFER FUNCTION
###

# plt.figure(figsize=(10,5))
# plt.plot(data3_times0,data3_filedata)
# plt.xlabel('time (s) after '+str(data3_t0iso))
# plt.ylabel('flux')
# plt.title()
# plt.show()

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

##### DM AND BOUND ON PERIOD FROM A DM-TIME GRID ANALYSIS
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

dedispersed_2d=data3_masked_downsampled_normalized.dedisperse(best_dm)
dedispersed_pulse_profile=dedispersed_2d.get_tim().data
# plt.figure()
# plt.plot(dedispersed_pulse_profile)
# plt.show()
# assert(1==0)

##### PULSE WIDTH SEARCH
# IMPLEMENT THE LOOPS HERE
# AMOUNT OF TIME THE SIGNAL IS ABOVE A CERTAIN SNR?
# FIT A GAUSSIAN AND USE THE FWHM OR SIGMA OR SOMETHING MORE STATISTICAL LIKE THAT??

width_lo=data3md_dt # can't expect to identify a pulse narrower than the minimum time spacing of samples in the dataset
widths_hi=[1.,0.05,0.015]
# width_hi=1. # by inspection, I'd be shocked if the pulse occupied more than 10% of the time samples in the ten-second dataset
n_widths_to_test=100

def pulse_template(centre,width,times):
    ''' Gaussian template for pulse search (arbitrary amplitude)
    centre = mu
    width  = sigma 
    times  = array of times at which to evaluate the Gaussian template (should be the times for which you have data)
    '''
    return np.exp(-(times-centre)**2/(2.*width))

correlations_varying_width=np.zeros((data3md_ns,n_widths_to_test))
fig,axs=plt.subplots(1,3,figsize=(12,4))
for i,width_hi in enumerate(widths_hi):
    widths_to_test=np.linspace(width_lo,width_hi,n_widths_to_test)
    for j,test_width in enumerate(widths_to_test):
        template_current_width=pulse_template(best_start_time,test_width,data3md_times0)
        correlations_varying_width[:,j]=np.correlate(dedispersed_pulse_profile,template_current_width)

    # plt.figure()
    axs[i].imshow(correlations_varying_width)
    axs[i].set_xlabel('pulse width (s)')
    axs[i].set_ylabel('correlation [dimensionless]')
    axs[i].set_title('pulse with inset '+str(i))
plt.suptitle('pulse profile template-data correlation for several pulse width ranges')
plt.tight_layout()
plt.savefig('pulse width identification')
plt.show()