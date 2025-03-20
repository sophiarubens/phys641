import numpy as np
from matplotlib import pyplot as plt
from sigpyproc.readers import FilReader
from astropy.time import Time
from numpy.fft import fft,ifft

def pulse_template(centre,width,times):
    ''' Gaussian template for pulse search (arbitrary amplitude)
    centre = mu
    width  = sigma 
    times  = array of times at which to evaluate the Gaussian template (should be the times for which you have data)
    '''
    return np.exp(-(times-centre)**2/(2.*width))

##### SETUP
# blank sky setup
# read in the masked blank sky the same way I filtered it in a previous part of the project
bsk,    bskheader, bsktimes, bskt0, bskdt, bskns, bsknchan, bskfch1, bskdf, bsknsamp, bskchannels, bskfreqs, bskblock,  bskdata,  bskspec=importfil('blank_sky.fil')
_, bskmask=bsk.clean_rfi(method='mad',threshold=3.)
bsk,    bskheader, bsktimes, bskt0, bskdt, bskns, bsknchan, bskfch1, bskdf, bsknsamp, bskchannels, bskfreqs, bskblock,  bskdata,  bskspec=importfil('blank_sky_masked.fil')

bskhead=bsk.header
bsk_dt=bskhead.tsamp
bsk_ns=bskhead.nsamples
bsk_times0=np.arange(0,bsk_ns)*bsk_dt
bsk_maskedblock=bsk.read_block(0,bsk_ns) # data is accessible by reading a block
bsk_filedata=bsk_maskedblock.data
bsk_timeseries=bsk_filedata.get_tim().data

# read in some alien candidate data
id_of_interest=261215947 # me
# id_of_interest=261213158 # test w/ a random classmate's data ... formalize this later with a loop over all or something
data3=FilReader('data_'+str(id_of_interest)+'.fil')
_, data3mask=data3.clean_rfi(method='mad',threshold=3.)
data3_masked=FilReader('data_'+str(id_of_interest)+'_masked.fil') # use the RFI-flagged version of the data

##### UNDERSTAND THE SINGLE-PULSE NOISE VIA SNR ANALYSIS -> NUMBER OF FOLDS REQUIRED FOR A CERTAIN SNR 
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
np.save('data3_freqs_'+str(id_of_interest)+'.npy',data3_freqs) # my Python environments are cooked beyond all recognition (despite going to the debug den and talking to a lot of people and reading a lot of forum posts and creating a lot of new environments and trying to reinstall a lot of packages, I don't have a single environment where I can import sigpyproc, scipy, and astropy) --> I did my power law fit in another script
data3_maskedblock=data3_masked.read_block(0,data3_ns) # data is accessible by reading a block
data3_filedata=data3_maskedblock.data

data3_t0obj=Time(data3_t0,format='mjd')
data3_t0iso=data3_t0obj.iso
hires=500

### RE-CREATE THE TRANSFER FUNCTION
transfer=np.load('transfer.npy') # I modified my part 2 script to add the single line np.save('transfer.npy',transfer) and then load this file here to access the transfer function without repeating so much redundant code

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
dm_candidates=np.linspace(dm_lo,dm_hi,n_dms) # use DM magnitudes up to the edge of the galaxy, but negative
# dm_t_array=np.zeros((n_dms,data3md_ns)) # dmt_transform-like array, to populate manually
# print('dm_t_array.shape=',dm_t_array.shape)

start_time_candidates=np.arange(0.,10.,data3md_dt) # possible start times
max_width=1. # cap at 1 s for now ?!
n_widths= 50
width_candidates=np.linspace(data3md_dt,max_width,n_widths) # don't expect to see anything narrower than one time sample

nbins=20

for i,test_dm in enumerate(dm_candidates): # consider the DM candidates
    dedispersed_data=data3_masked_downsampled_normalized.dedisperse(test_dm) # get 2D array dedispersed by the candidate amount
    dedispersed_pulse_profile=dedispersed_data.get_tim().data

    for j,test_start_time in enumerate(start_time_candidates):
        for k, test_width in enumerate(width_candidates):
            current_template=pulse_template(test_start_time,test_width,data3md_times0) # centre,width,times
            convolved_data3=ifft(fft(current_template)*fft(dedispersed_pulse_profile))
            convolved_blank=ifft(fft(current_template)*fft(bsk_timeseries))

            data3_hist,data3_bin_edges=np.histogram(convolved_data3,bins=nbins)
            blank_hist,blank_bin_edges=np.histogram(convolved_blank,bins=nbins)

            fig,axs=plt.subplots(1,2)
            axs[0].plot(data3_bin_edges[:-1],data3_hist)
            axs[0].set_xlabel('significance')
            axs[0].set_ylabel('number of instances')
            axs[0].set_title('data for ID'+str(id_of_interest))
            axs[1].plot(blank_bin_edges[:-1],blank_hist)            
            axs[1].set_xlabel('significance')
            axs[1].set_ylabel('number of instances')
            axs[1].set_title('blank sky')
            plt.suptitle('DM = {:-8.3}; start time (s after t0) = {:9.3}; width={:9.3}'.format(test_dm,test_start_time,test_width))
            plt.tight_layout()
            plt.savefig('histogram_check.png')
            plt.show()

            assert(1==0), "check histogram procedure so far"

    # dm_t_array[i,:] = dedisp_array.get_tim().data # sum over frequency channels to get the dedispersed data for that time and DM

# bestloc=np.unravel_index((np.abs(dm_t_array)).argmax(), dm_t_array.shape)
# best_start_time=data3md_times0[bestloc[1]]
# print('SNR-maximizing start time is',best_start_time)
# best_dm=dm_vec[bestloc[0]]

# print('SNR-maximizing DM is',best_dm)

# plt.figure(figsize=(10,5))
# plt.imshow(dm_t_array,aspect=1e-1,extent=[data3md_times0[0],data3md_times0[-1],dm_vec[-1],dm_vec[0]]) # extent=[L,R,B,T]
# plt.colorbar()
# plt.scatter(best_start_time,best_dm,s=1,c='r')
# plt.xlabel('time (s) after '+str(data3_t0iso))
# plt.ylabel('DM (pc cm^{-3})')
# plt.title('DM-time grid for '+str(id_of_interest))
# plt.savefig('dm_t_array_'+str(id_of_interest)+'.png',dpi=hires)
# plt.show() # since I only see one maximum, it's not worth searching in Fourier space ... doing a Fourier analysis would merely reveal that the zero-frequency component dominates (possibly with some ring-down)

# dedispersed_2d=data3_masked_downsampled_normalized.dedisperse(best_dm)

# plt.figure(figsize=(10,5))
# plt.imshow(dedispersed_2d.data,extent=[data3_times0[0],data3_times0[-1],data3_freqs[-1],data3_freqs[0]],aspect=1e-2)
# plt.xlabel('time after '+str(data3_t0iso))
# plt.ylabel('freq (MHz)')
# plt.title('dedispersion check')
# plt.savefig('dedispersion_check.png')
# plt.show()

# dedispersed_pulse_profile=dedispersed_2d.get_tim().data

# plt.figure()
# plt.plot(data3md_times0,dedispersed_pulse_profile)
# plt.xlabel('freq [MHz]')
# plt.ylabel('intensity [ADU]')
# plt.title('dedispersed pulse profile')
# plt.show()

##### PULSE WIDTH SEARCH
# width_lo=data3md_dt # can't expect to identify a pulse narrower than the minimum time spacing of samples in the dataset
# widths_hi=[1.,0.05]
# width_case_titles=['entire plausible range','inset'] # by inspection of even just the pre-dedispersion waterfall plot, I'd be shocked if the pulse occupied more than 10% of the time samples in the ten-second dataset ... but upon looking more closely, even this is far too conservative a guess, hence the inset
# n_widths_to_test=100

# correlations_varying_width=np.zeros(n_widths_to_test)
# fig,axs=plt.subplots(1,2,figsize=(8,6))
# for i,width_hi in enumerate(widths_hi):
#     widths_to_test=np.linspace(width_lo,width_hi,n_widths_to_test)
#     for j,test_width in enumerate(widths_to_test):
#         template_current_width=pulse_template(test_start_time,test_width,data3md_times0)
#         correlations_varying_width[j]=np.linalg.norm(np.correlate(dedispersed_pulse_profile,template_current_width))

#     axs[i].plot(widths_to_test,correlations_varying_width)
#     axs[i].set_xlabel('pulse width (s)')
#     axs[i].set_ylabel('timeseries norm of correlation amplitude [dimensionless]')
#     axs[i].set_title(width_case_titles[i])
# best_pulse_width=widths_to_test[np.argmax(correlations_varying_width)]
# axs[1].axvline(best_pulse_width,c='r')
# plt.suptitle('pulse profile template-data correlation')
# plt.tight_layout()
# plt.savefig('pulse_width_identification.png')
# plt.show()