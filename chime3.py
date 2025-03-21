import numpy as np
from matplotlib import pyplot as plt
from sigpyproc.readers import FilReader
from astropy.time import Time

def pulse_template(width,times,centre=0):
    ''' Gaussian template for pulse search (arbitrary amplitude)
    centre = mu
    width  = sigma 
    times  = array of times at which to evaluate the Gaussian template (should be the times for which you have data)
    '''
    return np.exp(-(times-centre)**2/(2.*width))

##### SETUP
tfac=16 # use slightly less downsampling than in the example ... sacrifice a bit of evaluation speed for more finely sampled results
transfer=np.load('transfer.npy') # re-create the transfer function in one step (I modified my part 2 script to add the single line np.save('transfer.npy',transfer) and then load this file here to access the transfer function without repeating so much redundant code)

# BLANK SKY
blank=FilReader('blank_sky.fil')
_, blankmask=blank.clean_rfi(method='mad',threshold=3.)
blank=FilReader('blank_sky_masked.fil')
blank.downsample(tfactor=tfac) # downsample by a factor of tfactor
blank=FilReader('blank_sky_masked_f1_t'+str(tfac)+'.fil')
blankhead=blank.header
blank_dt=blankhead.tsamp
blank_ns=blankhead.nsamples
blank_times0=np.arange(0,blank_ns)*blank_dt
blank_maskedblock=blank.read_block(0,blank_ns) # data is accessible by reading a block
blank_filedata=blank_maskedblock.data
blank_masked_downsampled_normalized=blank_maskedblock.normalise()

# PERSONALIZED DATA
id_of_interest=261215947 # me
# id_of_interest=261209827 # Kim
# id_of_interest=261215346 # Zach
# id_of_interest=0 # placeholder (want to leave my code the same-ish when inspecting the injection)
data3=FilReader('data_'+str(id_of_interest)+'.fil')
# data3=FilReader('data_w_injected_sig.fil')
_, data3mask=data3.clean_rfi(method='mad',threshold=3.)
data3_masked=FilReader('data_'+str(id_of_interest)+'_masked.fil') # use the RFI-flagged version of the data
# data3_masked=FilReader('data_w_injected_sig_masked.fil')
data3_non_downsampled_dt=data3_masked.header.tsamp

data3_masked.downsample(tfactor=tfac) # downsample by a factor of tfactor
data3_masked_downsampled=FilReader('data_'+str(id_of_interest)+'_masked_f1_t'+str(tfac)+'.fil')
# data3_masked_downsampled=FilReader('data_w_injected_sig_masked_f1_t'+str(tfac)+'.fil')
data3head=data3_masked_downsampled.header
data3_t0=data3head.tstart
data3_dt=data3head.tsamp
data3_ns=data3head.nsamples 
data3_phases=np.arange(0,data3_ns) 
data3_times0=data3_phases*data3_dt
data3_times=data3_t0+data3_times0 
data3_nchan=data3head.nchans # number of frequency channels
data3_fch1=data3head.fch1 # frequency of initial channel
data3_df=data3head.foff # frequency offset of channels
data3_nsamp=data3head.nsamples # number of samples per channel = number of time steps of observation
data3_channels=np.arange(0,data3_nchan) # index channel numbers
data3_freqs=data3_fch1+data3_channels*data3_df
data3_downsampled_block=data3_masked_downsampled.read_block(0,data3_ns)
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

##### SETUP FOR LOOPS OVER DM AND WIDTH (correlation takes care of start time automatically)
fterm=1/(400.**2)-1/(800.**2) # GHz; tau = kDM*DM*fterm so deltatau = kDM*deltaDM*fterm -> deltaDM = deltatau/(kDM*fterm)
kDM=4148.8 # MHz**2 pc**{-1} cm**3 s
deltaDM=data3_dt/(kDM*fterm) # pc cm**{-3} **desired spacing for the dm search
dm_lo=-50
# dm_lo=-100
dm_hi=0
DMrange=dm_hi-dm_lo 
n_dms=int(DMrange/deltaDM)
dm_candidates=np.linspace(dm_lo,dm_hi,n_dms) # use DM magnitudes up to the edge of the galaxy, but negative

max_width=0.1 # cap at 1 s for now
coarse_width_search=False
if coarse_width_search: # hard-coded smaller parameter set to speed the search
    n_widths= 25
    width_candidates=np.linspace(data3_non_downsampled_dt,max_width,n_widths) 
else: # more compute time but more accurate to the limits of the survey
    width_candidates=np.arange(data3_non_downsampled_dt,max_width,data3_non_downsampled_dt)
    n_widths=len(width_candidates)
    print('n_widths=',n_widths)

dm_width_array=np.zeros((n_dms,n_widths)) # max value for best-fit DM and width
start_index_array=np.zeros((n_dms,n_widths)) # stored values = indices of start time where the correlation is strongest
# snr_array=np.zeros((n_dms,n_widths))

##### SEARCH THE DM-WIDTH PARAMETER SPACE
n_hist_bins=75
for i,test_dm in enumerate(dm_candidates): # consider the DM candidates
    dedispersed_data=data3_masked_downsampled_normalized.dedisperse(test_dm) # get 2D array dedispersed by the candidate amount
    dedispersed_pulse_profile=dedispersed_data.get_tim().data
    # dedispersed_blank=blank_masked_downsampled_normalized.dedisperse(test_dm)
    # dedispersed_blank_profile=dedispersed_blank.get_tim().data

    for j, test_width in enumerate(width_candidates): # consider the width candidates
        current_template=pulse_template(test_width,data3_times0) # width,times
        convolved_data3=np.convolve(current_template,dedispersed_pulse_profile)
        # ##
        # convolved_blank=np.convolve(current_template,dedispersed_blank_profile)
        # data3_hist,data3_bin_edges=np.histogram(convolved_data3,bins=n_hist_bins)
        # data3_hist_mean=data3_hist@data3_bin_edges[:-1]/np.sum(data3_hist)
        # blank_hist,blank_bin_edges=np.histogram(convolved_blank,bins=n_hist_bins)
        # blank_hist_mean=blank_hist@blank_bin_edges[:-1]/np.sum(blank_hist)
        # ##

        maxidx=np.argmax(convolved_data3)
        start_index_array[i,j]=maxidx
        dm_width_array[i,j]=convolved_data3[maxidx]
        # snr_array[i,j]=data3_hist_mean/blank_hist_mean

##### VISUALIZE LOOP DATA PRODUCTS
loop_aspect=2e-3
fig,axs=plt.subplots(1,2,figsize=(20,5))
im=axs[0].imshow(start_index_array,aspect=loop_aspect,extent=[width_candidates[0],width_candidates[-1],dm_candidates[-1],dm_candidates[0]]) # # extent=[left,right,bottom,top]
plt.colorbar(im,ax=axs[0])
axs[0].set_xlabel('width')
axs[0].set_ylabel('DM')
axs[0].set_title('start index')
im=axs[1].imshow(dm_width_array,aspect=loop_aspect,extent=[width_candidates[0],width_candidates[-1],dm_candidates[-1],dm_candidates[0]])
plt.colorbar(im,ax=axs[1])
axs[1].set_xlabel('width')
axs[1].set_ylabel('DM')
axs[1].set_title('signal')
# im=axs[2].imshow(,aspect=loop_aspect,extent=[width_candidates[0],width_candidates[-1],dm_candidates[-1],dm_candidates[0]])
# plt.colorbar(im,ax=axs[2])
# axs[3].set_xlabel('width')
# axs[3].set_ylabel('DM')
# axs[3].set_title('noise')
# im=axs[3].imshow(snr_array,aspect=loop_aspect,extent=[width_candidates[0],width_candidates[-1],dm_candidates[-1],dm_candidates[0]])
# plt.colorbar(im,ax=axs[3])
# axs[3].set_xlabel('width')
# axs[3].set_ylabel('DM')
# axs[3].set_title('SNR')
plt.tight_layout()
plt.show()

##### RESULTING PULSE
alien_dm_idx,alien_width_idx=np.unravel_index((np.abs(dm_width_array)).argmax(), dm_width_array.shape)
alien_dm=dm_candidates[alien_dm_idx]
alien_width=width_candidates[alien_width_idx]
alien_start_time_idx=int(start_index_array[alien_dm_idx,alien_width_idx]) # FIGURE OUT OFFSET
alien_start_time=data3_times0[alien_start_time_idx]
print('alien signal properties:')
print('start time=',alien_start_time)
print('DM=',alien_dm)
print('width=',alien_width)

final_dedispersed_data=data3_masked_downsampled_normalized.dedisperse(alien_dm)
alien_pulse_profile=final_dedispersed_data.get_tim().data
plt.figure()
plt.plot(data3_times0,alien_pulse_profile,label='final dedispersed pulse profile')
plt.plot(data3_times0,np.max(alien_pulse_profile)*np.exp(-(data3_times0-alien_start_time)**2/(2*alien_width**2)),label='alien template')
plt.xlabel('time')
plt.ylabel('intensity')
plt.legend()
plt.title('reality check for pulse profile')
plt.show()

##### HISTOGRRAM ANALYSIS
dedispersed_blank_sky=blank_masked_downsampled_normalized.dedisperse(alien_dm)
blank_pulse_profile=dedispersed_blank_sky.get_tim().data

alien_template=pulse_template(alien_width,data3_times0) # width,times

convolved_data3=np.convolve(alien_template,alien_pulse_profile)
convolved_blank=np.convolve(alien_template,blank_pulse_profile)

data3_hist,data3_bin_edges=np.histogram(convolved_data3,bins=n_hist_bins)
data3_hist_mean=data3_hist@data3_bin_edges[:-1]/np.sum(data3_hist)
blank_hist,blank_bin_edges=np.histogram(convolved_blank,bins=n_hist_bins)
blank_hist_mean=blank_hist@blank_bin_edges[:-1]/np.sum(blank_hist)

fig,axs=plt.subplots(2,1,figsize=(5,10),sharex=True)
axs[0].stairs(data3_hist,data3_bin_edges,label='histogram',fill=True)
axs[0].axvline(data3_hist_mean,linestyle='--',c='C1',label='mean')
axs[0].set_yscale('log')
axs[0].set_xlabel('significance')
axs[0].set_ylabel('number of instances')
axs[0].set_title('data for ID'+str(id_of_interest))
axs[0].legend()
axs[1].stairs(blank_hist,blank_bin_edges,label='histogram',fill=True)  
axs[1].axvline(blank_hist_mean,linestyle='--',c='C1',label='mean')  
axs[1].set_yscale('log')        
axs[1].set_xlabel('significance')
axs[1].set_ylabel('number of instances')
axs[1].set_title('blank sky')
axs[1].legend()
plt.suptitle('DM = {:-5.3}; start time (s after t0) = {:4.3}; width={:8.3}'.format(alien_dm,alien_start_time,alien_width))
plt.tight_layout()
plt.savefig('histogram_dm_'+str(test_dm)+'_t_'+str(alien_start_time)+'_w_'+str(test_width)+'.png')
plt.show()

##### BE MORE FORMAL ABOUT THE CONFIDENCE INTERVALS