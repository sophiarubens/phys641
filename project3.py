import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits

def probe_fits(fname):
    with fits.open(fname) as hdulist:
        for i,hdu in enumerate(hdulist):
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>new hdu with index',i)
            test_header= hdu.header
            if test_header==None:
                print('no header in hdu',i)
            else:
                print('current hdu.header:')
                print(test_header)
            test_data  = hdu.data
            print('current hdu.data=',test_data)
    return None

path_head='/Users/sophiarubens/'
installation_q2_file='caldb/data/nicer/xti/cpf/arf/nixtiaveonaxis20170601v001.arf'

installation_verbose=False
if installation_verbose:
    probe_fits(path_head+installation_q2_file)
# conclusion from running this loop:
# non-empty parts of the file in question:
# -- hdulist[0].header seems to have general info about the file
# -- hdulist[0].data is empty
# -- hdulist[1].header seems to have info specific to the data in hdulist[1].data
# -- hdulist[1].data seems to have the data question 2 of the installation section asks for (Nicole told me about accessing table fields using column headers)
with fits.open(path_head+installation_q2_file) as hdulist:
    q2_installation_energies =hdulist[1].data['ENERGY']
    q2_installation_specresp =hdulist[1].data['SPECRESP']

plt.figure()
plt.semilogx(q2_installation_energies,q2_installation_specresp)
plt.xlabel('photon energy  (keV)')
plt.ylabel('spectral response (cm^2)')
plt.title('NICER ancillary response function (ARF)')
plt.savefig('arf_vs_energy.png')
plt.show()

q2_prep_investig_fname='ni2584010501_0mpu7_cl.evt'
event_cl_path='/Users/sophiarubens/Downloads/W25/PHYS641/projects/project3_NICER/sax1808_burst_data/2584010501/xti/event_cl/'

prep_investig_verbose=False
if prep_investig_verbose:
    probe_fits(path_head+installation_q2_file)

with fits.open(event_cl_path+q2_prep_investig_fname) as hdulist:
    print('hdulist[1].data.dtype.names)=',hdulist[1].data.dtype.names) # I was struggling to figure out what the column keys were, so I (reluctantly) asked ChatGPT, which, shockingly, had this useful suggestion. Here's what I asked in order for it to recommend this: "I have a FITS file that I know contains some energies in a table that lives in hdulist[1].data. However, I don't know what that key is called (to the point where I get a KeyError if I try to access the table column using what I think the key might be). How do I print out a list of all the keys in this hdulist[1].data? (I tried print(hdulist[1].data.keys()) and it didn't work, giving me an error message that said AttributeError: recarray has no attribute keys)"
    q2_prep_investig_energies =hdulist[1].data['PHA']
    q2_prep_investig_times    =hdulist[1].data['TIME']

nbins=300
fig,[ax0,ax1]=plt.subplots(1,2,figsize=(15,5))
ax0.hist(q2_prep_investig_energies/1000.,log=True,bins=nbins)
ax0.set_xlabel('PHA energy (keV)')
ax0.set_ylabel('bin population')
ax0.set_title('Energy distribution')
ax1.hist(q2_prep_investig_times,log=True,bins=nbins)
ax1.set_xlabel('time (s)')
ax1.set_ylabel('bin population')
ax1.set_title('Time distribution')
plt.suptitle("Histograms for NICER's observation of the bright x-ray burst from SAX J1808 on 2019 August 21 at 02:04 UTC")
plt.tight_layout()
plt.savefig('q2_prep_investig_histograms')
plt.show()

burst_data_start='/Users/sophiarubens/Downloads/W25/PHYS641/projects/project3_NICER/sax1808_burst_data/'
barycorred=burst_data_start+'ni2584010501_barycorred.evt'
with fits.open(barycorred) as hdulist:
    q3_prep_investig_post_barycorr_time=hdulist[1].data['TIME']
    q3_prep_investig_post_barycorr_pi=hdulist[1].data['PHA']

plt.figure(figsize=(20,5))
plt.semilogy(q2_prep_investig_times,q2_prep_investig_energies,label='pre',lw=0.5)
plt.semilogy(q3_prep_investig_post_barycorr_time,q3_prep_investig_post_barycorr_pi,label='post',lw=0.5)
plt.xlabel('time (s)')
plt.ylabel('PHA (uncalibrated) energy (eV)')
plt.title('Visualizing the impact of barycentre correction for the NICER observation of the August 2019 SAX J1808.4–3658 burst')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('barycorr_comparison.png')
plt.show()

fig,[ax0,ax1]=plt.subplots(1,2,figsize=(15,5))
ax0.hist(q2_prep_investig_energies/1000.,log=True,label='pre',bins=nbins)
ax0.hist(q3_prep_investig_post_barycorr_pi/1000.,log=True,label='post',bins=nbins)
ax0.set_xlabel('PHA (uncalibrated) energy (keV)')
ax0.set_ylabel('bin population')
ax0.set_title('Energy distribution')
ax0.legend(loc='upper right')
ax1.hist(q2_prep_investig_times,log=True,label='pre',bins=nbins)
ax1.hist(q3_prep_investig_post_barycorr_time,log=True,label='post',bins=nbins)
ax1.set_xlabel('time (s)')
ax1.set_ylabel('bin population')
ax1.set_title('Time distribution')
ax1.legend(loc='upper right')
plt.suptitle("Comparison of NICER histograms for the August 2019 SAX J1808.4–3658 burst before and after barycentre correction")
plt.tight_layout()
plt.savefig('barycorr_hist_comparisons')
plt.show()

lc_start='ni2584010501mpu7_sr'
lc_list=['low','mid','high']
lc_verbose=False
if lc_verbose:
    probe_fits(event_cl_path+lc_start+'low.lc')

with fits.open(event_cl_path+lc_start+'low.lc') as hdulist:
    lc_low_time=hdulist[1].data['TIME']
    lc_low_rate=hdulist[1].data['RATE']
with fits.open(event_cl_path+lc_start+'mid.lc') as hdulist:
    lc_mid_time=hdulist[1].data['TIME']
    lc_mid_rate=hdulist[1].data['RATE']
with fits.open(event_cl_path+lc_start+'high.lc') as hdulist:
    lc_high_time=hdulist[1].data['TIME']
    lc_high_rate=hdulist[1].data['RATE']
    # print('hdulist[1].data.dtype.names)=',hdulist[1].data.dtype.names)

fig,[axlow,axmid,axhigh]=plt.subplots(1,3,figsize=(20,5))
axlow.plot(lc_low_time,lc_low_rate)
axlow.set_xlabel('time since t0 (s)')
axlow.set_ylabel('counts')
axlow.set_title('0.3-1 keV')
axmid.plot(lc_mid_time,lc_mid_rate)
axmid.set_xlabel('time since t0 (s)')
axmid.set_ylabel('counts')
axmid.set_title('0.3-10 keV')
axhigh.plot(lc_high_time,lc_high_rate)
axhigh.set_xlabel('time since t0 (s)')
axhigh.set_ylabel('counts')
axhigh.set_title('3-10 keV')
plt.savefig('energy_range_light_curves.png')
plt.show()

# put the specific 100 seconds here
t0=5562
fig,[axlow,axmid,axhigh]=plt.subplots(1,3,figsize=(20,5))
axlow.plot(lc_low_time,lc_low_rate)
axlow.set_xlabel('time since t0 (s)')
axlow.set_xlim(t0,t0+60)
axlow.set_ylabel('counts')
axlow.set_title('0.3-1 keV')
axmid.plot(lc_mid_time,lc_mid_rate)
axmid.set_xlabel('time since t0 (s)')
axmid.set_xlim(t0,t0+60)
axmid.set_ylabel('counts')
axmid.set_title('0.3-10 keV')
axhigh.plot(lc_high_time,lc_high_rate)
axhigh.set_xlabel('time since t0 (s)')
axhigh.set_xlim(t0,t0+60)
axhigh.set_ylabel('counts')
axhigh.set_title('3-10 keV')
plt.savefig('energy_range_light_curves_inset_specific.png')
plt.show()

fig,[lcfull,hardness]=plt.subplots(1,2,figsize=(15,5))
lcfull.plot(lc_mid_time,lc_mid_rate)
lcfull.set_xlabel('time since t0 (s)')
lcfull.set_xlim(t0,t0+60)
lcfull.set_ylabel('counts')
lcfull.set_title('0.3-10 keV light curve')
hardness.plot(lc_high_time,lc_high_rate/lc_low_rate)
hardness.set_xlabel('time since t0 (s)')
hardness.set_xlim(t0,t0+60)
hardness.set_ylabel('hardness ratio (dimensionless)')
hardness.set_title('hardness ratio: 3-10 keV / 0.3-1 keV')
plt.suptitle('Flaring event inset')
plt.savefig('flaring_event_inset.png')
plt.show()

##########
print('prelim investigation for GTI file creation')
fields_of_interest=['MJDREFI','MJDREFF','TIMEZERO','TIMESYS','TIMEREF','TSTART','TSTOP']
with fits.open(barycorred) as hdulist:
    for field in fields_of_interest:
        print(field,'=',hdulist[3].header[field])
    np.savetxt('gti.txt',hdulist[3].data)