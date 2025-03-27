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
q2_prep_investig_path='/Users/sophiarubens/Downloads/W25/PHYS641/projects/project3_NICER/sax1808_burst_data/2584010501/xti/event_cl/'

prep_investig_verbose=False
if prep_investig_verbose:
    probe_fits(path_head+installation_q2_file)

with fits.open(q2_prep_investig_path+q2_prep_investig_fname) as hdulist:
    print('hdulist[1].data.dtype.names)=',hdulist[1].data.dtype.names) # I was struggling to figure out what the column keys were, so I (reluctantly) asked ChatGPT, which, shockingly, had this useful suggestion. Here's what I asked in order for it to recommend this: "I have a FITS file that I know contains some energies in a table that lives in hdulist[1].data. However, I don't know what that key is called (to the point where I get a KeyError if I try to access the table column using what I think the key might be). How do I print out a list of all the keys in this hdulist[1].data? (I tried print(hdulist[1].data.keys()) and it didn't work, giving me an error message that said AttributeError: recarray has no attribute keys)"
    q2_prep_investig_energies =hdulist[1].data['PI']
    q2_prep_investig_times    =hdulist[1].data['TIME']

fig,[ax0,ax1]=plt.subplots(1,2,figsize=(15,5))
ax0.hist(q2_prep_investig_energies/1000.,log=True)
ax0.set_xlabel('PI energy (keV)')
ax0.set_ylabel('bin population')
ax0.set_title('Energy distribution')
ax1.hist(q2_prep_investig_times,log=True)
ax1.set_xlabel('time (s)')
ax1.set_ylabel('bin population')
ax1.set_title('Time distribution')
plt.suptitle("Histograms for NICER's observation of the bright x-ray burst from SAX J1808 on 2019 August 21 at 02:04 UTC")
plt.tight_layout()
plt.savefig('q2_prep_investig_histograms')
plt.show()

with fits.open('/Users/sophiarubens/Downloads/W25/PHYS641/projects/project3_NICER/sax1808_burst_data/ni2584010501_barycorred.evt') as hdulist:
    q3_prep_investig_post_barycorr_time=hdulist[1].data['TIME']
    q3_prep_investig_post_barycorr_pi=hdulist[1].data['PI']

plt.figure(figsize=(20,5))
plt.semilogy(q2_prep_investig_times,q2_prep_investig_energies,label='pre',lw=0.5)
plt.semilogy(q3_prep_investig_post_barycorr_time,q3_prep_investig_post_barycorr_pi,label='post',lw=0.5)
plt.xlabel('time (s)')
plt.ylabel('PI energy (eV)')
plt.title('Visualizing the impact of barycentre correction for the NICER observation of the August 2019 SAX 1808 burst')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('barycorr_comparison.png')
plt.show()

fig,[ax0,ax1]=plt.subplots(1,2,figsize=(15,5))
ax0.hist(q2_prep_investig_energies/1000.,log=True,label='pre')
ax0.hist(q3_prep_investig_post_barycorr_pi/1000.,log=True,label='post')
ax0.set_xlabel('PI energy (keV)')
ax0.set_ylabel('bin population')
ax0.set_title('Energy distribution')
ax0.legend(loc='upper right')
ax1.hist(q2_prep_investig_times,log=True,label='pre')
ax1.hist(q3_prep_investig_post_barycorr_time,log=True,label='post')
ax1.set_xlabel('time (s)')
ax1.set_ylabel('bin population')
ax1.set_title('Time distribution')
ax1.legend(loc='upper right')
plt.suptitle("Comparison of NICER histograms for the August 2019 SAX 1808 burst before and after barycentre correction")
plt.tight_layout()
plt.savefig('barycorr_hist_comparisons')
plt.show()