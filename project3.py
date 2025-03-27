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

installation_verbose=True
if installation_verbose:
    probe_fits(path_head+installation_q2_file)

# conclusion from running this loop:
# non-empty parts of the file in question:
# -- hdulist[0].header seems to have general info about the file
# -- hdulist[0].data is empty
# -- hdulist[1].header seems to have info specific to the data in hdulist[1].data
# -- hdulist[1].data seems to have the data question 2 of the installation section asks for (Nicole told me about accessing table fields using column headers)

with fits.open(path_head+installation_q2_file) as hdulist:
    # q2_installation_header=hdulist[1].header
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
q2_prep_investig_path='/Users/sophiarubens/Downloads/W25/PHYS641/projects/project3_NICER/sax1808_burst_data/2584010501/xti/event_cl'

prep_investig_verbose=True
if prep_investig_verbose:
    probe_fits(path_head+installation_q2_file)

# q2_prep_investig_energies=
# q2_prep_investig_times=

# fig,[ax0,ax1]=plt.subplots(1,2,figsize=(15,5))
# ax0.hist(q2_prep_investig_energies)
# ax0.set_xlabel('')
# ax0.set_ylabel('')
# ax0.set_title('')
# ax1.hist(q2_prep_investig_times)
# ax1.set_xlabel('')
# ax1.set_ylabel('')
# ax1.set_title('')
# plt.suptitle('')
# plt.tight_layout()
# plt.savefig('q2_prep_investig_histograms')
# plt.show()