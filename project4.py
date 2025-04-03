import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
# import astropy.units as u
from matplotlib import pyplot as plt

######################################## prep 
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

######################################## data exploration
#################### q1
gammafile='gammaray.fits'
with fits.open(gammafile) as hdulist:
    print(hdulist.info())
    ra_rad=hdulist[1].data['RA(rad)']
    dec_rad=hdulist[1].data['Dec(rad)']
probe_fits(gammafile)
ra_deg=ra_rad*180./np.pi
dec_deg=dec_rad*180./np.pi

h2dT,xedges,yedges=np.histogram2d(ra_deg,dec_deg,bins=500)
h2d=h2dT.T

plt.figure()
plt.hist2d(ra_deg,dec_deg,bins=500) # Stackoverflow: hist2d bin populations need to be transposed before plotting (https://stackoverflow.com/questions/59795238/how-to-use-or-manipulate-the-output-return-values-of-hist2d-and-create-a-new-h)
plt.colorbar()
plt.xlabel('RA (deg)')
plt.ylabel('dec (deg)')
plt.title('2D histogram of the provided VERITAS gamma-ray data')
plt.axis('equal')
plt.savefig('hist2d_veritas.png')
plt.show()

#################### q2, q3: no separate code

#################### q4
ra_pointing_deg=83.6329
dec_pointing_deg=22.5258
pointing_ctr_deg=SkyCoord(ra_pointing_deg, dec_pointing_deg, unit="deg")
print('pointing ctr coords=',pointing_ctr_deg)

# ra_obj_deg_idx,dec_obj_deg_idx=np.unravel_index(h2dT.argmax(), h2dT.shape)
# ra_obj_ctr_deg=xedges[ra_obj_deg_idx]
# dec_obj_ctr_deg=yedges[dec_obj_deg_idx]
ra_obj_ctr_hms='05h34m31.8s'
dec_obj_ctr_hms='22h01m03s'
ra_obj_ctr_deg=5+34./60.+31.8/3600.
dec_obj_ctr_deg=22+1./60.+3./3600.

obj_ctr_deg=SkyCoord(ra_obj_ctr_deg, dec_obj_ctr_deg, unit='deg')
print('object ctr coords=',obj_ctr_deg)

plt.figure(figsize=(30,5))
plt.hist2d(ra_deg,dec_deg,bins=500) # Stackoverflow: hist2d bin populations need to be transposed before plotting (https://stackoverflow.com/questions/59795238/how-to-use-or-manipulate-the-output-return-values-of-hist2d-and-create-a-new-h)
plt.colorbar()
plt.scatter(ra_pointing_deg,dec_pointing_deg,label='pointing centre',c='C1')
plt.scatter(ra_obj_ctr_deg,dec_obj_ctr_deg,label='object centre',c='C3')
plt.xlabel('RA (deg)')
plt.ylabel('dec (deg)')
plt.xlim(5,95)
plt.ylim(20,30)
plt.axis('equal')
plt.title('2D histogram of the provided VERITAS gamma-ray data')
plt.legend()
plt.savefig('hist2d_veritas_with_centres.png')
plt.show()

offset_ra=ra_pointing_deg-ra_obj_ctr_deg
offset_dec=dec_pointing_deg-dec_obj_ctr_deg
print('offset_ra,offset_dec=',offset_ra,offset_dec)

######################################## statistical detection
#################### q1
def event_toSkyCoord(event_idx_ra,event_idx_dec):
    return SkyCoord(event_idx_ra, event_idx_dec, unit="deg")

#################### q2
#################### q3
#################### q4
#################### q5
#################### q6

######################################## cut optimization
#################### q1
#################### q2
#################### q3
#################### q4