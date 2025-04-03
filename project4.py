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
# probe_fits(gammafile)
ra_deg=ra_rad*180./np.pi
dec_deg=dec_rad*180./np.pi

h2dT,xedges,yedges=np.histogram2d(ra_deg,dec_deg,bins=500)
h2d=h2dT.T

plt.figure()
plt.imshow(h2d,extent=[xedges[0],xedges[-1],yedges[-1],yedges[0]],aspect=1) # Stackoverflow: hist2d bin populations need to be transposed before plotting (https://stackoverflow.com/questions/59795238/how-to-use-or-manipulate-the-output-return-values-of-hist2d-and-create-a-new-h)
plt.colorbar()
plt.xlabel('RA (deg)')
plt.ylabel('dec (deg)')
plt.title('2D histogram of the provided VERITAS gamma-ray data')
plt.savefig('hist2d_veritas.png')
plt.show()

# plt.figure()
# plt.hist2d(ra_deg,dec_deg,bins=500) # Stackoverflow: hist2d bin populations need to be transposed before plotting (https://stackoverflow.com/questions/59795238/how-to-use-or-manipulate-the-output-return-values-of-hist2d-and-create-a-new-h)
# plt.colorbar()
# plt.xlabel('RA (deg)')
# plt.ylabel('dec (deg)')
# plt.title('2D histogram of the provided VERITAS gamma-ray data')
# plt.savefig('hist2d_veritas.png')
# plt.show()

#################### q2, q3: no separate code

#################### q4
ra_pointing_deg=83.6329
dec_pointing_deg=22.5258
pointing_ctr_deg=SkyCoord(ra_pointing_deg, dec_pointing_deg, unit="deg")
print('pointing ctr coords=',pointing_ctr_deg)

ra_obj_deg_idx,dec_obj_deg_idx=np.unravel_index(h2dT.argmax(), h2dT.shape)
obj_ctr_deg=SkyCoord(xedges[ra_obj_deg_idx], yedges[dec_obj_deg_idx], unit="deg")
print('object ctr coords=',obj_ctr_deg)

plt.figure()
plt.imshow(h2d,extent=[xedges[0],xedges[-1],yedges[-1],yedges[0]],aspect=1) # Stackoverflow: hist2d bin populations need to be transposed before plotting (https://stackoverflow.com/questions/59795238/how-to-use-or-manipulate-the-output-return-values-of-hist2d-and-create-a-new-h)
plt.scatter(ra_pointing_deg,dec_pointing_deg,label='pointing centre',c='C1')
plt.scatter(xedges[ra_obj_deg_idx], yedges[dec_obj_deg_idx],label='object centre',c='C3')
plt.colorbar()
plt.xlabel('RA (deg)')
plt.ylabel('dec (deg)')
plt.title('2D histogram of the provided VERITAS gamma-ray data')
plt.legend()
plt.savefig('hist2d_veritas_with_centres.png')
plt.show()

# object_pointing_offset=pointing_ctr_deg-obj_ctr_deg # error bc algebra bw SkyCoord objects not supported
pointing_obj_position_angle=pointing_ctr_deg.position_angle(obj_ctr_deg) # this gives the midpoint
print('pointing_obj_position_angle=',pointing_obj_position_angle)
pointing_obj_separation=pointing_ctr_deg.separation(obj_ctr_deg)
print('pointing_obj_separation=',pointing_obj_separation)
object_pointing_offset=pointing_ctr_deg.directional_offset_by(pointing_obj_position_angle,pointing_obj_separation/2)
print('object pointing offset=',object_pointing_offset,'deg RA/dec')

######################################## statistical detection

#################### q1
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