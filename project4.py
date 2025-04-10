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
    # print(hdulist.info())
    ra_rad=hdulist[1].data['RA(rad)']
    dec_rad=hdulist[1].data['Dec(rad)']
    mscl=hdulist[1].data['mscl']
    mscw=hdulist[1].data['mscw']
# probe_fits(gammafile)
ra_deg=ra_rad*180./np.pi
dec_deg=dec_rad*180./np.pi

h2dT,xedges,yedges=np.histogram2d(ra_deg,dec_deg,bins=500)
h2d=h2dT.T

plot_hist2d=False
if plot_hist2d:
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

ra_obj_ctr_hms='05h34m31.8s'
dec_obj_ctr_hms='22h01m03s'
ra_obj_ctr_hrs=5+34./60.+31.8/3600.
ra_obj_ctr_deg=ra_obj_ctr_hrs*360/24
dec_obj_ctr_deg=22+1./60.+3./3600.

obj_ctr_deg=SkyCoord(ra_obj_ctr_deg, dec_obj_ctr_deg, unit='deg')
print('object ctr coords=',obj_ctr_deg)

plot_hist2d_with_centres=False
if plot_hist2d_with_centres:
    plt.figure()
    plt.hist2d(ra_deg,dec_deg,bins=500) # Stackoverflow: hist2d bin populations need to be transposed before plotting (https://stackoverflow.com/questions/59795238/how-to-use-or-manipulate-the-output-return-values-of-hist2d-and-create-a-new-h)
    plt.colorbar()
    plt.scatter(ra_pointing_deg,dec_pointing_deg,label='pointing centre',c='C1')
    plt.scatter(ra_obj_ctr_deg,dec_obj_ctr_deg,label='object centre',c='C3')
    plt.xlabel('RA (deg)')
    plt.ylabel('dec (deg)')
    plt.axis('equal')
    plt.title('2D histogram of the provided VERITAS gamma-ray data')
    plt.legend()
    plt.savefig('hist2d_veritas_with_centres.png')
    plt.show()

ra_offset_deg=ra_pointing_deg-ra_obj_ctr_deg
dec_offset_deg=dec_pointing_deg-dec_obj_ctr_deg
print('ra_offset_deg,dec_offset_deg=',ra_offset_deg,dec_offset_deg)
ref_a=SkyCoord(ra_pointing_deg-dec_offset_deg*np.cos(dec_pointing_deg),dec_pointing_deg, unit='deg') # "left" of the pointing centre
ref_b=SkyCoord(ra_pointing_deg,dec_pointing_deg+dec_offset_deg, unit='deg') # "below" the pointing centre
ref_c=SkyCoord(ra_pointing_deg+dec_offset_deg*np.cos(dec_pointing_deg),dec_pointing_deg, unit='deg') # "right" of the pointing centre
# ref_a=SkyCoord(ra_pointing_deg-dec_offset_deg,dec_pointing_deg, unit='deg') # "left" of the pointing centre
# ref_b=SkyCoord(ra_pointing_deg,dec_pointing_deg+dec_offset_deg, unit='deg') # "below" the pointing centre
# ref_c=SkyCoord(ra_pointing_deg+dec_offset_deg,dec_pointing_deg, unit='deg') # "right" of the pointing centre

check_stats_regions=True
if check_stats_regions:
    plt.figure()
    plt.hist2d(ra_deg,dec_deg,bins=500) # Stackoverflow: hist2d bin populations need to be transposed before plotting (https://stackoverflow.com/questions/59795238/how-to-use-or-manipulate-the-output-return-values-of-hist2d-and-create-a-new-h)
    plt.colorbar()
    plt.scatter(ra_pointing_deg,dec_pointing_deg,label='pointing centre',c='C1')
    plt.scatter(ra_obj_ctr_deg,dec_obj_ctr_deg,label='object centre',c='C3')
    plt.scatter(ra_pointing_deg-dec_offset_deg,dec_pointing_deg,label='ref a',c='C2')
    plt.scatter(ra_pointing_deg,dec_pointing_deg+dec_offset_deg,label='ref b',c='C4')
    plt.scatter(ra_pointing_deg+dec_offset_deg,dec_pointing_deg,label='ref c',c='C5')
    plt.xlabel('RA (deg)')
    plt.ylabel('dec (deg)')
    plt.axis('equal')
    plt.title('Check noise stats region positions')
    plt.legend()
    plt.savefig('check_stats_regions.png')
    plt.show()

######################################## statistical detection
#################### q1
def event_sep_from_reference(event_coords,reference_coords):
    return reference_coords.separation(event_coords).arcmin

counts_path="/Users/sophiarubens/Downloads/W25/PHYS641/projects/project4_VERITAS/stats_files/"
def recalculate_event_Ns(event_list_ras,event_list_decs,case):
    event_seps_from_source=event_sep_from_reference(SkyCoord(event_list_ras,event_list_decs,unit="deg"),obj_ctr_deg)
    event_seps_from_ref_a= event_sep_from_reference(SkyCoord(event_list_ras,event_list_decs,unit="deg"),ref_a)
    event_seps_from_ref_b= event_sep_from_reference(SkyCoord(event_list_ras,event_list_decs,unit="deg"),ref_b)
    event_seps_from_ref_c= event_sep_from_reference(SkyCoord(event_list_ras,event_list_decs,unit="deg"),ref_c)
    N_on=np.count_nonzero(event_seps_from_source<6)
    N_a=np.count_nonzero(event_seps_from_ref_a<6)
    N_b=np.count_nonzero(event_seps_from_ref_b<6)
    N_c=np.count_nonzero(event_seps_from_ref_c<6)
    np.savetxt(counts_path+case+'_N_values.txt',[N_on,N_a,N_b,N_c],fmt="%d",header="N_on,N_a,N_b,N_c")
    return N_on,N_a,N_b,N_c

def print_count_summary(N_on,N_a,N_b,N_c):
    print("N_on=",N_on)
    print("N_a=",N_a)
    print("N_b=",N_b)
    print("N_c=",N_c)
    print("N_off=",N_a+N_b+N_c," (=N_a+N_b+N_c)")
    return None

N_events=len(ra_deg)
print('N_events=',N_events)

base_case='no_cuts'
N_on,N_a,N_b,N_c=recalculate_event_Ns(ra_deg,dec_deg,case=base_case)
N_off=N_a+N_b+N_c
print("WITHOUT CUTS")
print_count_summary(N_on,N_a,N_b,N_c)

#################### q2: no separate code
#################### q3: for computational efficiency, accomplished in the same loop as in q1 above
#################### q4: no separate code
#################### q5
def S(alpha,N_on,N_off):
    term1=N_on*np.log((N_on*(1+alpha))/(alpha*(N_on+N_off)))
    term2=N_off*np.log(((1+alpha)*N_off)/(N_on+N_off))
    return np.sqrt(2)*np.sqrt(term1+term2)

alpha=1./3. # t_on/t_off
significance=S(alpha,N_on,N_off)
print("significance of this observation is", significance)

#################### q6: no separate code

######################################## cut optimization
#################### q1
nbins=100
plt.figure(figsize=(10,5))
plt.hist(mscl,bins=nbins,label="MSCL",alpha=0.7)
plt.hist(mscw,bins=nbins,label="MSCW",alpha=0.7)
plt.xlabel("Event scale proxy [dimensionless]")
plt.ylabel("Number of instances")
plt.title("Event scale parameter histograms")
plt.legend()
plt.tight_layout()
plt.savefig("mean_scale_histograms.png")
plt.show()

#################### q2
recalculate_cut_Ns=True
mscx_cut_candidates=np.arange(0.9,1.9,0.1)
n_xcut_candidates=len(mscx_cut_candidates)
cut_significances=np.zeros((n_xcut_candidates,n_xcut_candidates))
for i,mscw_cut_candidate in enumerate(mscx_cut_candidates):
    for j,mscl_cut_candidate in enumerate(mscx_cut_candidates):
        cut_case_id=str(i)+'_'+str(j)
        keep=(mscw<mscw_cut_candidate)&(mscl<mscl_cut_candidate)
        ra_deg_keep=ra_deg[keep]
        dec_deg_keep=dec_deg[keep]
        N_on_cut,N_a_cut,N_b_cut,N_c_cut=recalculate_event_Ns(ra_deg_keep,dec_deg_keep,case=cut_case_id)
        N_off_cut=N_a_cut+N_b_cut+N_c_cut

        cut_significances[i,j]=S(1./3.,N_on_cut,N_off_cut)

#################### q3
plt.figure()
plt.imshow(cut_significances,extent=[mscx_cut_candidates[0],mscx_cut_candidates[-1],mscx_cut_candidates[-1],mscx_cut_candidates[0]]) #extent=[L,R,B,T]
plt.xlabel('MSCW cut (ceiling)')
plt.ylabel('MSCL cut (ceiling)')
plt.title('Significance for all considered cut combinations')
plt.colorbar()
plt.tight_layout()
plt.savefig('significance_of_cut_combos.png')
plt.show()

print("\n\nCuts (ceilings) that lead to the highest significance:")
best_mscw_cut_idx,best_mscl_cut_idx=np.unravel_index(np.nanargmax(cut_significances), cut_significances.shape)
best_mscw_cut=mscx_cut_candidates[best_mscw_cut_idx]
best_mscl_cut=mscx_cut_candidates[best_mscl_cut_idx]
print("MSCW ->",best_mscw_cut)
print("MSCL ->",best_mscl_cut)
print("\nWITH CUTS")
N_off_cut=N_a_cut+N_b_cut+N_c_cut
print_count_summary(N_on_cut,N_a_cut,N_b_cut,N_c_cut)
print("significance of this observation is", cut_significances[best_mscw_cut_idx,best_mscl_cut_idx])

#################### q4: no separate code