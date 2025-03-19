import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import correlate2d

cam_unit=np.reshape(np.arange(0,64),(8,8)) # unrealistically non-binary test case to check that things are being tiled/stacked as intended
# plt.figure()
# plt.imshow(cam_unit)
# plt.colorbar()
# plt.title('cam unit')
# plt.show()

n_cam_units=8
cam_tiled_row=np.tile(cam_unit,n_cam_units)
# plt.figure()
# plt.imshow(cam_tiled_row)
# plt.colorbar()
# plt.title('cam tiled as row')
# plt.show()

def recursive_hstack(array,ncopies):
    accumulated=np.copy(array)
    for i in range(ncopies-1):
        accumulated=np.vstack([accumulated,array])
    return accumulated

cam_tiled_square=recursive_hstack(cam_tiled_row,n_cam_units)
cam_tiled_square_autocorr=correlate2d(cam_tiled_square,cam_tiled_square)
# plt.figure()
# plt.imshow(cam_tiled_square)
# plt.colorbar()
# plt.title('cam tiled as square')
# plt.show()

fig,axs=plt.subplots(1,2,figsize=(10,5))
im=axs[0].imshow(cam_tiled_square)
axs[0].set_title('tiled CAM')
plt.colorbar(im, ax=axs[0])
im=axs[1].imshow(cam_tiled_square_autocorr)
axs[1].set_title('tiled CAM autocorrelation')
plt.colorbar(im, ax=axs[1])
plt.show()


##im=axs[0].imshow(scipy_quad_inner_r_integral,extent=[0.,ubound,ubound,0.]) # [L,R,B,T]
    # plt.colorbar(im, ax=axs[0])