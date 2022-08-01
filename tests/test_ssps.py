import numpy as np
import scipy
import neurosymPI


domain_dim = 2
bounds = 15*np.array([[-1,1],[-1,1]])
# Creating an SSP space
# domain_dim - dimensionality of underlying space (most commonly 2 or 3)
# domain_bounds - bounds on that space (needed for the decode and plot functions in HexagonalSSPSpace)
# ssp_dim - dimensionality of SSP representation
# scale_min, scale_max, length_scale - SSP parameters
ssp_space = neurosymPI.sspspace.HexagonalSSPSpace(domain_dim,ssp_dim=151, 
                 scale_min=2*np.pi/np.sqrt(6) - 0.5, scale_max=2*np.pi/np.sqrt(6) + 0.5,
                 domain_bounds=bounds, length_scale=1)
S0 = ssp_space.encode(np.array([1.3,-3.4]))

print(ssp_space.decode(S0,method='from-set'))

ssp_space.similarity_plot(S0)
ssp_space.similarity_plot(S0,plot_type='contour',vmin=0)