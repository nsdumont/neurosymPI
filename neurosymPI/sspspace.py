import numpy as np
import scipy
from scipy.stats import qmc
from scipy.stats import special_ortho_group
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings


####################################
#  SSP space classes used in paper #
####################################

# An SSP S(x) is a d-dim real-valued vector that represents a n-dim varaible x
# where n<<d, i.e. we are projecting x into a high dimensional vector space
# Search "hyperdimensional computing" or "vector symbolic architectures" to learn more about this approach in general
# Or read " Simulating and Predicting Dynamical Systems With Spatial Semantic Pointers" to learn details about our particular approach

#########################
#  SSPSpace base class  #
#########################
# An SSP vector space is \mathcal{S} = {S(x)=IDFT(exp(iAx/l)) \in R^d | x \in \mathcal{X}} where \mathcal{X} is a subset of R^n, and 
#   A \in R^{d x n} is a fixed phase matrix, and l \in \mathcal{R}^n are the length scale parameters
# exp(iAx) will be a vector of phasors. The phases of the phasors is given by the similarity/dot product between x and the row vectors that make up A
class SSPSpace:
    def __init__(self, domain_dim: int, ssp_dim: int, axis_matrix=None, phase_matrix=None,
                 domain_bounds=None, length_scale=1):
        # A general way to specfiy a SSP vector space requries
        #   domain_dim: n, the dimensionality of the varaibles represented by the SSP (x)
        #   ssp_dim: d, the dimensionality of the SSP representation itself
        #   phase_matrix (optional if axis_matrix given): the SSP encoding is defined by the dxn matrix A. This can be provided as input via this arg
        #   axis_matrix (optional if phase_matrix given): this can be provided instead of phase_matrix. This is IDFT(exp(iA)) \in R^{d x n}. Sometimes it is preferable to define the SSP with this
        #   domain_bounds (optional): the bounds of the \mathcal{X}. 
        #           (SSPs can represent x from an unbounded space but decoding x from S(x) requries either bounds or a good initial guess. Also, the plotting methods requrie bounds to be given)
        #   length_scale (default=1): l, the length scale parameters. A decent rule of thumb is that if your bounds are [-b,b] then use l=b/10
        
        self.sample_points = None
        self.sample_ssps = None
        self.domain_dim = domain_dim
        self.ssp_dim = ssp_dim
        
        if not isinstance(length_scale, np.ndarray) or length_scale.size == 1:
            self.length_scale = length_scale * np.ones((self.domain_dim,))
        
        if domain_bounds is not None:
            assert domain_bounds.shape[0] == domain_dim
        
        self.domain_bounds = domain_bounds
        
        if (axis_matrix is None) & (phase_matrix is None):
            raise RuntimeError("SSP spaces must be defined by either a axis matrix or phase matrix. Use subclasses to construct spaces with predefined axes.")
        elif (phase_matrix is None):
            assert axis_matrix.shape[0] == ssp_dim, f'Expected ssp_dim {axis_matrix.shape[0]}, got {ssp_dim}.'
            assert axis_matrix.shape[1] == domain_dim
            self.axis_matrix = axis_matrix
            self.phase_matrix = (-1.j*np.log(np.fft.fft(axis_matrix,axis=0))).real
        elif (axis_matrix is None):
            assert phase_matrix.shape[0] == ssp_dim
            assert phase_matrix.shape[1] == domain_dim
            self.phase_matrix = phase_matrix
            self.axis_matrix = np.fft.ifft(np.exp(1.j*phase_matrix), axis=0).real
        
    # Updating the length scale. In some application you might want the length scale to be learned online
    def update_lengthscale(self, scale):
        if not isinstance(scale, np.ndarray) or scale.size == 1:
            self.length_scale = scale * np.ones((self.domain_dim,))
        else:
            assert scale.size == self.domain_dim
            self.length_scale = scale
        assert self.length_scale.size == self.domain_dim
    
    # Encoding a varaible x. Given x (\in R^{n x m} where m is the number of points to be encoded) as input, return SSPs S(x) \in R^{d x m}
    def encode(self,x):
        assert x.shape[0] == self.domain_dim
        ls_mat = np.atleast_2d(np.diag(1/self.length_scale.flatten()))
        scaled_x = ls_mat @ x
        data = np.fft.ifft( np.exp( 1.j * self.phase_matrix @ scaled_x ), axis=0 ).real
        return data
    
    # Encode x in the Fourier domain. Will return exp(iAx/l) (i.e. DFT{S(x)})
    def encode_fourier(self,x):
        assert x.shape[0] == self.domain_dim
        ls_mat = np.atleast_2d(np.diag(1/self.length_scale.flatten()))
        scaled_x = ls_mat @ x
        data =  np.exp( 1.j * self.phase_matrix @ scaled_x )
        return data
    
    # Decoding. Given an SSP S(x) return x. There are several method to do this. 
    # method='from-set' (recommnded): reliable, but slow. Needs num_sample_pts to be quite large
    #       ='direct-optim': can be faster than 'from-set' but not as reliable if num_init_pts is too low
    # This will be updated in the future with some new faster methods 
    def decode(self,ssp,method='from-set', num_sample_pts=10000,from_set_method='grid',num_init_pts =10):
        if method=='least-squares':
            # problems duw to complex log
            x = np.linalg.lstsq(self.phase_matrix, (1.j*np.log(np.fft.fft(ssp,axis=0))).real)[0]
            #raise NotImplementedError()
            #fssp = np.fft.fft(ssp,axis=0)
            #x = np.linalg.lstsq(np.tile(self.phase_matrix,(2,1)), np.hstack([np.arccos(fssp.real), np.arcsin(fssp.imag)]))
            return x
        elif method=='from-set':
            sample_ssps, sample_points = self.get_sample_ssps(num_sample_pts,method=from_set_method)
            sims = sample_ssps.T @ ssp
            return sample_points[:,np.argmax(sims)]
        elif method=='direct-optim':
            x0 = self.decode(ssp, method='from-set',num_sample_pts=num_init_pts)
            def min_func(x,target=ssp):
                x_ssp = self.encode(np.atleast_2d(x))
                return -np.inner(x_ssp, target).flatten()
            soln = minimize(min_func, x0, method='L-BFGS-B')
            return soln.x
        elif method=='grad_descent':
            x = self.decode(ssp, method='from-set',num_sample_pts=num_init_pts)
            fssp = np.fft.fft(ssp,axis=0)
            ls_mat = np.diag(1/self.length_scale.flatten())
            for j in range(10):
                scaled_x = ls_mat @ x
                x_enc = np.exp(1.j * self.phase_matrix @ scaled_x)
                grad_mat = (1.j * (self.phase_matrix @ ls_mat).T * x_enc)
                grad =  (grad_mat @ fssp.T).flatten()
                x = x - 0.1*grad.real
            return x
        elif method=='nonlin-reg':
            x = self.decode(ssp, method='from-set',num_sample_pts=num_init_pts)
            fssp = np.fft.fft(ssp,axis=0)
            dy = np.hstack([fssp.real, fssp.imag])

            ls_mat = np.diag(1/self.length_scale.flatten())
            for j in range(10):
                J = np.vstack([self.phase_matrix * np.sin(self.phase_matrix @ x @ ls_mat).reshape(1,-1),
                               -self.phase_matrix * np.cos(self.phase_matrix @ x @ ls_mat).reshape(1,-1)])
                soln = np.linalg.pinv(J.T @ J) @ J.T @ dy
                x = x + soln
            return x
        else:
            raise NotImplementedError()
        
    # Given a vector S, decoding x from S, then encode S(x) and return
    # This is useful when S is not a perfect SSP but a noisey one that needs to be 'cleaned-up'
    def clean_up(self,ssp,**kwargs):
        x = self.decode(ssp,**kwargs)
        return self.encode(x)
        
    # Sample points from the domain \mathcal{X}
    # Two methods: 'grid' (return points gridded over n-dim space) or 'sobol' (quasi-random numbers that cover space well)
    def get_sample_points(self,num_points,method='grid'):
        if self.domain_bounds is None:
            bounds = np.vstack([-10*np.ones(self.domain_dim), 10*np.ones(self.domain_dim)]).T
        else:
            bounds = self.domain_bounds
        if method=='grid':
                n_per_dim = int(num_points**(1/self.domain_dim))
                if n_per_dim**self.domain_dim != num_points:
                    warnings.warn((f'Evenly distributing points over a '
                                   f'{self.domain_dim} grid requires numbers '
                                   f'of samples to be powers of {self.domain_dim}.'
                                   f'Requested {num_points} samples, returning '
                                   f'{n_per_dim**self.domain_dim}'), RuntimeWarning)
                ### end if
                xs = np.linspace(bounds[:,0],bounds[:,1],n_per_dim)
                xxs = np.meshgrid(*[xs[:,i] for i in range(self.domain_dim)])
                sample_points = np.array([x.reshape(-1) for x in xxs])
                return sample_points
        elif method=='sobol':
            sampler = qmc.Sobol(d=self.domain_dim) 
            lbounds = bounds[:,0]
            ubounds = bounds[:,1]
            u_sample_points = sampler.random(num_points)
            sample_points = qmc.scale(u_sample_points, lbounds, ubounds)
            return sample_points.T 
        else:
            raise NotImplementedError()
            
        
    # Get sample SSPs. Calls get_sample_points then encodes the result
    def get_sample_ssps(self,num_points,**kwargs): # make new if num_pts different than whats stored?
        sample_points = self.get_sample_points(num_points,**kwargs)
        sample_ssps = self.encode(sample_points)
        return sample_ssps, sample_points
    
    # Normalize a vector. Note: all SSPs are unit length but this might be useful when dealing with noisey SSPs or sums of SSPs
    def normalize(self,ssp):
        return ssp/np.sqrt(np.sum(ssp**2))
    
    # Make a vector unitary. Note: all SSPs are unitary but this might be useful when dealing with noisey SSPs or sums of SSPs
    def unitary(self,ssp):
        fssp = np.fft.fft(ssp)
        fssp = fssp/np.sqrt(fssp.real**2 + fssp.imag**2)
        return np.fft.ifft(fssp).real 

    # Rather than decode a single point, decode a whole sequence of SSPs
    def decode_path(self, ssp_path, N_ma=None, n_samples = 10000):
        sample_ssps, sample_points = self.get_sample_ssps(n_samples)
        path = np.zeros((ssp_path.shape[0], self.domain_dim))
        max_sims = np.zeros(ssp_path.shape[0])
        for i in range(ssp_path.shape[0]):
            sims = sample_ssps.T @ ssp_path[i,:]
            max_sims[i] = np.max(sims)
            path[i,:] = sample_points[:,np.argmax(sims)]
        
        return path, max_sims
    
    # Make a similarity plot. Given an input vector 'ssp', compute its simiarity with SSPs representing 'n_grid' points tiled over the \mathcal{X} space
    # and plot these simiarities over space. This is a method for visualizing these high-dim SSP vectors 
    # For SSPs representing 1D data, this will be a line plot
    # For SSPs representing 2D data, this can be plotted as a heatmap, contour, or contourf plot (use keywarg 'plot_type')
    #           (Note: such information could also be visulized by a surface plot, but that is not included here)
    # This does not work for n>2
    def similarity_plot(self,ssp,n_grid=100,plot_type='heatmap',cmap="YlGnBu",ax=None,**kwargs):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
        if self.domain_dim == 1:
            xs = np.linspace(self.domain_bounds[0,0],self.domain_bounds[0,1], n_grid)
            ax.plot(xs, self.ssp_space.encode(xs.reshape(1,-1)).T @ self.data)
            ax.set_xlim(self.domain_bounds[0,0],self.domain_bounds[0,1])
        elif self.domain_dim == 2:
            xs = np.linspace(self.domain_bounds[0,0],self.domain_bounds[0,1], n_grid)
            ys = np.linspace(self.domain_bounds[1,0],self.domain_bounds[1,1], n_grid)
            X,Y = np.meshgrid(xs,ys)
            sims = self.encode(np.vstack([X.reshape(-1),Y.reshape(-1)])).T @ ssp
            if plot_type=='heatmap':
                ax.pcolormesh(X,Y,sims.reshape(X.shape),cmap=cmap,**kwargs)
            elif plot_type=='contour':
                ax.contour(X,Y,sims.reshape(X.shape),cmap=cmap,**kwargs)
            elif plot_type=='contourf':
                ax.contourf(X,Y,sims.reshape(X.shape),cmap=cmap,**kwargs)
        else:
            raise NotImplementedError()
        return ax
        
###################################################
#  Subclasses for using particular phase_matrix A #
####################################################   
            
          
#################################################
#  RandomSSPSpace uses a random phase_matrix A #
#################################################
# No extra parameters are requried for this subclass. rng is an optional kwarg for specfiying a random generator for reproducibility
# This will construct a random A is such a way that exp(iA) will have conjugate symmertry so resulting SSPs will be real-valued
class RandomSSPSpace(SSPSpace):
    def __init__(self, domain_dim: int, ssp_dim: int,  domain_bounds=None, length_scale=1, rng=np.random.default_rng()):
        partial_phases = rng.random.rand(ssp_dim//2,domain_dim)*2*np.pi - np.pi
        axis_matrix = _constructaxisfromphases(partial_phases)
        super().__init__(domain_dim,ssp_dim,axis_matrix=axis_matrix,
                       domain_bounds=domain_bounds,length_scale=length_scale)

##############################################################
#  HexagonalSSPSpace uses a special structured phase_matrix A #
##############################################################
# This constructs A as follows
#       Compute the vectors that make up an n-simplex inscribed in a unit hypersphere
#       Create 'n_scales' scalar values from 'scale_min' to 'scale_max'
#       Create 'n_rotates' rotation matrices 
#       Create all permutations of scaling and rotating the vectors that make up the simplex
#       Concatenate all resulting vectors together to create a ((n+1)*n_rotates*n_scales) x n matrix
#       Make that matrix have conjugate symmertry, resulting in a (2*(n+1)*n_rotates*n_scales + 1) x n matrix
#       That matrix is A
# SSPs defined by such a A have esrieable properties, including smoother similairty plots. They can be represented by populations of grid cells too
# See "Accurate representation for spatial cognition using grid cells" for more details
class HexagonalSSPSpace(SSPSpace):
    def __init__(self,  domain_dim:int,ssp_dim: int=151, n_rotates:int=5, n_scales:int=5, 
                 scale_min=2*np.pi/np.sqrt(6) - 0.5, scale_max=2*np.pi/np.sqrt(6) + 0.5,
                 domain_bounds=None, length_scale=1):

        if (n_rotates==5) & (n_scales==5) & (ssp_dim != 151):
            n_rotates = int(np.max([1,np.sqrt((ssp_dim-1)/(2*(domain_dim+1)))]))
            n_scales = n_rotates
            
        phases_hex = np.hstack([np.sqrt(1+ 1/domain_dim)*np.identity(domain_dim) - (domain_dim**(-3/2))*(np.sqrt(domain_dim+1) + 1),
                         (domain_dim**(-1/2))*np.ones((domain_dim,1))]).T
        
        self.grid_basis_dim = domain_dim + 1
        self.num_grids = n_rotates*n_scales

        scales = np.linspace(scale_min,scale_max,n_scales)
        phases_scaled = np.vstack([phases_hex*i for i in scales])
        
        if (n_rotates==1):
            phases_scaled_rotated = phases_scaled
        elif (domain_dim==1):
            scales = np.linspace(scale_min,scale_max,n_scales+n_rotates)
            phases_scaled_rotated = np.vstack([phases_hex*i for i in scales])
        elif (domain_dim == 2):
            angles = np.linspace(0,2*np.pi/3,n_rotates)
            R_mats = np.stack([np.stack([np.cos(angles), -np.sin(angles)],axis=1),
                        np.stack([np.sin(angles), np.cos(angles)], axis=1)], axis=1)
            phases_scaled_rotated = (R_mats @ phases_scaled.T).transpose(0,2,1).reshape(-1,domain_dim)
        else:
            R_mats = special_ortho_group.rvs(domain_dim, size=n_rotates)
            phases_scaled_rotated = (R_mats @ phases_scaled.T).transpose(0,2,1).reshape(-1,domain_dim)
        
        axis_matrix = _constructaxisfromphases(phases_scaled_rotated)
        ssp_dim = axis_matrix.shape[0]
        super().__init__(domain_dim,ssp_dim,axis_matrix=axis_matrix,
                       domain_bounds=domain_bounds,length_scale=length_scale)
  
    # This returns vectors which can be used as encoders in nengo networks to obtain grid cells
    def sample_grid_encoders(self, n):
        sample_pts = self.get_sample_points(n)
        N = self.num_grids
        if N < n:
            sorts = np.hstack([np.arange(N), np.random.randint(0, N - 1, size = n - N)])
        else:
            sorts = np.arange(n)
        encoders = np.zeros((self.ssp_dim,n))
        for i in range(n):
            sub_mat = _get_sub_SSP(sorts[i],N,sublen=self.grid_basis_dim)
            proj_mat = _proj_sub_SSP(sorts[i],N,sublen=self.grid_basis_dim)
            sub_space = SSPSpace(self.domain_dim,2*self.grid_basis_dim + 1, axis_matrix= sub_mat @ self.axis_matrix)
            encoders[:,i] = N * proj_mat @ sub_space.encode(sample_pts[:,i])
        return encoders

####################
# Helper functions #
####################
    
def _constructaxisfromphases(K):
    d = K.shape[0]
    n = K.shape[1]
    axes = np.ones((d*2 + 1,n))
    for i in range(n):
        F = np.ones((d*2 + 1,), dtype="complex")
        F[0:d] = np.exp(1.j*K[:,i])
        F[-d:] = np.flip(np.conj(F[0:d]))
        F = np.fft.ifftshift(F)
        axes[:,i] = np.fft.ifft(F).real
    return axes

def _get_sub_FourierSSP(n, N, sublen=3):
    # Return a matrix, \bar{A}_n
    # Consider the multi scale representation (S_{total}) and sub vectors (S_n) described in the paper 
    # Then
    # \bar{A}_n F{S_{total}} = F{S_n}
    # i.e. pick out the sub vector in the Fourier domain
    tot_len = 2*sublen*N + 1
    FA = np.zeros((2*sublen + 1, tot_len))
    FA[0:sublen, sublen*n:sublen*(n+1)] = np.eye(sublen)
    FA[sublen, sublen*N] = 1
    FA[sublen+1:, tot_len - np.arange(sublen*(n+1),sublen*n,-1)] = np.eye(sublen)
    return FA

def _get_sub_SSP(n,N,sublen=3):
    # Return a matrix, A_n
    # Consider the multi scale representation (S_{total}) and sub vectors (S_n) described in the paper 
    # Then
    # A_n S_{total} = S_n
    # i.e. pick out the sub vector in the time domain
    tot_len = 2*sublen*N + 1
    FA = _get_sub_FourierSSP(n,N,sublen=sublen)
    W = np.fft.fft(np.eye(tot_len))
    invW = np.fft.ifft(np.eye(2*sublen + 1))
    A = invW @ np.fft.ifftshift(FA) @ W
    return A.real

def _proj_sub_FourierSSP(n,N,sublen=3):
    # Return a matrix, \bar{B}_n
    # Consider the multi scale representation (S_{total}) and sub vectors (S_n) described in the paper 
    # Then
    # \sum_n \bar{B}_n F{S_{n}} = F{S_{total}}
    # i.e. project the sub vector in the Fourier domain such that summing all such projections gives the full vector in Fourier domain
    tot_len = 2*sublen*N + 1
    FB = np.zeros((2*sublen + 1, tot_len))
    FB[0:sublen, sublen*n:sublen*(n+1)] = np.eye(sublen)
    FB[sublen, sublen*N] = 1/N # all sub vectors have a "1" zero freq term so scale it so full vector will have 1 
    FB[sublen+1:, tot_len - np.arange(sublen*(n+1),sublen*n,-1)] = np.eye(sublen)
    return FB.T

def _proj_sub_SSP(n,N,sublen=3):
    # Return a matrix, B_n
    # Consider the multi scale representation (S_{total}) and sub vectors (S_n) described in the paper 
    # Then
    # \sum_n B_n S_{n} = S_{total}
    # i.e. project the sub vector in the time domain such that summing all such projections gives the full vector
    tot_len = 2*sublen*N + 1
    FB = _proj_sub_FourierSSP(n,N,sublen=sublen)
    invW = np.fft.ifft(np.eye(tot_len))
    W = np.fft.fft(np.eye(2*sublen + 1))
    B = invW @ np.fft.ifftshift(FB) @ W
    return B.real
