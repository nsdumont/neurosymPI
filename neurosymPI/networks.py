import nengo
import numpy as np

#################################
#  Nengo networks used in paper #
#################################


######################################################################
#  A network that represents an HexSSP (given as input) with grid cells #
#######################################################################
# As input it needs an SSPSpace class object and 'n_neurons', the number of neurons to use in the network
class SSPNetwork(nengo.network.Network):
    def __init__(self, ssp_space, n_neurons, **kwargs):
        # Input with initializing: ssp_space
        super().__init__()
        d = ssp_space.ssp_dim
        G_encoders = ssp_space.sample_grid_encoders(n_neurons).T
        with self:
            self.input = nengo.Node(size_in=d)
            self.ssp = nengo.Ensemble(n_neurons, d, encoders=G_encoders,**kwargs)
            nengo.Connection(self.input, self.ssp)


###################################
#  The path integration network  #
###################################
# High level: given only velocity information over time (and, optionally, an initial SSP for starting position), represent the SSP 
#       estimation of current position over time, computed via recurrent connections in a spiking neural network
# Some details: This creates an EnsembleArray (a set of indpendent neural populations). Each of the populations is a VCO that encodes the 
#       real & imag values of one componet of the SSP in the Fourier domain, along with the freq at which these componets are oscillating in 
#       the Fourier domain.
#       Each population is recurrently connected to itself with weights set to approx the dynamics described in the paper
# Input:
#   ssp_space: a SSPSpace class object
#   recurrent_tau: the post-synatipic time constant used in the connections that implement the dyanmical system (suggested values: 0.01-0.1)
#   n_neurons: the number of neurons used in each population within the EnsembleArray
#   scaling_factor (default=1): for scaling the velocity input
#   stable (default=True): whether or not to use nonlinear attractor dynamics. A function can be passed instead of a boolean if you wish to define your own dynamics
# Ouput:
#   A nengo Network object that can be embedded inside other networks
# Use:
#   Create a nengo Network and define this inside the model. Provide input to the 'velocity_input' node and 'input' node, read out the position estimate 
#   from 'output'
#  Exmaple (se scripts in 'experiments' folder for complete examples):
#   model = nengo.Network()
#     with model:
#         # Create inputs
#         vel_input = nengo.Node( **insert some function that return velocity over time** , size_out=n)
#         stim = nengo.Node(lambda t: **insert initial SSP** if t<0.1 else np.zeros((d,))) #IMPORTANT: only provide this initial SSP for a short time at the begining
#
#         #Create PI model
#         pathintegrator = PathIntegration(ssp_space, tau, n_neurons, 
#                                           scaling_factor=1, stable=True)
#
#         # Connect inputs to model
#         nengo.Connection(vel_input,pathintegrator.velocity_input, synapse=None)
#         nengo.Connection(stim,pathintegrator.input, synapse=None)
#         
#         # Read out the model's output
#         ssp_p  = nengo.Probe(pathintegrator.output, synapse=None)
#
#   # Run the model for T seconds
#   sim = nengo.Simulator(model,dt=simdt)
#   sim.run(T)
#
#   # Now we have sim.data[ssp_p], a nt by d matrix with the position estimates (represented as d-dim SSPs) over the simulation (with nt sim timesteps)
#   # To decode the position estimates in X space:
#   sim_path, _ = ssp_space.decode_path(sim.data[ssp_p])
#
#   plt.plot(sim_path[:,0],sim_path[:,1])

class PathIntegration(nengo.network.Network):
    def __init__(self, ssp_space, recurrent_tau, n_neurons,
                 scaling_factor=1, stable=True, **kwargs):
        super().__init__(**kwargs)
        
        # Defining the dyanmical system to approximate via the recurrent connections
        # NOTE: the network does not just use this function exactly while running,
        #       this function is just used to train the recurrent connection weights before the simulation
        if stable==True:
           # Attractor version of dynamics
            def feedback(x):
                w = (x[0]/scaling_factor)/ssp_space.length_scale[0] ###!!!
                r = np.maximum(np.sqrt(x[1]**2 + x[2]**2), 1e-8)
                dx1 = x[1]*(1-r**2)/r - x[2]*w 
                dx2 = x[2]*(1-r**2)/r + x[1]*w 
                return recurrent_tau*dx1 + x[1], recurrent_tau*dx2 + x[2]
        elif callable(stable):
            # User-defined dynamics
            feedback = stable
        else:
            # Non-attractor (i.e. just a simple harmonic osc) version of dynamics
            def feedback(x): 
                w = (x[0]/scaling_factor)/ssp_space.length_scale[0]
                dx1 = - x[2]*w
                dx2 =  x[1]*w
                return recurrent_tau*dx1 + x[1], recurrent_tau*dx2 + x[2]

        
        d = ssp_space.ssp_dim
        N = ssp_space.domain_dim
        n_oscs = d//2
        self.n_oscs = n_oscs
        reordered_phases = np.fft.fftshift(ssp_space.phase_matrix,axes=0)[:d//2,:]
        
        real_ids = np.arange(1,n_oscs*3,3)
        imag_ids = np.arange(2,n_oscs*3,3)
        S_ids = np.zeros(n_oscs*2 + 1, dtype=int)
        S_ids[0:d//2] = real_ids
        S_ids[d//2:(n_oscs*2)] = imag_ids
        S_ids[-1] = n_oscs*3 
        
        to_SSP = _get_to_SSP_mat(d)
        from_SSP = _get_from_SSP_mat(d)
        with self:
            # Velocity input should be passed to this node
            self.velocity_input = nengo.Node(label="velocity_input", size_in=N)
            # SSP representing starting postion (or corrections, if you'd like) can be passed to this node
            self.input = nengo.Node(label="input", size_in=d)
            
            # A neural pop representing the velocity
            self.velocity = nengo.Ensemble(n_neurons, dimensions=N,label='velocity')
            nengo.Connection(self.velocity_input, self.velocity)
            
            # The set of VCOs representing the SSP in the Fourier domain
            self.oscillators = nengo.networks.EnsembleArray(n_neurons, n_oscs + 1, 
                                                            ens_dimensions = 3,
                                                            radius=np.sqrt(2), label="oscillators")
            self.oscillators.output.output = lambda t, x: x
            
            nengo.Connection(self.input,self.oscillators.input, transform=from_SSP, synapse=None)
            
            for i in np.arange(n_oscs):
                
                # Dot product of velocity & ith row of phase matrix A is the frequency of VCO. The VCO pop gets this as input 
                nengo.Connection(self.velocity, self.oscillators.ea_ensembles[i][0], 
                                 transform = reordered_phases[i,:].reshape(1,-1), synapse=recurrent_tau)
                
                # The VCO pop is recurrently connnected to implement the VCO dynamics 
                nengo.Connection(self.oscillators.ea_ensembles[i], self.oscillators.ea_ensembles[i][1:], 
                                 function=feedback, 
                                 synapse=recurrent_tau)
                
            zerofreq = nengo.Node([1,0,0])
            nengo.Connection(zerofreq, self.oscillators.ea_ensembles[-1])
            
            self.output = nengo.Node(size_in=d)
            nengo.Connection(self.oscillators.output[S_ids], self.output, 
                             transform = to_SSP/0.97, synapse=0.05)
   


###################################
#  The gated memory network  #
###################################
class InputGatedMemory(nengo.Network):
    def __init__(
        self,
        n_neurons,
        dimensions,
        feedback=1.0,
        gain=1.0,
        recurrent_synapse=0.1,
        difference_synapse=None,
        **kwargs,
    ):

        
        super().__init__(**kwargs)

        if difference_synapse is None:
            difference_synapse = recurrent_synapse

        n_total_neurons = n_neurons * dimensions

        with self:
            # integrator to store value
            self.mem = nengo.networks.EnsembleArray(n_neurons, dimensions, label="mem")
            nengo.Connection(
                self.mem.output,
                self.mem.input,
                transform=feedback,
                synapse=recurrent_synapse,
            )

            # calculate difference between stored value and input
            self.input_ns = nengo.networks.EnsembleArray(n_neurons, dimensions)

            # feed difference into integrator
            nengo.Connection(
                self.input_ns.output,
                self.mem.input,
                transform=gain,
                synapse=difference_synapse,
            )

            # gate difference (if gate==0, update stored value,
            # otherwise retain stored value)
            self.gate = nengo.Node(size_in=1)
            self.input_ns.add_neuron_input()
            nengo.Connection(
                self.gate,
                self.input_ns.neuron_input,
                transform=np.ones((n_total_neurons, 1)) * -10,
                synapse=None,
            )

            # reset input (if reset=1, remove all values, and set to 0)
            self.reset = nengo.Node(size_in=1)
            nengo.Connection(
                self.reset,
                self.mem.add_neuron_input(),
                transform=np.ones((n_total_neurons, 1)) * -3,
                synapse=None,
            )

        self.input = self.input_ns.input
        self.output = self.mem.output
    
    
#######################
#  Helper functions  #
########################

def _get_to_SSP_mat(D):
    W = np.fft.ifft(np.eye(D))
    W1 = W.real @ np.fft.ifftshift(np.eye(D),axes=0)
    W2 = W.imag @ np.fft.ifftshift(np.eye(D),axes=0)
    shiftmat1 = np.vstack([np.eye(D//2), np.zeros((1,D//2)), np.flip(np.eye(D//2), axis=0)])
    shiftmat2 = np.vstack([np.eye(D//2), np.zeros((1,D//2)), -np.flip(np.eye(D//2), axis=0)])
    shiftmat = np.vstack([ np.hstack([shiftmat1, np.zeros(shiftmat2.shape)]),
                          np.hstack([np.zeros(shiftmat2.shape), shiftmat2])])
    shiftmat = np.hstack([shiftmat, np.zeros((shiftmat.shape[0],1))])
    shiftmat[D//2,-1] = 1
    tr = np.hstack([W1, -W2]) @ shiftmat 
    return tr

def _get_from_SSP_mat(D):
    W = np.fft.fft(np.eye(D))
    W1 = np.fft.fftshift(np.eye(D),axes=0) @ W.real 
    W2 = np.fft.fftshift(np.eye(D),axes=0) @ W.imag 
    W1 = W1[:(D//2 + 1),:]
    W2 = W2[:(D//2 + 1),:]
    shiftmat = np.zeros((3*(D//2 + 1),2*(D//2 + 1)))
    shiftmat[1::3,:(D//2 + 1)] = np.eye(D//2 +1)
    shiftmat[2::3,(D//2 + 1):] = np.eye(D//2 +1)
    
    tr = shiftmat @ np.vstack([W1, W2]) 
    return tr


     
# Helper funstions 
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
