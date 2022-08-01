import nengo
import numpy as np
import matplotlib.pyplot as plt
import neurosymPI




domain_dim = 2
bounds = 15*np.array([[-1,1],[-1,1]])
ssp_space = neurosymPI.sspspace.HexagonalSSPSpace(domain_dim,ssp_dim=151, 
                 scale_min=2*np.pi/np.sqrt(6) - 0.5, scale_max=2*np.pi/np.sqrt(6) + 0.5,
                 domain_bounds=bounds, length_scale=1)
d = ssp_space.ssp_dim

T = 15
dt = 0.001
timesteps = np.arange(0, T, dt)
path = np.vstack([np.linspace(-5,5,len(timesteps)), np.linspace(-5,5,len(timesteps))]).T


pathlen = path.shape[0]
vels = (1/dt)*( path[(np.minimum(np.floor(timesteps/dt) + 1, pathlen-1)).astype(int),:] -
               path[(np.minimum(np.floor(timesteps/dt), pathlen-2)).astype(int),:])
real_freqs = (ssp_space.phase_matrix @ vels.T)
scale_fac = 1
vels_scaled = vels*scale_fac

real_ssp = ssp_space.encode(path.T)


n_neurons = 5000
tau = 0.05
model = nengo.Network(seed=1)
with model:
    vel_input = nengo.Node(lambda t: vels_scaled[int(t/dt)-1,:], size_out=2)
    stim = nengo.Node(lambda t: real_ssp[:,int(t/dt)-1] if t<20*dt else np.zeros((d,)))
    pathintegrator = neurosymPI.networks.PathIntegration(ssp_space, tau, n_neurons, 
                  scaling_factor=scale_fac)
    nengo.Connection(vel_input,pathintegrator.velocity_input,synapse=None)
    nengo.Connection(stim,pathintegrator.input,synapse=None)
    
    ssp_p  = nengo.Probe(pathintegrator.output, synapse=None)
    
sim = nengo.Simulator(model)
sim.run(T)
