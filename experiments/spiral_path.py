import nengo
import numpy as np
import matplotlib.pyplot as plt
import time
import neurosymPI
from path_generation import spiral_path



radius=9
T = 60*2
dt = 0.001
timesteps = np.arange(0, T, dt)
path = spiral_path(len(timesteps), radius)

domain_dim = 2
bounds = np.array([[np.min(path[:,0])-1,np.max(path[:,0])+1],[np.min(path[:,1])-1,np.max(path[:,1])+1]])
ssp_space = neurosymPI.sspspace.HexagonalSSPSpace(domain_dim,ssp_dim=151, 
                 scale_min=2*np.pi/np.sqrt(6) - 0.5, scale_max=2*np.pi/np.sqrt(6) + 0.5,
                 domain_bounds=bounds, length_scale=1)
d = ssp_space.ssp_dim

ax = ssp_space.similarity_plot(ssp_space.encode(path[0,:]))
ax.plot(path[:,0],path[:,1])

xs = np.linspace(np.min(path[:,0])-0.2,np.max(path[:,0])+0.2,100)
ys = np.linspace(np.min(path[:,1])-0.2,np.max(path[:,1])+0.2,100)

pathlen = path.shape[0]
vels = (1/dt)*( path[1:,:] - path[:-1,:])
max_v = np.max(np.abs(vels))
real_freqs = (ssp_space.phase_matrix @ vels.T)
scale_fac = 2/(np.max(real_freqs) - np.min(real_freqs))
vels_scaled = vels*scale_fac

real_ssp = ssp_space.encode(path.T)


def save_node(filename,t,x):
    if (np.abs(t % 0.003) < 1e-6):
        f = open(filename,"a+")
        f.write(",".join(map(str,x)) + "\n")
        f.close()
    return 0

filename1 = "osc_spikes_spiral_1.txt"
filename2 = "grid_spikes_spiral.txt"

osc_p_file = open(filename1,"w+")
grid_p_file = open(filename2,"w+")


n_neurons = 2000
n_neurons2 = 10000
tau = 0.05
simdt=0.001
model = nengo.Network(seed=1)
with model:
    vel_input = nengo.Node(lambda t: vels_scaled[np.argmax(t-timesteps <= 0)-1,:], size_out=2)
    stim = nengo.Node(lambda t: real_ssp[:,int(t/dt)-1] if t<0.1 else np.zeros((d,)))
    pathintegrator = neurosymPI.networks.PathIntegration(ssp_space, tau, n_neurons, 
                  scaling_factor=scale_fac, stable=True)
    nengo.Connection(vel_input,pathintegrator.velocity_input, synapse=None)
    nengo.Connection(stim,pathintegrator.input, synapse=None)
    
    gridcells = neurosymPI.networks.SSPNetwork(ssp_space, n_neurons2)
    nengo.Connection(pathintegrator.output,gridcells.input, synapse=None)
    
    osc_n_p = nengo.Node(lambda t,x: save_node(filename1,t,x), size_in=n_neurons)
    nengo.Connection(pathintegrator.oscillators.ea_ensembles[1].neurons, osc_n_p,synapse=None)
    
    grid_n_p = nengo.Node(lambda t,x: save_node(filename2,t,x), size_in=n_neurons2)
    nengo.Connection(gridcells.ssp.neurons, grid_n_p,synapse=None)

    osc_p = nengo.Probe(pathintegrator.oscillators.output, synapse=tau)
    ssp_p  = nengo.Probe(pathintegrator.output, synapse=None)
    ssp2_p  = nengo.Probe(gridcells.ssp, synapse=tau)
    
sim = nengo.Simulator(model,dt=simdt)
sim.run(T)
osc_p_file.close()
grid_p_file.close()

timestr = 'pathint_spiral_' + time.strftime("%Y%m%d-%H%M%S") + '.npz'
np.savez(timestr, sim_ssp = sim.data[ssp_p], sim_osc = sim.data[osc_p], path=path, ts= sim.trange(),
         ssp_space=ssp_space)



sim_path, max_sims = ssp_space.decode_path(sim.data[ssp_p])


def plot_path(path, real_path, N_ma=None):
        fig = plt.figure(figsize=(6.50127/3, 6.50127/3))
        ax = fig.add_subplot(111)
        if real_path is not None:
            ax.plot(real_path[:,0],real_path[:,1],c='lightgrey',linewidth=1,zorder = 1)
        if N_ma is None:
            ax.plot(path[:,0],path[:,1],linestyle='--',color="black",zorder = 1,linewidth=0.7)
        else:
            path_ma = np.vstack([np.convolve(path[:,0], np.ones(N_ma)/N_ma, mode='valid'), 
                                  np.convolve(path[:,1], np.ones(N_ma)/N_ma, mode='valid')]).T
            ax.plot(path_ma[:,0],path_ma[:,1],linestyle='--',color="black",zorder = 1,linewidth=0.7)
        return ax

ax = plot_path(sim_path, path, N_ma=200)
ax.set_xlim(-7,2.1)
ax.set_ylim(-7,2.1)
ax.set_axis_off()
