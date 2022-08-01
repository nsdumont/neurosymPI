import nengo
import nengo_spa as spa
import numpy as np
import matplotlib.pyplot as plt
import neurosymPI
from path_generation import generate_signal





domain_dim = 2
bounds = 15*np.array([[-1,1],[-1,1]])
ssp_space = neurosymPI.sspspace.HexagonalSSPSpace(domain_dim,ssp_dim=295, 
                 scale_min=2*np.pi/np.sqrt(6) - 0.5, scale_max=2*np.pi/np.sqrt(6) + 0.5,
                 domain_bounds=bounds, length_scale=3)
d = ssp_space.ssp_dim


T = 20
dt = 0.001
timesteps = np.arange(0, T, dt)
radius=12
path,_ = generate_signal(T, dt, dims=2, rms=radius, limit=0.08, seed=8)
path[:,0] = (radius-1)*path[:,0]/np.max(np.abs(path[:,0]))
path[:,1] = (radius-1)*path[:,1]/np.max(np.abs(path[:,1]))

plt.figure(figsize=(5,5))
plt.plot(path[:,0],path[:,1])

xs = np.linspace(-radius,radius,100)
ys = np.linspace(-radius,radius,100)

pathlen = path.shape[0]
vels = (1/dt)*( path[(np.minimum(np.floor(timesteps/dt) + 1, pathlen-1)).astype(int),:] -
               path[(np.minimum(np.floor(timesteps/dt), pathlen-2)).astype(int),:])
max_v = np.max(np.abs(vels))
real_freqs = (ssp_space.phase_matrix @ vels.T)
scale_fac = 1/np.max(np.abs(real_freqs))
vels_scaled = vels*scale_fac

real_ssp = ssp_space.encode(path.T)


item_names = ["CIRCLE", "SQUARE", "TRIANGLE"]
item_dt = 0.5
def items(t):
    if np.abs(t - 5)< item_dt:
        return item_names[0]
    elif np.abs(t - 10)< item_dt:
        return item_names[1]
    elif np.abs(t - 15)< item_dt:
        return item_names[2]
    else:
        return "0"
    
def is_item(t):
    if np.abs(t - 5)< item_dt:
        return 0
    elif np.abs(t - 10)< item_dt:
        return 0
    elif np.abs(t - 15)< item_dt:
        return 0
    else:
        return 10
    
sample_ssps, _ = ssp_space.get_sample_ssps(5000)
def clean_up_fun(t,x):
    sims = sample_ssps.T @ x
    return sample_ssps[:,np.argmax(sims)]
    

n_neurons1 = 500
n_neurons = 800
tau = 0.05
model = spa.Network(seed=1)
with model:
    vel_input = nengo.Node(lambda t: vels_scaled[int(t/dt)-1,:], size_out=2)
    stim = nengo.Node(lambda t: real_ssp[:,int(t/dt)-1] if t<20*dt else np.zeros((d,)))
    pathintegrator = neurosymPI.networks.PathIntegration(ssp_space, tau, n_neurons1, 
                  scaling_factor=scale_fac, stable=True)
    nengo.Connection(vel_input,pathintegrator.velocity_input)
    nengo.Connection(stim,pathintegrator.input)
    
    item = spa.Transcode(items, output_vocab=ssp_space.ssp_dim)
    state = spa.State(vocab=ssp_space.ssp_dim, subdimensions=1)
    nengo.Connection(item.output, state.input)
    
    circonv = nengo.networks.CircularConvolution(n_neurons, ssp_space.ssp_dim)
    nengo.Connection(pathintegrator.output, circonv.input_a)
    nengo.Connection(state.output, circonv.input_b)
    
    memory = neurosymPI.networks.InputGatedMemory(n_neurons, ssp_space.ssp_dim, feedback=1,  gain=1, recurrent_synapse=0.1)
    nengo.Connection(circonv.output, memory.input)
    
    gate = nengo.Node(is_item)
    nengo.Connection(gate, memory.gate)
    
    items = []
    circonvs = []
    circonvs2 = []
    cleanup_nodes = []
    querylocs_ps = []
    queryvec_ps = []
    querycleanlocs_ps=[]
    cleanup_ens=[]
    for j in range(len(item_names)):
        # part 1
        items.append(spa.Transcode(item_names[j], output_vocab=ssp_space.ssp_dim))
        circonvs.append(nengo.networks.CircularConvolution(n_neurons, ssp_space.ssp_dim,invert_b=True))
        cleanup_nodes.append(nengo.Node(clean_up_fun, size_in = ssp_space.ssp_dim))
        cleanup_ens.append(nengo.Ensemble(n_neurons, ssp_space.ssp_dim))
        
        nengo.Connection(memory.output, circonvs[-1].input_a, synapse=tau)
        nengo.Connection(items[-1].output, circonvs[-1].input_b)
        nengo.Connection(circonvs[-1].output, cleanup_nodes[-1], synapse=tau)
        nengo.Connection(cleanup_nodes[-1], cleanup_ens[-1], synapse=None)
        
        querylocs_ps.append(nengo.Probe(circonvs[-1].output, synapse=tau))
        querycleanlocs_ps.append(nengo.Probe(cleanup_ens[-1], synapse=tau))
        
        # part 2
        circonvs2.append(nengo.networks.CircularConvolution(n_neurons, ssp_space.ssp_dim,invert_a=True))
        nengo.Connection(pathintegrator.output, circonvs2[-1].input_a)
        nengo.Connection(cleanup_nodes[-1], circonvs2[-1].input_b)
        
        queryvec_ps.append(nengo.Probe(circonvs2[-1].output, synapse=tau))
        
    ssp_p  = nengo.Probe(pathintegrator.output, synapse=None)
    mem_p = nengo.Probe(memory.output)
    
    
sim = nengo.Simulator(model)
sim.run(T)

import time
timestr = 'pathint_mem_' + time.strftime("%Y%m%d-%H%M%S") + '.npz'
np.savez(timestr, sim_ssp = sim.data[ssp_p], sim_mem = sim.data[mem_p], sim_querylocs=[sim.data[p] for p in querylocs_ps] ,
         sim_queryvec=[sim.data[p] for p in queryvec_ps], 
         sim_querycleanlocs=[sim.data[p] for p in querycleanlocs_ps],
         path=path, ts= sim.trange(),
         ssp_space=ssp_space)

true_locs = []
true_locs.append(ssp_space.encode(path[int(5/dt),:]))
true_locs.append(ssp_space.encode(path[int(10/dt),:]))
true_locs.append(ssp_space.encode(path[int(15/dt),:]))

fig = plt.figure(figsize=(6.9/2,6.9/3))
ax = fig.add_subplot(111)
p = []
simss = []
for j in range(len(item_names)):
    simseries = np.sum(sim.data[querylocs_ps[j]] * true_locs[j].reshape(1,-1), axis=1)
    p.append(ax.plot(sim.trange(), simseries, label=item_names[j].lower(),linewidth=1,alpha=0.8))
    
ax.set_xlabel('Time (s)', fontsize=8)
ax.set_ylabel("Similarity", fontsize=8)
ax.set_xlim(0,T)
ylim = ax.get_ylim()
symbols = ["o","s","^"]
for j in range(len(item_names)):
    pt = ax.plot(5*(j+1),ylim[0],marker=symbols[j],c=p[j][0].get_color(), zorder=10,markeredgewidth=0.5,markersize=10)[0]
    pt.set_clip_on(False)
ax.set_ylim(ylim[0],1.8)
fig.tight_layout()
fig.savefig('memory_sims.pdf', format='pdf', bbox_inches='tight')



fig = plt.figure(figsize=(6.9/2,6.9/3))
ax = fig.add_subplot(111)
true_vecs = []
true_vecs.append(ssp_space.encode((path[int(5/dt),:] - path).T))
true_vecs.append(ssp_space.encode((path[int(10/dt),:] - path).T))
true_vecs.append(ssp_space.encode((path[int(15/dt),:] - path).T))
p = []
for j in range(len(item_names)):
    simseries = np.sum(sim.data[queryvec_ps[j]] * true_vecs[j].T, axis=1)
    p.append(ax.plot(sim.trange(), simseries, label=item_names[j].lower(),linewidth=1,alpha=0.8))
ax.set_xlabel('Time (s)', fontsize=8)
ax.set_ylabel("Similarity", fontsize=8)
ax.set_xlim(0,T)
ylim = ax.get_ylim()
symbols = ["o","s","^"]
for j in range(len(item_names)):
    pt = ax.plot(5*(j+1),ylim[0],marker=symbols[j],c=p[j][0].get_color(), zorder=10,markeredgewidth=0.5,markersize=10)[0]
    pt.set_clip_on(False)
ax.set_ylim(ylim[0],1.2)
fig.tight_layout()
fig.savefig('memory_vec_sims.pdf', format='pdf', bbox_inches='tight')


sample_ssps, sample_points = ssp_space.get_sample_ssps(10000)
fig=plt.figure(figsize=(6.9/2,6.9/3))
ax = fig.add_subplot(111)
ax.plot(path[:,0],path[:,1],c='k')
for j in range(len(item_names)):
    ax.plot(path[int(5*(j+1)/dt),0],path[int(5*(j+1)/dt),1],marker=symbols[j],c=p[j][0].get_color(), zorder=10,markeredgewidth=0.5,markersize=10)
    sims = sample_ssps.T @ sim.data[querylocs_ps[j]][-1,:]
    closestloc = sample_points[:,np.argmax(sims, axis=0)]
    ax.plot(closestloc[0],closestloc[1],marker='X',c=p[j][0].get_color(), zorder=11,markersize=5,markeredgewidth=0.8)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
fig.tight_layout()
fig.savefig('memory_final_result.pdf', format='pdf', bbox_inches='tight')

