import nengo
import numpy as np
import matplotlib.pyplot as plt
import neurosymPI

import pytry
from argparse import ArgumentParser
import os
import os.path
import random
import pickle

from path_generation import generate_signal

class PITrial(pytry.Trial):
    def params(self):
        self.param('ssp_dim', ssp_dim=151)
        self.param('domain_dim', domain_dim=2)
        self.param('sim time', time=60)
        self.param('path limit', limit=0.08)
        self.param('n_neurons', n_neurons=800)
    
    def evaluate(self, p):
        domain_dim = p.domain_dim
        bounds = 15*np.tile([-1,1],(domain_dim,1))
        ssp_space = neurosymPI.sspspace.HexagonalSSPSpace(domain_dim,ssp_dim=p.ssp_dim, 
                         scale_min=2*np.pi/np.sqrt(6) - 0.5, scale_max=2*np.pi/np.sqrt(6) + 0.5,
                         domain_bounds=bounds, length_scale=1)
        d = ssp_space.ssp_dim
        
        T = p.time
        dt = 0.001
        timesteps = np.arange(0, T, dt)
        radius=12
        path,_ = generate_signal(T, dt, dims=domain_dim, rms=radius, limit=p.limit, seed=p.seed)
        path = (radius-1)*path/np.max(np.abs(path),axis=0)
        
        
        pathlen = path.shape[0]
        vels = (1/dt)*( path[(np.minimum(np.floor(timesteps/dt) + 1, pathlen-1)).astype(int),:] -
                       path[(np.minimum(np.floor(timesteps/dt), pathlen-2)).astype(int),:])
        real_freqs = (ssp_space.phase_matrix @ vels.T)
        scale_fac = 1/np.max(np.abs(real_freqs))
        vels_scaled = vels*scale_fac
        
        real_ssp = ssp_space.encode(path.T)
        real_fssp = np.fft.fftshift(ssp_space.encode_fourier(path.T),axes=0)
        stim_fssp = np.ones((d-1,len(timesteps)))
        stim_fssp[:d//2,:] = real_fssp.real[:d//2,:]
        stim_fssp[d//2:,:] =real_fssp.imag[:d//2,:]
        
        n_neurons = p.n_neurons
        tau = 0.05
        model = nengo.Network(seed=p.seed)
        with model:
            vel_input = nengo.Node(lambda t: vels_scaled[int(t/dt)-1,:], size_out=domain_dim)
            stim = nengo.Node(lambda t: real_ssp[:,int(t/dt)-1] if t<20*dt else np.zeros((d,)))
            pathintegrator = neurosymPI.networks.PathIntegration(ssp_space, tau, n_neurons, 
                          scaling_factor=scale_fac, stable=True)
            nengo.Connection(vel_input,pathintegrator.velocity_input)
            nengo.Connection(stim,pathintegrator.input)
            
            osc_p = nengo.Probe(pathintegrator.oscillators.output, synapse=tau)
            ssp_p  = nengo.Probe(pathintegrator.output, synapse=None)
            
        sim = nengo.Simulator(model)
        sim.run(T)
        
        sim_to_exact = np.sum(sim.data[ssp_p]*real_ssp.T, axis=1)
        sim_path, sim_to_closest = ssp_space.decode_path(sim.data[ssp_p], n_samples = 1000000)
        
        return dict(
             sim_ssp = sim.data[ssp_p],
             sim_osc = sim.data[osc_p],
             path=path,
             ts= sim.trange(),
             ssp_space=ssp_space,
             scale_fac=scale_fac,
             sim_to_exact = sim_to_exact,
             sim_path = sim_path,
             sim_to_closest=sim_to_closest
        )




if __name__=='__main__':
    parser = ArgumentParser()

    parser.add_argument('--ssp-dim', dest='ssp_dim', type=int, default=151)
    parser.add_argument('--domain-dim', dest='domain_dim', type=int, default=2)
    parser.add_argument('--n-neurons', dest='n_neurons', type=int, default=1000)
    parser.add_argument('--trial-time', dest='trial_time', type=float, default=60)
    parser.add_argument('--path-gen-param', dest='limit', type=float, default=0.08)
    parser.add_argument('--num-trials', dest='num_trials', type=int, default=10)
    parser.add_argument('--data-dir', dest='data_dir', type=str, default='/home/ns2dumon/Documents/cogsci2022-pathintegration/data_2d/')

    
    args = parser.parse_args()

    random.seed(1)
    seeds = [random.randint(1,100000) for _ in range(args.num_trials)]

    data_path = os.path.join(args.data_dir)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    for seed in seeds:
        params = {'n_neurons':args.n_neurons,
                  'data_format':'npz',
                  'data_dir':data_path,
                  'seed':seed, 
                  'ssp_dim':args.ssp_dim,
                  'domain_dim':args.domain_dim
                  }
        r = PITrial().run(**params)

