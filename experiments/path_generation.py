import numpy as np
import math

def generate_signal(T,dt,dims = 1, rms=0.5,limit=10, seed=1):
    np.random.seed(seed)             
    N = int(T/dt)
    dw = 2*np.pi/T
    
    # Don't get samples for outside limit, those coeffs will stay zero
    num_samples = max(1,min(N//2, int(2*np.pi*limit/dw)))
    
    x_freq = np.zeros((N,dims), dtype=complex)
    x_freq[0,:] = np.random.randn(dims) #zero-frequency coeffient
    x_freq[1:num_samples+1,:] = np.random.randn(num_samples,dims) + 1j*np.random.randn(num_samples,dims) #postive-frequency coeffients
    x_freq[-num_samples:,:] += np.flip(x_freq[1:num_samples+1,:].conjugate(),axis=0)  #negative-frequency coeffients
      
    x_time = np.fft.ifft(x_freq,n=N,axis=0)
    x_time = x_time.real # it is real, but in case of numerical error, make sure
    rescale = rms/np.sqrt(dt*np.sum(x_time**2)/T)
    x_time = rescale*x_time
    x_freq = rescale*x_freq
    
    x_freq = np.fft.fftshift(x_freq)    
    return(x_time,x_freq)



def spiral_points(arc=1, separation=1):
    """generate points on an Archimedes' spiral
    with `arc` giving the length of arc between two points
    and `separation` giving the distance between consecutive 
    turnings
    - approximate arc length with circle arc at given distance
    - use a spiral equation r = b * phi
    """
    def p2c(r, phi):
        """polar to cartesian
        """
        return (r * math.cos(phi), r * math.sin(phi))

    # yield a point at origin
    yield (0, 0)

    # initialize the next point in the required distance
    r = arc
    b = separation / (2 * math.pi)
    # find the first phi to satisfy distance of `arc` to the second point
    phi = float(r) / b
    while True:
        yield p2c(r, phi)
        # advance the variables
        # calculate phi that will give desired arc length at current radius
        # (approximating with circle)
        phi += float(arc) / r
        r = b * phi + 0.000001*math.sin(phi**2/5)
        
def spiral_path(ntimesteps, radius):
    spiral_path = np.zeros((ntimesteps +10,2))
    i=0
    for res in spiral_points(1e-8,5e-6):
        spiral_path[i,:] = res
        i += 1
        if i >= ( ntimesteps + 10):
            break
    spiral_path = spiral_path[10:,:]
    
    path = spiral_path
    path = radius*path/np.max(np.abs(path))
    return path


def real_path(idx,radius):
    onlinedata = np.load('multiple_sources_data_array.npy',allow_pickle=True)
    onlinedatatimes = np.load('multiple_sources_timestamps.npy',allow_pickle=True)
    path = radius*(onlinedata[idx][:,:2] - onlinedata[idx][0,:2])
    timesteps = onlinedatatimes[idx]
    dt = timesteps[1:] - timesteps[:-1]
    vels = (1/dt.reshape(-1,1))*( path[1:,:] - path[:-1,:])
    return path, timesteps, vels
    