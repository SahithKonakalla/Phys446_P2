import statistics
import numpy as np

def Stats(data):
    n = len(data)
    variance = data.var()
    mean = data.mean()
    error = statistics.stdev(data)/np.sqrt(n)

    data = data-mean
    r = np.correlate(data, data, mode = 'full')[-n:]
    assert np.allclose(r, np.array([(data[:n-k]*data[-(n-k):]).sum() for k in range(n)]))
    autocorrelation = r/(variance*(np.arange(n, 0, -1)))

    return (mean, variance, error, autocorrelation)
    
