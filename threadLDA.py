import threading
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import psi

def rho(tau, kappa, t):
	return pow(tau + t, - kappa)

def digamma(mat):
	if (len(mat.shape) == 1):
		return(psi(mat) - psi(np.sum(mat)))
	else:
		return(psi(mat) - psi(np.sum(mat, 0))[np.newaxis, :])

def worker_estep(tid,batch,gamma,temp_list,ExpELogBeta,dtm):
	global alpha
	
	temp_topics_thread = np.zeros(ExpELogBeta.shape)

	for d in batch:

		ids = np.nonzero(dtm[d, :])[0]
		cts = dtm[d, ids]
		ExpELogBetad = ExpELogBeta[ids, :]

		gammad = gamma[d, :]
		ElogTethad = digamma(gammad)
		ExpLogTethad = np.exp(ElogTethad)

		for inner_it in range(1000):

			oldgammad = gammad

			phi = ExpLogTethad * ExpELogBetad
			phi = phi / (phi.sum(axis=1)+0.00001)[:, np.newaxis]

			gammad = alpha + np.dot(cts, phi)

			ElogTethad = digamma(gammad)
			ExpLogTethad = np.exp(ElogTethad)

			if np.mean((gammad-oldgammad)**2) < 0.0000001:
				break

		gamma[d, :] = gammad

		temp_topics_thread[ids, :] += phi * cts[:, np.newaxis]

	temp_list[tid] = temp_topics_thread

 
def lda_batch(dtm, ntopic, batch_size, tau, kappa,nthreads):
	global alpha

	nvoc = dtm.shape[1]
	ndoc = dtm.shape[0]
	nu = 1./ntopic
	alpha = 1./ntopic

	topics = np.random.gamma(100., 1./100., (nvoc, ntopic))
	gamma = np.random.gamma(100., 1./100., (ndoc, ntopic))

	numbatch = ndoc / batch_size
	batches = np.array_split(range(ndoc), numbatch)	

	for it_batch in range(numbatch):
		ELogBeta = digamma(topics)
		ExpELogBeta = np.exp(ELogBeta)

		temp_list = [0]*nthreads
		threads = range(nthreads)
		
		thread_batch = np.array_split(batches[it_batch], nthreads)
		
		for tid in range(nthreads):
			threads[tid] = threading.Thread(target=worker_estep,args=(tid,thread_batch[tid],gamma,temp_list,ExpELogBeta,dtm))  
			threads[tid].start()

		for thread in threads:
			thread.join()

		indices = np.unique(np.nonzero(dtm[batches[it_batch],:])[1])

		temp_topics = np.sum(temp_list,0)

		rt = rho(tau, kappa, it_batch)

		topics[indices] = (1 - rt) * topics[indices, :] + rt * \
			ndoc * (nu + temp_topics[indices, :]) / len(batches[it_batch])

	return topics, gamma