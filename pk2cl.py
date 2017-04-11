import numpy as np 
import matplotlib.pyplot as plt 
from sys import exit, argv
from WeakLensingLib import WeakLensingLib as WL
from scipy.interpolate import interp2d
from glob import glob

#==============================================================================

class pk2cl(WL):
	"""
	Main doc string 
	"""
	def __init__(self, nbins_z, nn):
		"""
		Constructor doc string
		"""
		self.pkfunc, self.nsamples = self.load_data(nn)

		WL.__init__(self, \
					CosmoParams=[0.3,0.8,0.7,0.96,0.046,-1.0,0.0,0.0,0.0],\
					NumberOfBins=nbins_z, z0=0.5)

		print "Class is initialised"

#------------------------------------------------------------------------------

	def Func_pkmatrix(self, zz, kk):
		return self.pkfunc(zz, kk)

#------------------------------------------------------------------------------

	def load_data(self, nn):
		PK_DIR = '/Users/mohammed/Dropbox/fermilabwork/with_gnedin/sim1/PK/'
		filenames = glob(PK_DIR+'*.pk')
		# filenames = self.choose_files(filenames, n_pk)
		pk = []
		k = []
		N = []
		z = []

		nsamples = 0
		for i in range((len(filenames)-1)%nn, len(filenames), nn):
			# print i, filenames[i]
			nsamples += 1
			name = filenames[i].replace(PK_DIR, '')
			name = name.replace('matter_power_a=', '')
			name = float(name.replace('.pk', ''))
			z.append(1.0/name - 1)
			data = np.genfromtxt(filenames[i], skip_header=2)
			# data_temp = np.genfromtxt(filenames[-1], skip_header=2)
			pk.append(data[:,2])
			# pk.append(data_temp[:,2])
			k.append(data[:,1])
			N.append(data[:,3])

		k = np.mean(np.array(k), axis=0)
		N = np.mean(np.array(N), axis=0)
		pk = np.array(pk)
		pk = np.transpose(pk)
		z = np.array(z)
		z = z[::-1]
		pkfunc = interp2d(z, k, pk)
		print "Data loaded"
		# exit()
		return pkfunc, nsamples


	# def choose_files(self, filenames, nn):
	# 	newfiles = list(np.random.choice(np.array(filenames), nn, False))
	# 	newfiles.sort()
	# 	return newfiles

#==============================================================================

if __name__=="__main__":
	obj = pk2cl(1, nn=1)
