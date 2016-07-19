import numpy as np 
import matplotlib.pyplot as plt 
from sys import exit
from scipy import integrate
from scipy.interpolate import interp1d, interp2d
from CosmoPowerLib import CosmoPowerLib as CPL


#-----------------------------------------------------------------------------------------

class WeakLensingLib(object):
	'''
	Most of the equations here are taken from Takada and Jain 2009 paper.
	Link: http://arxiv.org/pdf/0810.4170v2.pdf
	'''

	def __init__(self, CosmoParams=[0.3, 0.8, 0.7, 0.96, 0.046, -1.0, 0.0, 0.0, 0.0], \
				NumberOfBins=5, z0=0.5):
		# Setting up cosmology
		#---------------------
		self.CosmoParams = CosmoParams
		[self.Omega_m, self.Sigma_8, self.h, self.n_s, self.Omega_b, \
				self.w0, self.wa, self.Omega_r, self.Omega_k] = self.CosmoParams
		self.Omega_l = 1.0 - self.Omega_m - self.Omega_r - self.Omega_k

		# Constants
		#----------
		self.SpeedOfLight = 299792.458 
		self.GravitationConstant = 6.6740831e-11
		self.CriticalDensity = 2.778e11

		# Making Functions for distance redshift relations
		#-------------------------------------------------
		self.zArray = np.linspace(0.001, 5.0, 100)
		self.kiArray = np.zeros((len(self.zArray)))
		qArray = np.zeros((len(self.zArray)))
		for i in range(len(self.zArray)):
			self.kiArray[i] = self.ComovingDistance(self.zArray[i])
		self.Func_z2ki = interp1d(self.zArray, self.kiArray)
		self.Func_ki2z = interp1d(self.kiArray, self.zArray)

		# Power Spectrum (Default: Linear)
		#--------------------
		self.kArray = 10**np.linspace(-3, 1, 10000)
		self.PKmatrix = self.PK_linear(self.kArray, self.zArray)
		self.Func_pkmatrix_interp = interp2d(self.zArray, self.kArray, self.PKmatrix)

		# Setting up Number of Bins and Bin edges
		#----------------------------------------
		self.NumberOfBins = NumberOfBins
		self.z0 = z0
		self.binedges_z = self.MakeBins(True)
		self.binedges_ki = self.Func_z2ki(self.binedges_z)

		# Lensing Weights Matrix
		#-----------------------
		self.qMatrix = np.zeros((len(self.zArray), self.NumberOfBins))
		self.Make_qMatrix(True)

		# Computing C_ell
		#----------------
		self.ellArray = 10**np.linspace(np.log10(50.0), np.log10(4e3), 40)
		self.CellArray = []
		for i in range(NumberOfBins):
			for j in range(i, NumberOfBins):
					xx = self.CellVector(self.ellArray, i, j)
					self.CellArray.append(xx)
					print i,j
		self.CellArray = np.transpose(np.array(self.CellArray))
		# self.CellArray = np.reshape(self.CellArray, (len(self.ellArray), \
								# NumberOfBins*(NumberOfBins+1)/2))
		for i in range(NumberOfBins*(NumberOfBins+1)/2):
			plt.loglog(self.ellArray, self.CellArray[:, i])
		plt.show()
		print np.shape(self.CellArray)

#-----------------------------------------------------------------------------------------

	def Func_pkmatrix(self, zz, kk):
		# if min(self.kArray)<kk<max(self.kArray):
			return self.Func_pkmatrix_interp(zz, kk)
		# else: 
			# return 0.0

#-----------------------------------------------------------------------------------------

	def Ez(self, z=0.0):
		if z==0.0:
			return 1.0
		else:
			a = 1.0 / (1.0 + z)
			return (self.Omega_m/a**3 + \
						self.Omega_k/a**2 + \
						self.Omega_r/a**4 +\
						self.Omega_l/a**(3.0*(1.0+self.w0+self.wa))/\
						np.exp(3.0*self.wa*(1.0-a)))**0.5

#-----------------------------------------------------------------------------------------

	def ComovingDistance(self, z=1.0):
		# Returns Comoving distance in Units Mpc/h
		func = lambda zz: 1.0/self.Ez(zz)
		ki = integrate.romberg(func, 0.0, z)
		return ki * self.SpeedOfLight / (100.0)

#-----------------------------------------------------------------------------------------

	def AngularDiameterDistance(self, z=1.0):
		# Returns Angular Diameter distance in Units Mpc/h
		return self.ComovingDistance(z)/(1.0+z)

#-----------------------------------------------------------------------------------------

	def LuminosityDistance(self, z=1.0):
		# Returns Luminosity distance in Units Mpc/h
		return self.ComovingDistance(z) * (1.0+z)

#-----------------------------------------------------------------------------------------

	def p_s(self, z):
		return 1.18e9 * 4.0 * z**2 * np.exp(-z/self.z0)

#-----------------------------------------------------------------------------------------

	def n_i(self, z1=0.001, z2=20.0):
		result = integrate.quad(self.p_s, z1, z2, limit=200)[0]
		return result

#-----------------------------------------------------------------------------------------

	def MakeBins(self, plot=False):
		zArray = np.linspace(0.1, 5.0, 100)
		nArray = np.zeros((len(zArray)))
		for i in range(len(zArray)):
			nArray[i] = self.n_i(0.001, zArray[i])
		nArray /= max(nArray)
		binedges = [0.001]
		for i in range(self.NumberOfBins - 1):
			binedges.append(np.interp(float(i+1)/self.NumberOfBins, nArray, zArray))
		binedges.append(5.0)
		if plot:
			plt.plot(zArray,self.p_s(zArray), 'k', lw=2)
			for i in range(1, len(binedges)-1):
				plt.axvline(x=binedges[i], color='b', ls='--', lw=1)
				plt.xlim(xmax=5.0)
				plt.xlabel('$\mathtt{Redshift}$', fontsize=22)
				plt.ylabel('$\mathtt{DistributionOfSources}$', fontsize=22)
			plt.show()
		return binedges

#-----------------------------------------------------------------------------------------

	def _q(self, zs, z):
	    return self.p_s(zs) * (self.Func_z2ki(zs) - self.Func_z2ki(z)) / self.Func_z2ki(zs)

	def q(self, chi, chi1, chi2):
		if chi>chi2:
			return 1e-35
		else:
		    from scipy import integrate
		    zz = self.Func_ki2z(chi)
		    z1 = self.Func_ki2z(chi1)
		    z2 = self.Func_ki2z(chi2)
		    result = integrate.quad(self._q, max(zz, z1), z2, args=tuple([zz]), limit=200)[0]
		    return result * 1.5 * self.h * 1e4 / 3e5**2 * chi * (1.0+zz) /self.n_i(z1, z2)

#-----------------------------------------------------------------------------------------

	def Make_qMatrix(self, plot=False):
		for i in range(self.NumberOfBins):
			for j in range(len(self.zArray)):
				self.qMatrix[j,i] = self.q(self.kiArray[j], \
					self.binedges_ki[i], self.binedges_ki[i+1])
			if plot:
				plt.plot(self.zArray, self.qMatrix[:,i])
		if plot:
			plt.show()

#-----------------------------------------------------------------------------------------

	def PK_linear(self, kk, zz):
		CPLo = CPL(self.CosmoParams, True)
		return CPLo.PKL_Camb_MultipleRedshift(kk, zz)

#-----------------------------------------------------------------------------------------

	def _Cell(self, chi, ell, bin1, bin2):
		zz = self.Func_ki2z(chi)
		integrand = 10**np.interp(zz, self.zArray, np.log10(self.qMatrix[:, bin1])) * \
					10**np.interp(zz, self.zArray, np.log10(self.qMatrix[:, bin2])) * \
					self.Func_pkmatrix(zz, ell/chi) / chi**2
		return integrand

	def Cell(self, ell, bin1, bin2):
		chimin = max(ell/10.0, min(self.kiArray))
		chimax = min(ell/0.001, max(self.kiArray))
		result = integrate.romberg(self._Cell, chimin, chimax, \
					args=tuple([ell, bin1, bin2]), divmax=20)
		return result

	def CellVector(self, ell, bin1, bin2):
		cc = np.zeros((len(ell)))
		for i in range(len(ell)):
			cc[i] = self.Cell(ell[i], bin1, bin2)
		return cc

#-----------------------------------------------------------------------------------------

#===============================================================

if __name__=="__main__":
	co = WeakLensingLib(NumberOfBins=2)
	# print co.q(1200.0, 1000., 1500.0)

	# ell = np.linspace(50, 5000, 40)
	# # cell = co.CellVector(ell, 0,1)
	# plt.loglog(ell, co.CellVector(ell, 0,0), 'r')
	# plt.loglog(ell, co.CellVector(ell, 0,1), 'b')
	# plt.loglog(ell, co.CellVector(ell, 1,1), 'g')
	# plt.show()




