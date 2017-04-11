"""
Description
"""

import numpy as np 
import matplotlib.pyplot as plt 
from sys import exit
from scipy import integrate
from scipy.interpolate import interp1d, interp2d
from CosmoPowerLib import CosmoPowerLib as CPL
from scipy import integrate

#------------------------------------------------------------------------------

class WeakLensingLib(object):
	'''
	Most of the equations here are taken from Takada and Jain 2009 paper.
	Link: http://arxiv.org/pdf/0810.4170v2.pdf
	'''

	def __init__(self, NumberOfBins=1, z0=1.0/3.0, \
					zmin=0.1, zmax=5.0, zdim=100, \
					kmin = 0.001, kmax=10, kdim=1000, \
					lmin = 10, lmax=10000, ldim=40, \
					CosmoParams=[0.3,0.8,0.7,0.96,0.046,-1.0,0.0,0.0,0.0]):
		"""
		Doc string of the constructor
		"""

		self.CosmoParams = CosmoParams
		self.set_cosmology()
		self.kmin = kmin
		self.kmax = kmax
		self.kdim = kdim
		self.lmin = lmin
		self.lmax = lmax
		self.ldim = ldim

		# Constants
		#----------
		self.SpeedOfLight = 299792.458 

		# Making Functions for distance redshift relations
		#-------------------------------------------------
		self.zArray = np.linspace(zmin, zmax, zdim)
		self.kiArray = np.zeros((len(self.zArray)))
		for i in range(len(self.zArray)):
			self.kiArray[i] = self.ComovingDistance(self.zArray[i])
		self.Func_z2ki = interp1d(self.zArray, self.kiArray)
		self.Func_ki2z = interp1d(self.kiArray, self.zArray)

		# Setting up Number of Bins and Bin edges
		#----------------------------------------
		self.NumberOfBins = NumberOfBins
		self.z0 = z0
		self.binedges_z = self.MakeBins(plot=False)
		self.binedges_ki = self.Func_z2ki(self.binedges_z)
		self.nsources_bins = self.n_i_bins()

		# Lensing Weights Matrix
		#-----------------------
		self.qMatrix = np.zeros((len(self.zArray), self.NumberOfBins))
		self.Make_qMatrix(plot=False)

#------------------------------------------------------------------------------

	def set_cosmology(self):
		# Setting up cosmology
		#---------------------
		[self.Omega_m, self.Sigma_8, self.h, self.n_s, self.Omega_b, \
			self.w0, self.wa, self.Omega_r, self.Omega_k] = self.CosmoParams
		self.Omega_l = 1.0 - self.Omega_m - self.Omega_r - self.Omega_k

#------------------------------------------------------------------------------

	def Func_pkmatrix(self, zz, kk):
		return self.Func_pkmatrix_interp(zz, kk)

#------------------------------------------------------------------------------

	def load_pk(self, mode='linear', plot=False):
		# Power Spectrum (Default: Linear)
		#--------------------
		self.kArray = 10**np.linspace(np.log10(self.kmin), \
							np.log10(self.kmax), self.kdim)
		if mode=='linear':
			self.PKmatrix = self.PK_linear(self.kArray, self.zArray)
		elif mode=='nonlinear':
			self.PKmatrix = self.PK_nonlinear(self.kArray, self.zArray)
		else:
			print "Current supported modes are: linear, nonlinear"
			exit()
		self.Func_pkmatrix_interp = interp2d(self.zArray, \
									self.kArray, self.PKmatrix)
		if plot:
			plt.figure(figsize=(10,6))
			for i in range(len(self.zArray)):
				plt.loglog(self.kArray, self.PKmatrix[:,i], 'k', lw=0.5)
			plt.xlabel('$\mathtt{k\ [h/Mpc]}$', fontsize=22)
			plt.ylabel('$\mathtt{P(k)\ [Mpc/h]^3}$', fontsize=22)
			plt.xlim(min(self.kArray), max(self.kArray))
			plt.show()

#------------------------------------------------------------------------------

	def get_pk(self):
		return self.kArray, self.zArray, self.PKmatrix

#------------------------------------------------------------------------------

	def PK_linear(self, kk, zz):
		CPLo = CPL(self.CosmoParams, True)
		return CPLo.PKL_Camb_MultipleRedshift(kk, zz)

	def PK_nonlinear(self, kk, zz):
		CPLo = CPL(self.CosmoParams, True)
		return CPLo.PKNL_CAMB_MultipleRedshift(kk, zz)

#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------

	def ComovingDistance(self, z=1.0):
		# Returns Comoving distance in Units Mpc/h
		func = lambda zz: 1.0/self.Ez(zz)
		ki = integrate.romberg(func, 0.0, z)
		return ki * self.SpeedOfLight / (100.0)

#------------------------------------------------------------------------------

	def AngularDiameterDistance(self, z=1.0):
		# Returns Angular Diameter distance in Units Mpc/h
		return self.ComovingDistance(z)/(1.0+z)

#------------------------------------------------------------------------------

	def LuminosityDistance(self, z=1.0):
		# Returns Luminosity distance in Units Mpc/h
		return self.ComovingDistance(z) * (1.0+z)

#------------------------------------------------------------------------------

	def p_s(self, z):
		"""
		Refer to paper: https://arxiv.org/pdf/0810.4170.pdf; eqn 20; Sec 4.1
		z_m = 0.7, 1.0, 1.2, 1.5
		z0 = z_m / 3
		n_g = 10, 30 51, 100 arcmin^(-2)
		survey: DES, Subaru, LSST, SNAP
		"""
		return 1.18e9 * 4.0 * z**2 * np.exp(-z/self.z0)

#------------------------------------------------------------------------------

	def n_i(self, z1=0.001, z2=20.0):
		"""
		Refer to paper: https://arxiv.org/pdf/0810.4170.pdf; eqn 20; Sec 4.1
		z_m = 0.7, 1.0, 1.2, 1.5
		z0 = z_m / 3
		n_g = 10, 30 51, 100 arcmin^(-2)
		survey: DES, Subaru, LSST, SNAP
		"""
		result = integrate.quad(self.p_s, z1, z2, limit=500)[0]
		return result

#------------------------------------------------------------------------------

	def n_i_bins(self):
		nsources_bins = np.zeros((self.NumberOfBins))
		for i in range(self.NumberOfBins):
			nsources_bins[i] = self.n_i(self.binedges_z[i], \
								self.binedges_z[i+1])
		return nsources_bins

#------------------------------------------------------------------------------

	def MakeBins(self, plot=False):
		zArray = np.linspace(min(self.zArray)*2, max(self.zArray), 100)
		nArray = np.zeros((len(zArray)))
		for i in range(len(zArray)):
			nArray[i] = self.n_i(min(self.zArray), zArray[i])
		nArray /= max(nArray)
		binedges = [min(self.zArray)]
		for i in range(self.NumberOfBins - 1):
			binedges.append(np.interp(float(i+1)/self.NumberOfBins, \
				nArray, zArray))
		binedges.append(max(self.zArray))
		if plot:
			plt.figure(figsize=(10,6))
			plt.plot(self.zArray,self.p_s(self.zArray), 'k', lw=2)
			for i in range(1, len(binedges)-1):
				plt.axvline(x=binedges[i], color='b', ls='--', lw=1)
			plt.xlim(0, max(self.zArray))
			plt.xlabel('$\mathtt{Redshift}$', fontsize=22)
			plt.ylabel('$\mathtt{DistributionOfSources}$', fontsize=22)
			plt.tight_layout()
			plt.show()
		return binedges

#------------------------------------------------------------------------------

	def _q(self, zs, z):
		return self.p_s(zs) * \
				(self.Func_z2ki(zs) - self.Func_z2ki(z)) / \
	    		self.Func_z2ki(zs)

	def q(self, chi, chi1, chi2):
		if chi>chi2:
			return 1e-35
		else:		    
			zz = self.Func_ki2z(chi)
			z1 = self.Func_ki2z(chi1)
			z2 = self.Func_ki2z(chi2)
			result = integrate.quad(self._q, max(zz, z1), z2, \
									args=tuple([zz]), limit=500)[0]
			return result * 1.5 * 1e4 * self.Omega_m / \
		    		self.SpeedOfLight**2 * chi * (1.0+zz) /self.n_i(z1, z2)

#------------------------------------------------------------------------------

	def Make_qMatrix(self, plot=False):
		if plot:
			plt.figure(figsize=(10,6))
			plt.axvline(x=self.binedges_z[0], color='k', ls=':', lw=0.5)


		for i in range(self.NumberOfBins):
			for j in range(len(self.zArray)):
				self.qMatrix[j,i] = self.q(self.kiArray[j], \
					self.binedges_ki[i], self.binedges_ki[i+1])
			if plot:
				plt.plot(self.zArray, self.qMatrix[:,i], lw=2, \
								label='$\mathtt{Bin:\ %i}$'%(i+1))
				plt.axvline(x=self.binedges_z[i+1], color='k', ls=':', lw=0.5)

		if plot:
			plt.legend(loc=1, fontsize=18)
			plt.xlim(0, max(self.zArray))
			plt.xlabel('$\mathtt{Redshift}$', fontsize=22)
			plt.ylabel('$\mathtt{q(z)\ LensingKernel}$', fontsize=22)
			plt.tight_layout()			
			plt.show()

#------------------------------------------------------------------------------

	def _Cell(self, chi, ell, bin1, bin2):
		zz = self.Func_ki2z(chi)
		integrand = np.interp(zz, self.zArray, \
						(self.qMatrix[:, bin1])) * \
					np.interp(zz, self.zArray, \
						(self.qMatrix[:, bin2])) * \
					self.Func_pkmatrix(zz, ell/chi) / chi**2				
		return integrand

#------------------------------------------------------------------------------

	def Cell(self, ell, bin1, bin2):
		chimin = min(self.kiArray)
		chimax = max(self.kiArray)
		result=0.0
		for i in range(len(self.kiArray)-1):
			result += self._Cell(self.kiArray[i], ell, bin1, bin2) * \
							(self.kiArray[i+1] - self.kiArray[i])
		return result

#------------------------------------------------------------------------------

	def CellVector(self, ell, bin1, bin2):
		cc = np.zeros((len(ell)))
		for i in range(len(ell)):
			cc[i] = self.Cell(ell[i], bin1, bin2)
		return cc

#------------------------------------------------------------------------------

	def CellMatrix(self, ell, mode='linear', plot=False):
		self.load_pk(mode=mode, plot=plot)
		CellArray = np.zeros((self.NumberOfBins, self.NumberOfBins, len(ell)))
		for i in range(self.NumberOfBins):
			for j in range(i, self.NumberOfBins):
					CellArray[i,j,:] = self.CellVector(ell, i, j)
					CellArray[j,i,:] = CellArray[i,j,:]
					if plot:
						if i==j:
							ls = '-'
						else:
							ls='--'
						plt.loglog(ell, \
							CellArray[i,j,:] * ell * \
										(ell+1)/2.0/np.pi, \
							ls=ls, label='%i, %i'%(i,j))
		if plot:
			plt.legend(loc=2, fontsize=14)
			plt.xlim(min(ell), max(ell))
			plt.xlabel('$\mathtt{\ell}$', fontsize=22)
			plt.ylabel('$\mathtt{C_{\ell}}$', fontsize=22)
			plt.show()
		return CellArray

#------------------------------------------------------------------------------

	def compute_cell(self, mode='linear', plot=False):
		self.ellArray = 10**np.linspace(np.log10(self.lmin), \
								np.log10(self.lmax), self.ldim)
		self.CellArray = self.CellMatrix(self.ellArray, mode=mode, plot=plot)
		return self.ellArray ,self.CellArray

#==============================================================================

if __name__=="__main__":
	co = WeakLensingLib(NumberOfBins=1)
	# ell = 10**np.linspace(1, 5, 40)
	# cell = co.CellMatrix(ell, mode='linear', plot=True)
	ell, cell = co.compute_cell('nonlinear', True)
