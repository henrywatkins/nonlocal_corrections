from scipy import integrate
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import cProfile
from scipy.optimize import curve_fit
from scipy.linalg import solve_triangular
plt.rc('font', family='serif')
plt.rc('text', usetex=False)

###############################
#
#	NONLOCAL CORRECTIONS
#	CALCULATION-FINDING
#	H_para AND H_perp AND H_wedg
#
###############################

#Switches
Fullm=True
calculate_G=False
visualise_G=False
comparison_plots=False
coef_plots=False
F0_term=False
SAVE_FIGS=False

filename1='gxx_A1.h5'
filename2='gxy_A1.h5'
filename3='gxx_fullm.h5'
filename4='gxy_fullm.h5'
filename5='hpara_conv_100.h5'

################
#
#	parameters
#
################
#find a simple functional form for the nonlocality term in a magnetic field
#should be a function of chi, eta (magnetization and nonlocality,respectively)

etalim=1000.0
chilim=40.0
vlim=5.0
vnum=100
v=np.linspace(1e-10,vlim,num=2*vnum)
th_ind=int(2*vnum/vlim)
#thermal 3D meshgrid #-2
etatherm=np.logspace(-3,np.log10(etalim),num=2*vnum)
chitherm=np.logspace(-3,np.log10(chilim),num=vnum)
#chitherm=np.log10(chitherm)
chitherm=np.append(0.0,chitherm)
#chitherm=np.linspace(0,chilim,num=vnum)

#dnchi=int(vnum/chilim)
#lineouts=[0,2*dnchi,5*dnchi,10*dnchi,15*dnchi,20*dnchi]
dnchilog=(np.log10(chilim)+3.0)/(vnum)
lineouts=[0,44,66,80,87,97] #pick out values where chi=0,0.1,1,10,30
chi_mesh, eta_mesh, v_mesh = np.meshgrid(chitherm,etatherm,v)
eta_curve=eta_mesh*v_mesh**4.0
chi_curve=chi_mesh*v_mesh**3.0

#m=1 case - calculate the g function matrix that provides the high mode corrections
#max l to start with
lmax=80

##################
#
#	H_perp + H_wedg
#
##################
#finding H_perp and H_wedge using m=1 approximation

print('Calculation of H_perp + H_wedg')

if calculate_G and not Fullm:

	ident=np.zeros((3,3,chi_mesh.shape[0],chi_mesh.shape[1],chi_mesh.shape[2]))
	ident[0,0]=np.ones(chi_mesh.shape)
	ident[1,1]=np.ones(chi_mesh.shape)
	ident[2,2]=np.ones(chi_mesh.shape)

	#initial gmat is the identity
	gmat=ident

	#constructing matricies
	A=np.zeros(gmat.shape)
	B=np.zeros(gmat.shape)
	newmat=np.zeros(gmat.shape)

	for l in range(lmax,0,-1):
		print(l)
		#calculate values of chi and eta
		etap=2.0*eta_curve/(l*(l+1))
		chip=2.0*chi_curve/(l*(l+1))
		etaperp=etap/(1+chip*chip*0.5*l*(l+1))
		#calculate A
		A[0,0]=etaperp*(l+1)/(2*l+3)
		A[0,1]=etaperp*0.5*l*(l+1)*(l+2)*chip/(2*l+3)
		A[1,0]=-etaperp*(l+1)*chip/(2*l+3)
		A[1,1]=etaperp*(l+2)/(2*l+3)
		A[2,2]=etap*(l+2)/(2*l+3)
		print('Calc A')
		l2=l+1
		etap=2.0*eta_curve/(l2*(l2+1))
		chip=2.0*chi_curve/(l2*(l2+1))
		etaperp=etap/(1+chip*chip*0.5*l2*(l2+1))
		#calculate B
		B[0,0]=etaperp*l2/(2*l2-1)
		B[0,1]=etaperp*0.5*l2*(l2+1)*(l2-1)*chip/(2*l2-1)
		B[1,0]=-etaperp*l2*chip/(2*l2-1)
		B[1,1]=etaperp*(l2-1)/(2*l2-1)
		B[2,2]=etap*(l2-1)/(2*l2-1)
		print('Calc B')

		#inverse gmat alternative
		gmat=gmat.transpose(2,3,4,0,1)
		A=A.transpose(2,3,4,0,1)
		B=B.transpose(2,3,4,0,1)
		gmat=np.linalg.inv(gmat)
		print('Inverse')
		newmat=np.matmul(A,gmat)
		newmat=np.matmul(newmat,B)
		newmat=newmat.transpose(3,4,0,1,2)
		A=A.transpose(3,4,0,1,2)
		B=B.transpose(3,4,0,1,2)

		#new gmat
		gmat=ident+newmat
		print('reset G')

#calculate using full m matrix
if calculate_G and Fullm:

	#constructing matricies
	D=[]
	N=[]
	P=[]
	newmat=[]
	idn=[]
	for i in range(len(chitherm)):

		chitherm2=np.array([chitherm[i]])
		chi_mesh, eta_mesh, v_mesh = np.meshgrid(chitherm2,etatherm,v)
		eta_curve=np.float32(eta_mesh*v_mesh**4.0)
		chi_curve=np.float32(chi_mesh*v_mesh**3.0)
		ones=np.float32(np.ones(eta_mesh.shape))
		ident=np.tensordot(ones,np.identity(lmax+2),axes=0)
		#initial gmat is the identity
		gmat=ident

		for l in range(lmax,0,-1):
			print('l=', l)
			print('Calculate ident')
			ident=np.tensordot(ones,np.identity(l+1),axes=0)
			idn=ident*l*(l+1)/2

			bufmat=np.float32(np.zeros((l+1,l+1)))

			print('Calculate P')
			val_array2=np.arange(l+1,2*l+2)/(2*l+3)
			np.fill_diagonal(bufmat,val_array2)
			P=np.tensordot(eta_curve,bufmat,axes=0)

			bufmat=np.float32(np.zeros((l+2,l+2)))

			print('Calculate N')
			val_array3=np.arange(l+1,-1,-1)/(2*(l+1)-1)
			np.fill_diagonal(bufmat,val_array3)
			N=np.tensordot(eta_curve,bufmat,axes=0)

			bufmat=np.float32(np.zeros((l+1,l+1)))

			print('Calculate D')
			val_array=-0.5*np.array([(l-i)*(l+i+1) for i in range(l+1)])
			np.fill_diagonal(bufmat[1:],0.5)
			np.fill_diagonal(bufmat[:,1:],val_array)
			D=np.tensordot(chi_curve,bufmat,axes=0)

			print('Invert G , set RHS matrix')
			#gmat=np.linalg.inv(gmat)
			#gmat=np.array([[[ solve_triangular(gmat[i,j,k],N[i,j,k]) for k in range(eta_mesh.shape[2])] for j in range(eta_mesh.shape[1])] for i in range(eta_mesh.shape[0])])
			gmat=np.linalg.solve(gmat,N)
			gmat=gmat[:,:,:,:l+1,:l+1]
			#newmat=np.linalg.solve(gmat,N)
			#gmat=gmat[:,:,:,:l+1,:l+1]
			newmat=np.matmul(P,gmat)
			#newmat=np.matmul(newmat,N)
			print('reset G')
			gmat=idn+D+newmat


		gxx=gmat[:,:,:,0,0]
		gxy=gmat[:,:,:,0,1]

		filename3='intermediatexx_'+str(i)+'.h5'
		filename4='intermediatexy_'+str(i)+'.h5'

		hf1=h5py.File(filename3,'w')
		hf2=h5py.File(filename4,'w')

		hf1.create_dataset('gxx', data=gxx)
		hf2.create_dataset('gxy', data=gxy)

		hf1.close()
		hf2.close()


##################
#
#	H_para
#
##################

v_para_mesh, eta_para_mesh =np.meshgrid(v,etatherm)
eta_curve_para=eta_para_mesh*v_para_mesh**4.0

#finding H_para, find a better fit than epperlein result
print('Calculation of H_para')

#for L_value in lvals:

H_para=np.ones(eta_para_mesh.shape)

#if calculate_G:

for l in range(lmax,0,-1):

	a=(l+1)/(2*l+3)
	b=(l+1)/(2*l+1)
	newval=a*b/H_para
		#new gmat
	H_para=0.5*l*(l+1)+(eta_curve_para**2)*newval

	if F0_term:
		H_para=H_para+(eta_curve_para**2)/3

###################
#
#	extract gxx, gxy and define H functions
#
###################


if not Fullm and calculate_G:
	##Save gxx, gxy data to hdf5 from m=1 calculation

	gxx=gmat[0,0]
	gxy=gmat[0,1]

	hf1=h5py.File(filename1,'w')
	hf2=h5py.File(filename2,'w')

	hf1.create_dataset('gxx', data=gxx)
	hf2.create_dataset('gxy', data=gxy)

	hf1.close()
	hf2.close()
elif Fullm and calculate_G:
	##Save gxx, gxy data to hdf5 from full m calculation

	gxx=gmat[:,:,:,0,0]
	gxy=gmat[:,:,:,0,1]

	hf1=h5py.File(filename3,'w')
	hf2=h5py.File(filename4,'w')

	hf1.create_dataset('gxx', data=gxx)
	hf2.create_dataset('gxy', data=gxy)

	hf1.close()
	hf2.close()
elif not Fullm and not calculate_G:
	##open hdf5 file containing gxx, gxy data from m=1 calculation

	hf1 = h5py.File(filename1, 'r')
	hf2 = h5py.File(filename2, 'r')

	hf1.keys()
	hf2.keys()

	buf = hf1.get('gxx')
	gxx = np.array(buf)
	buf = hf2.get('gxy')
	gxy = np.array(buf)

	hf1.close()
	hf2.close()
elif Fullm and not calculate_G:
	##open hdf5 file containing gxx, gxy data from full m calculation

	hf1 = h5py.File(filename3, 'r')
	hf2 = h5py.File(filename4, 'r')

	hf1.keys()
	hf2.keys()

	buf = hf1.get('gxx')
	gxx = np.array(buf)
	buf = hf2.get('gxy')
	gxy = np.array(buf)

	hf1.close()
	hf2.close()
else:
	print('Specifiy what to do with data')


f0fac=eta_curve**2/3


if F0_term and not Fullm:
	H_perp=(gxx**2+gxy**2)/(gxx+np.multiply(chi_curve,gxy))+f0fac/(1.0+chi_curve**2)
	H_wedg=((gxx**2+gxy**2)*chi_curve)/(gxx*chi_curve-gxy)+f0fac*chi_curve*(gxx+gxy)/(gxx*chi_curve-gxy)/(1+chi_curve**2)

	H_perp_inv=1/H_perp
	H_wedg_inv=1/H_wedg
elif not F0_term and not Fullm:
	H_perp=(gxx**2+gxy**2)/(gxx+np.multiply(chi_curve,gxy))
	H_wedg=-np.multiply((gxx**2+gxy**2),chi_curve)/(gxy-np.multiply(chi_curve,gxx))

	H_perp_inv=(gxx+np.multiply(chi_curve,gxy))/(gxx**2+gxy**2)
	H_wedg_inv=-(gxy/chi_curve-gxx)/(gxx**2+gxy**2)
elif Fullm and F0_term:
	H_perp=(gxy**2+(gxx+f0fac)**2)/(gxx+f0fac)/(1+chi_curve**2)
	H_wedg=-(gxy**2+(gxx+f0fac)**2)*chi_curve/(gxy*(1+chi_curve**2))

	H_perp_inv=1/H_perp
	H_wedg_inv=1/H_wedg

	H_wedg[:,0,:]=0.0
	H_wedg_inv[:,0,:]=0.0
elif Fullm and not F0_term:
	H_perp=(gxy**2+gxx**2)/(gxx*(1+chi_curve**2))
	H_wedg=-(gxy**2+gxx**2)*chi_curve/(gxy*(1+chi_curve**2))

	H_perp_inv=1/H_perp
	H_wedg_inv=1/H_wedg

	H_wedg[:,0,:]=0.0
	H_wedg_inv[:,0,:]=0.0


######################
#
#	Plotting and 2D maps
#
######################


#Hpara, Hperp, Hwedge unmagnetised comparison
##############
plt.loglog(etatherm,H_para[:,th_ind],'k-')
plt.title(r'$H_\parallel$')
plt.xlabel(r'$\eta_{th}$')
plt.grid(linestyle='dashed')
#plt.ylim(-1,5)
#plt.xlim(1.0,100)
if SAVE_FIGS:
	plt.savefig('Hparalineoutlog.png',dpi=600,transparent=True)
	plt.savefig('Hparalineoutlog.eps',dpi=600,transparent=True)
plt.show()


#colourmaps
##############
if visualise_G:
	plt.pcolormesh(chi_mesh[:,:,th_ind],eta_mesh[:,:,th_ind],gxx[:,:,th_ind],cmap='plasma',norm=clr.LogNorm())
	plt.title(r'$g_{xx}}$')
	plt.ylabel(r'$\eta_{th}$')
	plt.yscale('log')
	plt.xlabel(r'$\chi_{th}$')
	plt.colorbar()
	if SAVE_FIGS:
		plt.savefig('gxx2D.png',dpi=600,transparent=True)
		plt.savefig('gxx2D.eps',dpi=600,transparent=True)
	plt.show()

	plt.pcolormesh(chi_mesh[:,:,th_ind],eta_mesh[:,:,th_ind],-gxy[:,:,th_ind],cmap='plasma',norm=clr.LogNorm())
	plt.title(r'$g_{xy}$')
	plt.ylabel(r'$\eta_{th}$')
	plt.yscale('log')
	plt.xlabel(r'$\chi_{th}$')
	plt.colorbar()
	if SAVE_FIGS:
		plt.savefig('gxy2D.png',dpi=600,transparent=True)
		plt.savefig('gxy2D.eps',dpi=600,transparent=True)
	plt.show()


	plt.pcolormesh(chi_mesh[:,:,th_ind],eta_mesh[:,:,th_ind],H_perp[:,:,th_ind],cmap='plasma',norm=clr.LogNorm())
	plt.title(r'$H_{\perp}$')
	plt.ylabel(r'$\eta_{th}$')
	plt.yscale('log')
	plt.xlabel(r'$\chi_{th}$')
	plt.colorbar()
	if SAVE_FIGS:
		plt.savefig('Hperp2D.png',dpi=600,transparent=True)
		plt.savefig('Hperp2D.eps',dpi=600,transparent=True)
	plt.show()

	plt.pcolormesh(chi_mesh[:,:,th_ind],eta_mesh[:,:,th_ind],H_perp_inv[:,:,th_ind],cmap='plasma',norm=clr.LogNorm())
	plt.title(r'$1/H_{\perp}$')
	plt.ylabel(r'$\eta_{th}$')
	plt.yscale('log')
	plt.xlabel(r'$\chi_{th}$')
	plt.colorbar()
	if SAVE_FIGS:
		plt.savefig('Hperpinv2D.png',dpi=600,transparent=True)
		plt.savefig('Hperpinv2D.eps',dpi=600,transparent=True)
	plt.show()

	plt.pcolormesh(chi_mesh[:,:,th_ind],eta_mesh[:,:,th_ind],H_wedg[:,:,th_ind],cmap='plasma',norm=clr.LogNorm())
	plt.title(r'$H_{\wedge}$')
	plt.ylabel(r'$\eta_{th}$')
	plt.yscale('symlog')
	plt.xlabel(r'$\chi_{th}$')
	plt.colorbar()
	if SAVE_FIGS:
		plt.savefig('Hwedge2D.png',dpi=600,transparent=True)
		plt.savefig('Hwedge2D.eps',dpi=600,transparent=True)
	plt.show()


	plt.pcolormesh(chi_mesh[:,:,th_ind],eta_mesh[:,:,th_ind],H_wedg_inv[:,:,th_ind],cmap='plasma',norm=clr.LogNorm())
	plt.title(r'$1/H_{\wedge}$')
	plt.ylabel(r'$\eta_{th}$')
	plt.yscale('symlog')
	plt.xlabel(r'$\chi_{th}$')
	plt.colorbar()
	if SAVE_FIGS:
		plt.savefig('Hwedgeinv2D.png',dpi=600,transparent=True)
		plt.savefig('Hwedgeinv2D.eps',dpi=600,transparent=True)
	plt.show()


	#lineouts
	##############

	plt.semilogx(etatherm,H_perp[:,lineouts[0],th_ind],'k-',label=r'$\chi=0$')
	plt.semilogx(etatherm,H_perp[:,lineouts[1],th_ind],'r-',label=r'$\chi=0.1$')
	plt.semilogx(etatherm,H_perp[:,lineouts[2],th_ind],'y-.',label=r'$\chi=1$')
	plt.semilogx(etatherm,H_perp[:,lineouts[3],th_ind],'g:',label=r'$\chi=5$')
	plt.semilogx(etatherm,H_perp[:,lineouts[4],th_ind],'m--',label=r'$\chi=10$')
	plt.semilogx(etatherm,H_perp[:,lineouts[5],th_ind],'b-',label=r'$\chi=30$')
	plt.title(r'$H_\perp$ at Different Magnetisations')
	plt.xlabel(r'$\eta_{th}$')
	plt.legend()
	plt.grid(linestyle='dotted')
	plt.ylim(0,2)
	if SAVE_FIGS:
		plt.savefig('Hperp_lineouts.png',dpi=600,transparent=True)
		plt.savefig('Hperp_lineouts.eps',dpi=600,transparent=True)
	plt.show()

	plt.semilogx(etatherm,H_perp_inv[:,lineouts[0],th_ind],'k-',label=r'$\chi=0$')
	plt.semilogx(etatherm,H_perp_inv[:,lineouts[1],th_ind],'r-',label=r'$\chi=0.1$')
	plt.semilogx(etatherm,H_perp_inv[:,lineouts[2],th_ind],'y-.',label=r'$\chi=1$')
	plt.semilogx(etatherm,H_perp_inv[:,lineouts[3],th_ind],'g:',label=r'$\chi=5$')
	plt.semilogx(etatherm,H_perp_inv[:,lineouts[4],th_ind],'m--',label=r'$\chi=10$')
	plt.semilogx(etatherm,H_perp_inv[:,lineouts[5],th_ind],'b-',label=r'$\chi=30$')
	plt.title(r'$1/H_\perp$ at Different Magnetisations')
	plt.xlabel(r'$\eta_{th}$')
	plt.legend()
	plt.grid(linestyle='dotted')
	plt.ylim(0,2)
	if SAVE_FIGS:
		plt.savefig('Hperpinv_lineouts.png',dpi=600,transparent=True)
		plt.savefig('Hperpinv_lineouts.eps',dpi=600,transparent=True)
	plt.show()


	plt.semilogx(etatherm,H_wedg[:,lineouts[0],th_ind],'k-',label=r'$\chi=0$')
	plt.semilogx(etatherm,H_wedg[:,lineouts[1],th_ind],'r-',label=r'$\chi=0.1$')
	plt.semilogx(etatherm,H_wedg[:,lineouts[2],th_ind],'y-.',label=r'$\chi=1$')
	plt.semilogx(etatherm,H_wedg[:,lineouts[3],th_ind],'g:',label=r'$\chi=5$')
	plt.semilogx(etatherm,H_wedg[:,lineouts[4],th_ind],'m--',label=r'$\chi=10$')
	plt.semilogx(etatherm,H_wedg[:,lineouts[5],th_ind],'b-',label=r'$\chi=30$')
	plt.title(r'$H_\wedge$ at Different Magnetisations')
	plt.xlabel(r'$\eta_{th}$')
	plt.legend()
	plt.grid(linestyle='dotted')
	plt.ylim(0.8,1.5)
	plt.xlim(1e-2,1e1)
	if SAVE_FIGS:
		plt.savefig('Hwedg_lineouts.png',dpi=600,transparent=True)
		plt.savefig('Hwedg_lineouts.eps',dpi=600,transparent=True)
	plt.show()

	plt.semilogx(etatherm,H_wedg_inv[:,lineouts[0],th_ind],'k-',label=r'$\chi=0$')
	plt.semilogx(etatherm,H_wedg_inv[:,lineouts[1],th_ind],'r-',label=r'$\chi=0.1$')
	plt.semilogx(etatherm,H_wedg_inv[:,lineouts[2],th_ind],'y-.',label=r'$\chi=1$')
	plt.semilogx(etatherm,H_wedg_inv[:,lineouts[3],th_ind],'g:',label=r'$\chi=5$')
	plt.semilogx(etatherm,H_wedg_inv[:,lineouts[4],th_ind],'m--',label=r'$\chi=10$')
	plt.semilogx(etatherm,H_wedg_inv[:,lineouts[5],th_ind],'b-',label=r'$\chi=30$')
	plt.title(r'$1/H_\wedge$ at Different Magnetisations')
	plt.xlabel(r'$\eta_{th}$')
	plt.legend()
	plt.grid(linestyle='dotted')
	plt.ylim(-0.5,1.5)
	plt.xlim(1e-2,1e2)
	if SAVE_FIGS:
		plt.savefig('Hwedginv_lineouts.png',dpi=600,transparent=True)
		plt.savefig('Hwedginv_lineouts.eps',dpi=600,transparent=True)
	plt.show()


######################
#
#	transport coefficients
#
######################

#either start with the numerically derive H functions from the continued fractions
#or use the fitted forms,
#both will require defining a 3D mesh (eta by chi by v) and integrating over the v axis
#compare with the classical result in which H functions are 1
#normalise the functions to the classical lorentz unmagnetised result

corr=2*np.sqrt(2)*3*np.sqrt(3.1415)/4.0
'''
if Fullm and not F0_term:
	interp_Hperp=gxx/(gxx**2+gxy**2)
	interp_Hwedg=-gxy/(gxx**2+gxy**2)
elif Fullm and F0_term:
	interp_Hperp=gxx/(gxx**2+gxy**2+gxx*f0fac)
	interp_Hwedg=-gxy/(gxx**2+gxy**2+gxx*f0fac)
else:
'''

interp_Hperp=H_perp_inv
interp_Hwedg=H_wedg_inv
interp_Hpara=1.0/H_para

#maxwell boltzmann distribution
maxwell=0.0635*np.exp(-0.5*v_mesh**2.0)
maxwell2=0.0635*np.exp(-0.5*v_para_mesh**2.0)

#do as an array instead of looping over chi, eta. use outer to get meshgrid of eta,chi,v. integrate along axis

def para_integrator(powr):
	integrand=interp_Hpara*maxwell2*(v_para_mesh**powr)
	return integrate.simps(integrand,v)

def perp_integrator(powr):
#	if Fullm:
#		integrand=interp_Hperp*(maxwell*(v_mesh**powr))
#	else:
	denom=1.0+((chi_mesh/corr)**2.0)*((v_mesh)**6.0)
	integrand=interp_Hperp*(maxwell*(v_mesh**powr))/denom
	return integrate.simps(integrand,v)

def wedg_integrator(powr):
#	if Fullm:
#		integrand=interp_Hwedg*maxwell*(v_mesh**(powr+3))
#	else:
	denom=1.0+((chi_mesh/corr)**2.0)*((v_mesh)**6.0)
	integrand=interp_Hwedg*(chi_mesh*maxwell*(v_mesh**(powr+3))/corr)/denom
	return integrate.simps(integrand,v)


#functions for building the coefficients, built out of integrating the magnetised f1
v7para=para_integrator(7)
v9para=para_integrator(9)
v11para=para_integrator(11)

v7perp=perp_integrator(7)
v9perp=perp_integrator(9)
v11perp=perp_integrator(11)

v7wedg=wedg_integrator(7)
v9wedg=wedg_integrator(9)
v11wedg=wedg_integrator(11)

#dimensionless coefficients, functions of chitherm and etatherm
denom2=(v7perp**2.0+v7wedg**2.0)

alphapara=corr*0.238/v7para
alphaperp=corr*0.238*v7perp/denom2
alphawedge=corr*0.238*(0.5*v7wedg/denom2)

aexact=corr*(1.5*(v7wedg/denom2)-1.0)

betapara=0.5*v9para/v7para-2.5
betaperp=0.5*((v9perp*v7perp+v9wedg*v7wedg)/denom2)-2.5
betawedge=-0.5*((v9perp*v7wedg-v9wedg*v7perp)/denom2)

kappapara=1.047*(v11para-(v9para**2.0)/v7para)/corr
kappaperp=1.047*(v11perp-(2.0*v9perp*v9wedg*v7wedg+v7perp*v9perp**2.0-v7perp*v9wedg**2.0)/denom2)/corr
kappawedge=1.047*(v11wedg-(2.0*v9wedg*v7perp*v9perp+v7wedg*v9wedg**2.0-v7wedg*v9perp**2.0)/denom2)/corr

######################
#
#  Comparison curves: Expressions taken from epperlein/haines '86
#
######################
#lorentz approx Z->inf

chitherm2=chitherm
apara=0.2945*np.ones(etatherm.shape)
aperp=1.0-(chitherm2*4.73+9.17)/(chitherm2**2+chitherm2*13.8+13.0)
awedg=chitherm2*(chitherm2*2.53+1.33e3)/(chitherm2**3+1.18e3*chitherm2**2+2.19e4*chitherm2+3.54e3)**0.888

kpara=13.58*np.ones(etatherm.shape)
kperp=(chitherm2*3.25+6.21)/(chitherm2**3+8.53*chitherm2**2+chitherm2*4.81+4.57e-1)
kwedg=chitherm2*(chitherm2*2.5+1.86e-1)/(chitherm2**3+4.30e-1*chitherm2**2+chitherm2*1.8e-2+1.08e-3)

bpara=1.5*np.ones(etatherm.shape)
bperp=(6.33*chitherm2+2.2e3)/(chitherm2**3+8.09e2*chitherm2**2+1.68e4*chitherm2+3.65e3)**0.888
bwedg=chitherm2*(1.5*chitherm2+2.15)/(chitherm2**3+6.72*chitherm2**2+2.53*chitherm2+2.19e-1)

#######################
#
#	asymptotic form of kappa perp
#
#######################

compare2=kappaperp[0,lineouts[0]]/(1+etatherm**2)
compare1=kappaperp[0,lineouts[0]]/(1+etatherm)
plt.loglog(etatherm,kappaperp[:,lineouts[0]],label='from model')
plt.loglog(etatherm,compare1,label='power=1')
plt.loglog(etatherm,compare2,label='power=2')
plt.legend()
plt.show()

#######################
#
#	Comparison plots, between local values and EpHa forms
#
#######################

if comparison_plots:
	##parallel terms
	fs=12

	fig, ax= plt.subplots(1,3,figsize=(9,3))

	ax[0].semilogx(etatherm, apara,'k-')
	ax[0].semilogx(etatherm, alphapara,'b--')
	ax[0].grid(linestyle='dotted')
	ax[0].set_xlabel(r'$\eta_{th}$', fontsize=fs)
	ax[0].set_ylabel(r'$\alpha_\parallel^c$',fontsize=fs)
	ax[0].set_ylim([0,1])

	ax[1].semilogx(etatherm, bpara,'k-')
	ax[1].semilogx(etatherm, betapara,'b--')
	ax[1].grid(linestyle='dotted')
	ax[1].set_xlabel(r'$\eta_{th}$', fontsize=fs)
	ax[1].set_ylabel(r'$\beta_\parallel^c$',fontsize=fs)
	ax[1].set_ylim([-0.6,1.6])

	ax[2].semilogx(etatherm, kpara,'k-')
	ax[2].semilogx(etatherm, kappapara,'b--')
	ax[2].grid(linestyle='dotted')
	ax[2].set_xlabel(r'$\eta_{th}$', fontsize=fs)
	ax[2].set_ylabel(r'$\kappa_\parallel^c$',fontsize=fs)
	ax[2].set_ylim([-1,14])

	fig.tight_layout()
	if SAVE_FIGS:
		fig.savefig("para_comparison.png",dpi=600,transparent=True)
		fig.savefig("para_comparison.eps",dpi=600,transparent=True)
	plt.show()

	##perpendicular terms
	fig2, ax2= plt.subplots(3,2,figsize=(8,10))

	ax2[0,0].loglog(chitherm, aperp,'k-')
	ax2[0,0].loglog(chitherm, alphaperp[0,:],'b--')
	ax2[0,0].grid(linestyle='dotted')
	ax2[0,0].set_xlabel(r'$\chi_{th}$', fontsize=fs)
	ax2[0,0].set_ylabel(r'$\alpha_\perp^c$',fontsize=fs)
	#ax2[0,0].set_ylim([-0.5,2])
	ax2[0,0].set_xlim([0.001,40])

	ax2[0,1].loglog(chitherm, awedg,'k-')
	ax2[0,1].loglog(chitherm, alphawedge[0,:],'b--')
	ax2[0,1].grid(linestyle='dotted')
	ax2[0,1].set_xlabel(r'$\chi_{th}$', fontsize=fs)
	ax2[0,1].set_ylabel(r'$\alpha_\wedge^c$',fontsize=fs)
	#ax2[0,1].set_ylim([-0.5,1])
	ax2[0,1].set_xlim([0.001,40])

	ax2[1,0].loglog(chitherm, bperp,'k-')
	ax2[1,0].loglog(chitherm, betaperp[0,:],'b--')
	ax2[1,0].grid(linestyle='dotted')
	ax2[1,0].set_xlabel(r'$\chi_{th}$', fontsize=fs)
	ax2[1,0].set_ylabel(r'$\beta_\perp^c$',fontsize=fs)
	#ax2[1,0].set_ylim([-0.5,2])
	ax2[1,0].set_xlim([0.001,40])

	ax2[1,1].loglog(chitherm, bwedg,'k-')
	ax2[1,1].loglog(chitherm, betawedge[0,:],'b--')
	ax2[1,1].grid(linestyle='dotted')
	ax2[1,1].set_xlabel(r'$\chi_{th}$', fontsize=fs)
	ax2[1,1].set_ylabel(r'$\beta_\wedge^c$',fontsize=fs)
	#ax2[1,1].set_ylim([-0.1,0.75])
	ax2[1,1].set_xlim([0.001,40])

	ax2[2,0].loglog(chitherm, kperp,'k-')
	ax2[2,0].loglog(chitherm, kappaperp[0,:],'b--')
	ax2[2,0].grid(linestyle='dotted')
	ax2[2,0].set_xlabel(r'$\chi_{th}$', fontsize=fs)
	ax2[2,0].set_ylabel(r'$\kappa_\perp^c$',fontsize=fs)
	#ax2[2,0].set_ylim([-0.5,20])
	ax2[2,0].set_xlim([0.001,40])

	ax2[2,1].loglog(chitherm, kwedg,'k-')
	ax2[2,1].loglog(chitherm, kappawedge[0,:],'b--')
	ax2[2,1].grid(linestyle='dotted')
	ax2[2,1].set_xlabel(r'$\chi_{th}$', fontsize=fs)
	ax2[2,1].set_ylabel(r'$\kappa_\wedge^c$',fontsize=fs)
	#ax2[2,1].set_ylim([-0.5,10])
	ax2[2,1].set_xlim([0.001,40])

	fig2.tight_layout()
	if SAVE_FIGS:
		fig2.savefig("mag_comparison.png",dpi=600,transparent=True)
		fig2.savefig("mag_comparison.eps",dpi=600,transparent=True)
	plt.show()

#######################
#
#	coefficient plotting
#
#######################

if coef_plots:

	chi_mesh=chi_mesh[:,:,th_ind]
	eta_mesh=eta_mesh[:,:,th_ind]

	#paras
	###########
	plt.semilogx(etatherm,alphapara,'k-',label=r'$\alpha^c_\parallel$')
	plt.semilogx(etatherm,betapara,'r-',label=r'$\beta^c_\parallel$')
	plt.semilogx(etatherm,kappapara,'b-',label=r'$\kappa^c_\parallel$')
	plt.title(r'Parallel Transport Coefficients')
	plt.yscale('log')
	plt.xlabel(r'$\eta_{th}$')
	plt.legend()
	plt.grid(linestyle='dotted')
	plt.ylim(0.01,100)
	plt.xlim(0.001,100)
	if SAVE_FIGS:
		plt.savefig('parallel_lineouts.png',dpi=600,transparent=True)
		plt.savefig('parallel_lineouts.eps',dpi=600,transparent=True)
	plt.show()

	#norm=clr.LogNorm()


	#alphas
	###########
	plt.pcolormesh(chi_mesh,eta_mesh,alphaperp,cmap='plasma',norm=clr.LogNorm())
	plt.title(r'$\alpha^c_{\perp}$')
	plt.ylabel(r'$\eta_{th}$')
	plt.xlabel(r'$\chi_{th}$')
	plt.yscale('log')
	#plt.xscale('log')
	#plt.ylim(0.1,10)
	plt.colorbar()
	if SAVE_FIGS:
		plt.savefig('alphaperp2D.png',dpi=600,transparent=True)
		plt.savefig('alphaperp2D.eps',dpi=600,transparent=True)
	plt.show()

	plt.semilogx(etatherm,alphaperp[:,lineouts[0]],'k-',label=r'$\chi=0$')
	plt.semilogx(etatherm,alphaperp[:,lineouts[1]],'r-',label=r'$\chi=0.1$')
	plt.semilogx(etatherm,alphaperp[:,lineouts[2]],'y-.',label=r'$\chi=1$')
	plt.semilogx(etatherm,alphaperp[:,lineouts[3]],'g:',label=r'$\chi=5$')
	plt.semilogx(etatherm,alphaperp[:,lineouts[4]],'m--',label=r'$\chi=10$')
	plt.semilogx(etatherm,alphaperp[:,lineouts[5]],'b-',label=r'$\chi=30$')
	plt.title(r'$\alpha^c_\perp$ at Different Magnetisations')
	plt.xlabel(r'$\eta_{th}$')
	plt.legend()
	plt.grid(linestyle='dotted')
	#plt.yscale('log')
	plt.ylim(0,5)
	if SAVE_FIGS:
		plt.savefig('alphaperp_lineouts.png',dpi=600,transparent=True)
		plt.savefig('alphaperp_lineouts.eps',dpi=600,transparent=True)
	plt.show()

	plt.pcolormesh(chi_mesh,eta_mesh,alphawedge,cmap='plasma',norm=clr.SymLogNorm(linthresh=0.0001))
	plt.title(r'$\alpha^c_{\wedge}$')
	plt.ylabel(r'$\eta_{th}$')
	plt.xlabel(r'$\chi_{th}$')
	plt.yscale('log')
	#plt.xscale('log')
	plt.colorbar()
	if SAVE_FIGS:
		plt.savefig('alphawedge2D.png',dpi=600,transparent=True)
		plt.savefig('alphawedge2D.eps',dpi=600,transparent=True)
	plt.show()

	plt.semilogx(etatherm,alphawedge[:,lineouts[0]],'k-',label=r'$\chi=0$')
	plt.semilogx(etatherm,alphawedge[:,lineouts[1]],'r-',label=r'$\chi=0.1$')
	plt.semilogx(etatherm,alphawedge[:,lineouts[2]],'y-.',label=r'$\chi=1$')
	plt.semilogx(etatherm,alphawedge[:,lineouts[3]],'g:',label=r'$\chi=5$')
	plt.semilogx(etatherm,alphawedge[:,lineouts[4]],'m--',label=r'$\chi=10$')
	plt.semilogx(etatherm,alphawedge[:,lineouts[5]],'b-',label=r'$\chi=30$')
	plt.title(r'$\alpha^c_\wedge$ at Different Magnetisations')
	plt.xlabel(r'$\eta_{th}$')
	plt.legend()
	plt.grid(linestyle='dotted')
	#plt.yscale('symlog',linthresh=0.0001)
	plt.ylim(-1,5)
	if SAVE_FIGS:
		plt.savefig('alphawedg_lineouts.png',dpi=600,transparent=True)
		plt.savefig('alphawedg_lineouts.eps',dpi=600,transparent=True)
	plt.show()

	#betas
	###########

	plt.pcolormesh(chi_mesh,eta_mesh,betaperp,cmap='coolwarm',vmax=1,vmin=-1)
	plt.title(r'$\beta^c_{\perp}$')
	plt.ylabel(r'$\eta_{th}$')
	plt.xlabel(r'$\chi_{th}$')
	plt.yscale('log')
	#plt.xscale('log')
	plt.colorbar()
	if SAVE_FIGS:
		plt.savefig('betaperp2D.png',dpi=600,transparent=True)
		plt.savefig('betaperp2D.eps',dpi=600,transparent=True)
	plt.show()

	plt.semilogx(etatherm,betaperp[:,lineouts[0]],'k-',label=r'$\chi=0$')
	plt.semilogx(etatherm,betaperp[:,lineouts[1]],'r-',label=r'$\chi=0.1$')
	plt.semilogx(etatherm,betaperp[:,lineouts[2]],'y-.',label=r'$\chi=1$')
	plt.semilogx(etatherm,betaperp[:,lineouts[3]],'g:',label=r'$\chi=5$')
	plt.semilogx(etatherm,betaperp[:,lineouts[4]],'m--',label=r'$\chi=10$')
	plt.semilogx(etatherm,betaperp[:,lineouts[5]],'b-',label=r'$\chi=30$')
	plt.title(r'$\beta^c_\perp$ at Different Magnetisations')
	plt.xlabel(r'$\eta_{th}$')
	plt.legend()
	plt.grid(linestyle='dotted')
	#plt.yscale('symlog', linthresh=0.0001)
	#plt.ylim(-2,2)
	if SAVE_FIGS:
		plt.savefig('betaperp_lineouts.png',dpi=600,transparent=True)
		plt.savefig('betaperp_lineouts.eps',dpi=600,transparent=True)
	plt.show()

	plt.pcolormesh(chi_mesh,eta_mesh,betawedge,cmap='coolwarm',vmax=1,vmin=-1)
	plt.title(r'$\beta^c_{\wedge}$')
	plt.ylabel(r'$\eta_{th}$')
	plt.xlabel(r'$\chi_{th}$')
	plt.yscale('log')
	#plt.xscale('log')
	plt.colorbar()
	if SAVE_FIGS:
		plt.savefig('betawedge2D.png',dpi=600,transparent=True)
		plt.savefig('betawedge2D.eps',dpi=600,transparent=True)
	plt.show()

	plt.semilogx(etatherm,betawedge[:,lineouts[0]],'k-',label=r'$\chi=0$')
	plt.semilogx(etatherm,betawedge[:,lineouts[1]],'r-',label=r'$\chi=0.1$')
	plt.semilogx(etatherm,betawedge[:,lineouts[2]],'y-.',label=r'$\chi=1$')
	plt.semilogx(etatherm,betawedge[:,lineouts[3]],'g:',label=r'$\chi=5$')
	plt.semilogx(etatherm,betawedge[:,lineouts[4]],'m--',label=r'$\chi=10$')
	plt.semilogx(etatherm,betawedge[:,lineouts[5]],'b-',label=r'$\chi=30$')
	plt.title(r'$\beta_\wedge$ at Different Magnetisations')
	plt.xlabel(r'$\eta_{th}$')
	plt.legend()
	plt.grid(linestyle='dotted')
	#plt.yscale('symlog',linthresh=0.0001)
	#plt.ylim(-2,2)
	if SAVE_FIGS:
		plt.savefig('betawedg_lineouts.png',dpi=600,transparent=True)
		plt.savefig('betawedg_lineouts.eps',dpi=600,transparent=True)
	plt.show()

	#kappas
	############

	plt.pcolormesh(chi_mesh,eta_mesh,kappaperp,cmap='plasma')
	plt.title(r'$\kappa^c_{\perp}$')
	plt.ylabel(r'$\eta_{th}$')
	plt.xlabel(r'$\chi_{th}$')
	plt.yscale('log')
	#plt.xscale('log')
	plt.colorbar()
	if SAVE_FIGS:
		plt.savefig('kappaperp2D.png',dpi=600,transparent=True)
		plt.savefig('kappaperp2D.eps',dpi=600,transparent=True)
	plt.show()

	plt.semilogx(etatherm,kappaperp[:,lineouts[0]],'k-',label=r'$\chi=0$')
	plt.semilogx(etatherm,kappaperp[:,lineouts[1]],'r-',label=r'$\chi=0.1$')
	plt.semilogx(etatherm,kappaperp[:,lineouts[2]],'y-.',label=r'$\chi=1$')
	plt.semilogx(etatherm,kappaperp[:,lineouts[3]],'g:',label=r'$\chi=5$')
	plt.semilogx(etatherm,kappaperp[:,lineouts[4]],'m--',label=r'$\chi=10$')
	plt.semilogx(etatherm,kappaperp[:,lineouts[5]],'b-',label=r'$\chi=30$')
	plt.title(r'$\kappa^c_\perp$ at Different Magnetisations')
	plt.xlabel(r'$\eta_{th}$')
	plt.legend()
	plt.grid(linestyle='dotted')
	#plt.yscale('symlog')
	#plt.ylim(-1,2)
	if SAVE_FIGS:
		plt.savefig('kappaperp_lineouts.png',dpi=600,transparent=True)
		plt.savefig('kappaperp_lineouts.eps',dpi=600,transparent=True)
	plt.show()

	plt.pcolormesh(chi_mesh,eta_mesh,kappawedge,cmap='plasma')
	plt.title(r'$\kappa^c_{\wedge}$')
	plt.ylabel(r'$\eta_{th}$')
	plt.xlabel(r'$\chi_{th}$')
	plt.yscale('log')
	#plt.xscale('log')
	plt.colorbar()
	if SAVE_FIGS:
		plt.savefig('kappawedge2D.png',dpi=600,transparent=True)
		plt.savefig('kappawedge2D.eps',dpi=600,transparent=True)
	plt.show()

	plt.semilogx(etatherm,kappawedge[:,lineouts[0]],'k-',label=r'$\chi=0$')
	plt.semilogx(etatherm,kappawedge[:,lineouts[1]],'r-',label=r'$\chi=0.1$')
	plt.semilogx(etatherm,kappawedge[:,lineouts[2]],'y-.',label=r'$\chi=1$')
	plt.semilogx(etatherm,kappawedge[:,lineouts[3]],'g:',label=r'$\chi=5$')
	plt.semilogx(etatherm,kappawedge[:,lineouts[4]],'m--',label=r'$\chi=10$')
	plt.semilogx(etatherm,kappawedge[:,lineouts[5]],'b-',label=r'$\chi=30$')
	plt.title(r'$\kappa^c_\wedge$ at Different Magnetisations')
	plt.xlabel(r'$\eta_{th}$')
	plt.legend()
	plt.grid(linestyle='dotted')
	#plt.yscale('symlog')
	#plt.ylim(-1,2)
	if SAVE_FIGS:
		plt.savefig('kappawedg_lineouts.png',dpi=600,transparent=True)
		plt.savefig('kappawedg_lineouts.eps',dpi=600,transparent=True)
	plt.show()
