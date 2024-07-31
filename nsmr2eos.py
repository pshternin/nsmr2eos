import numpy as np
import os, sys
import emcee
import scipy.stats as stats
import time
import emcee
from multiprocessing import Pool
import scipy
import h5py 
import scipy.optimize as op


os.environ["OMP_NUM_THREADS"] = "1"
np.random.seed(42)

#Importing the sources data from PathToData
PathToData=r"DATA/"

#data in form of samples from posteriors

#Cassiopeia A
casaname=r"CasA.txt"
casa=np.genfromtxt(PathToData+casaname)

#PSR J0030+0451
miller00303spotname="J0030_3spot_RM.txt"
miller0030=np.genfromtxt(PathToData+miller00303spotname,comments="#")


#PSR J0740+6620
riley0740name=r"J0740_NICERxXMM_NANOGrav_CHIMEpulsar__xpsi_STU_FIH__columns__mass_solar__eqradius_km.txt"
riley0740=np.genfromtxt(PathToData+riley0740name)

#PSR J0437-4715
nicer0437name=r"nlive20000_expf3.3_noCONST_noMM_tol0.1post_equal_weights_mr.dat"
nicer0437=np.genfromtxt(PathToData+nicer0437name)

#GW170817
GW170817name=r"EoS-insensitive_posterior_samples.dat"
GW170817=np.genfromtxt(PathToData+GW170817name,comments="#")


#make KDEs
riley0740kde=stats.gaussian_kde(np.transpose(riley0740))

Casakde=stats.gaussian_kde(np.transpose(casa))
miller0030kde=stats.gaussian_kde(np.transpose(miller0030[:,[1,0]]))

nicer0437kde=stats.gaussian_kde(np.transpose(nicer0437))

#making KDE of 4D distribution for GW170817
GW170817kde=stats.gaussian_kde(np.transpose(GW170817[:,[0,4,1,5]]))



NGW=1


#bursters
#Here the data is provided in form of likelihood surfaces instead of sampes

#4U J1702-429
fname1702=r"1702_D_X_int.h5"
dset1702 = h5py.File(PathToData+fname1702,'r')

avgs1702 = dset1702['prob'][()]
r1702 = dset1702['xval'][()]
m1702 = dset1702['yval'][()]

int1702=scipy.interpolate.RegularGridInterpolator((m1702,r1702),avgs1702,bounds_error=False, fill_value=-1)


#SAX J1810.8-260
fname1810=r"1810.h5"
dset1810 = h5py.File(PathToData+fname1810,'r')

avgs1810 = dset1810['prob'][()]
r1810 = dset1810['xval'][()]
m1810 = dset1810['yval'][()]

int1810=scipy.interpolate.RegularGridInterpolator((m1810,r1810),avgs1810,bounds_error=False, fill_value=-1)

#4U 1724-307
fname1724=r"1724.h5"
dset1724 = h5py.File(PathToData+fname1724,'r')

avgs1724 = dset1724['prob'][()]
r1724 = dset1724['xval'][()]
m1724 = dset1724['yval'][()]

int1724=scipy.interpolate.RegularGridInterpolator((m1724,r1724),avgs1724,bounds_error=False, fill_value=-1)

#list of functions providing likelihoods for all sources except GW170817
likes=[riley0740kde.pdf,Casakde.pdf,miller0030kde.pdf,nicer0437kde.pdf,int1702, int1724, int1810]


#spiders

spiders=np.array([[2.22,0.1],[2.11,0.04],[2.15,0.16],[2.28,0.1],[2.35,0.17]])


#radiopulsars
radio=np.array([[2.01,0.04],[1.97,0.04]])


#Universal  relations

def MRddo(m,rho):
    a=-0.492 
    mu = np.sqrt(1-m)
    mu2=1-m
    c=0.8284271247461903
    return 1+mu*(a+c*rho) + mu2*(-2.-np.sqrt(2)*a+c*rho)

def fddo(pmax,rhomax,vec):
    (a,b,c,d)=vec
    return c*(pmax**a)*(rhomax**b)+d

vecM=[1.41,-1.39,5.86,0.27]
vecR=[0.744,-0.839,2.45,0.357]
vecc=[0.985,-0.990,3.02,0.529]  



Nlikes=len(likes)

Nsources=Nlikes+2*NGW # 2 for GW170817 


#physical constants
c=2.998e10
G=6.67e-8
rho0=2.8e14
Msun=1.988e33
RJ=c/np.sqrt(G*rho0)/1.e5
MJ=rho0*(RJ*1.e5)**3/Msun



def lnprior(p):
    
    pmax,rhomax, rho = p[0:3]
    Mss=p[3:]
    
    if pmax<0.1 or rhomax<0.1:
        return -np.inf
    
    Mmax=MJ*pmax**1.5/rhomax**2/fddo(pmax, rhomax, vecM)
    Rmax0=pmax**0.5/rhomax/fddo(pmax, rhomax, vecR)
    csmax=Rmax0*rhomax**0.5*fddo(pmax, rhomax, vecc)
    Rmax=Rmax0*RJ

        
    
    if np.any(Mss>Mmax):
        return -np.inf
    
    #account for sources with measured masses only
    
    radioprior=np.array([scipy.stats.norm.logcdf(Mmax,loc=y[0],scale=y[1]) for y in radio])
    
    spidersprior=np.array([scipy.stats.norm.logcdf(Mmax,loc=y[0],scale=y[1]) for y in spiders])
        
    kilonovaprior=scipy.stats.norm.logsf(Mmax, loc=2.16,scale=0.08)                             #kilonovaprior
    
    if 1.5<Mmax<3. and 1.<rho<2. and 5.<Rmax<30. and Mmax<0.24*Rmax and csmax<1.05:
        return radioprior.sum()+spidersprior.sum()+kilonovaprior
    
    return -np.inf

def lnlike(p):
    
    infblobs=[-np.inf for i in range(Nsources)] #blobs contains source radii
    
    lnp=lnprior(p)
    if not np.isfinite(lnp):
        return -np.inf, *infblobs
    
    pmax,rhomax, rho = p[0:3]

    MGW=p[3:5]

    Mss=p[5:]
    

    
    Mmax=MJ*pmax**1.5/rhomax**2/fddo(pmax, rhomax, vecM)
    Rmax0=pmax**0.5/rhomax/fddo(pmax, rhomax, vecR)
    Rmax=Rmax0*RJ
    
    Rs=Rmax*MRddo(Mss/Mmax,rho)
    ls=np.array([kde([m,r]) for kde,m,r in zip(likes,Mss,Rs)])
    
    if np.any(ls<=0):
        return - np.inf,*infblobs
    
    ll=np.log10(ls).sum()
    
#GW data require separate treatment, since here 4D posterior is used        
    RGW=Rmax*MRddo(MGW/Mmax,rho)
    likeGW=np.log(GW170817kde.pdf([MGW[0],RGW[0],MGW[1],RGW[1]]))
    
    return lnp+ll+likeGW,*RGW,*Rs



def neglnlike(p):
    r=-lnlike(p)[0]
#    print(p)
#    print(r)
    return r
    

#chain setup

Nwalk=32
Nsteps=30000
NPar=3+Nsources


#initial parameter guess
initpar=np.array([3.6,7.8,1.06,1.44,1.28,2.06,1.6,1.3,1.4,1.7,1.3,1.4])
initpars=np.array([initpar+0.001*(np.random.rand(NPar)-0.5) for i in range(Nwalk)])


chain_name=r"posterior_run0740_salmi2024_0030_vinci2024_STPST.h5"
backend=emcee.backends.HDFBackend(chain_name)


#to continue sampling set cont=1
cont=1
if not(cont):
    backend.reset(Nwalk,NPar)

#set fit=1 to perform MAP estimation instead of MCMC
#put cont=1 to start minimization from estimated posterior means
fit=0
def print_params(xk):
    
    print(xk,end="\r")

    



if fit:
    if backend.iteration<500:
        meanpars=initpar
    else:
        disc=backend.iteration//2
        flatchain=backend.get_chain(flat=True,discard=disc)
        meanpars=np.mean(flatchain,axis=0)
    
  #  meanpars=initpars
    
    lw=np.get_printoptions()['linewidth']
    np.set_printoptions(linewidth=np.inf)

    print(meanpars)
    
    result=op.minimize(neglnlike,meanpars,method='Nelder-Mead',options={'maxiter':100000, 'disp':True},callback=print_params)    

    
    print("N iterations= ",result.nit)
    print(result.x)
    
    
    
    pmax=result.x[0]
    rhomax=result.x[1]
    Mmax=MJ*pmax**1.5/rhomax**2/fddo(pmax, rhomax, vecM)
    Rmax0=pmax**0.5/rhomax/fddo(pmax, rhomax, vecR)
    Rmax=Rmax0*RJ
    
    print("Mtov=",Mmax," Rtov=",Rmax)
    
    
    np.set_printoptions(linewidth=lw)
    
    
    np.savetxt("bf_"+chain_name+".txt",result.x)       
    sys.exit()



acor_calc=100   #steps before the next autocorrelation time estimate

autocorr=np.empty((Nsteps//acor_calc)+1)

old_tau=0

index=0

#running MCMC chain with emcee. Convergence is controlled by autocorrellation time

with Pool() as pool:
    sampler = emcee.EnsembleSampler(Nwalk, NPar, lnlike,backend=backend,pool=pool)

    start=time.time()

    

    if cont:
        initpars=backend.get_last_sample()
    
    for sample in sampler.sample(initpars,iterations=Nsteps,progress=True):
        
        if sampler.iteration%acor_calc:
            continue
        
        tau=sampler.get_autocorr_time(tol=0,discard=sampler.iteration//2)
        autocorr[index]=np.mean(tau)
        index+=1
        print(np.mean(tau))
        print(f"\r\033[{2}A",end="")
        
        converged=np.all(tau*50<sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau)/tau<0.01)
        
       
        if converged:
            print("chain converged")
            print("half-chain acor time: ", tau)
            tauf=sampler.get_autocorr_time(tol=0,discard=0)
            print("full chain acor time: ", tauf)
            break
        old_tau=tau
        
    end=time.time()

    run_time=end-start

    print("MCMC run took {0:4.2f}seconds".format(run_time))








