'''
code in python3
Needs to be in the same directory as SONG  (/.../song/) 
The python part of song call songy is written in python2. Make the following change:
    In /song/python/songy.py line 170:
         range(0,N*(N+1)/2) --> list(range(0,int(N*(N+1)/2)))
'''
####################################################################################### import
import numpy as np
import os
import sys
from numba import jit,prange,config
import h5py
import subprocess 
import matplotlib.pyplot as plt
import time
import params as params
import importlib
importlib.reload(params)
sys.path.insert(0, params.song_path+'python')          # path to python module of song
import songy as s
import Pk_library as PKL

####################################################################################### 
####################################################################################### SONG wrapper
def run_song(hkmax,hkmin,opt):
    ''' Call this function in song repository:
        It will create the ini and pre files from the global parameters and run song
        '''

    ini_file=r"""output = delta_cdm_bk
T_cmb = 2.7255
N_eff = 3.046
reio_parametrization = reio_none
tau_reio = 0.0952
k_pivot = 0.05
A_s = {}
n_s = {}
YHe = 0.2477055
gauge = newtonian
output_single_precision = yes
output_class_perturbations = yes
background_verbose = 1
thermodynamics_verbose = 1
primordial_verbose = 1
spectra_verbose = 1
nonlinear_verbose = 1
lensing_verbose = 1
output_verbose = 1
perturbations_verbose = 1
perturbations2_verbose = 2
transfer_verbose = 1
transfer2_verbose = 1
bessels_verbose = 1
bessels2_verbose = 1
bispectra_verbose = 1
fisher_verbose = 1
format = camb
write parameters = yes
h={}
omega_b={}
omega_cdm={}
Omega_k={}
primordial_local_fnl_phi={}
z_out={}"""

    pre_file=r"""sources2_k3_sampling = {}
#k3_size = {}
k_min_tau0 = 0.05
k_max_tau0_over_l_max = 2.8 
k_step_sub =0.1
k_logstep_super = 1.2
k_step_super = 0.025
k_step_transition = 0.2
quadsources_time_interpolation = cubic
sources_time_interpolation = linear
sources_k3_interpolation = cubic #linear
tau_start_evolution_song = 0
start_small_k_at_tau_c_over_tau_h_song = 0.001
start_large_k_at_tau_h_over_tau_k_song = 0.04
sources2_k_sampling = {}
k_min_custom_song = {}
k_max_custom_song = {}
k_size_custom_song = {}"""

    ini="./matter_{}.ini".format(params.key_song)
    pre="./matter_{}.pre".format(params.key_song)
    
    file = open(ini, "w")
    if len(params.z_song)>1: 
        file.write(ini_file.format(params.A_s,params.n_s,params.h,params.omega_b*params.h**2,
            params.omega_cdm*params.h**2,params.omega_k,params.fnl,str(params.z_song[0])+','+str(params.z_song[1])))
    else: 
        file.write(ini_file.format(params.A_s,params.n_s,params.h,params.omega_b*params.h**2,
            params.omega_cdm*params.h**2,params.omega_k,params.fnl,params.z))
    file.close()

    file = open(pre, "w")
    file.write(pre_file.format('smart',int(params.N_song_k3),opt,hkmin,hkmax,int(params.N_song_k12)))
    file.close()

    os.system("./song "+ini+' '+pre)
    os.system("mv "+params.song_path+'output/sources_song_z000.dat '+params.song_path+"output/sources_song_z000_{}.dat".format(params.key_song))

    if len(params.z_song)>1: 
        os.system("mv "+params.song_path+'output/sources_song_z001.dat '+params.song_path+"output/sources_song_z001_{}.dat".format(params.key_song))

def song_output(hkmax,hkmin,opt,force):
    ''' Once song has run, this function load the output by using  
        songy (see song/python/songy.py) routine FixedTauFile. 
        It return the needed output:
            -song.get_source(b'delta_cdm') = song :second order kernel mulptiply by two transfer functions i.e.
                                                K(k1,k2,k3)*T_delta(k1)*T_delta(k2) in the expression
                                             int_k1_k2 (K(k1,k2,k3) T_delta(k1) T_delta(k2) zeta(k1) zeta(k2))
            -song.tau conformal time corresponding to the redshift. It is needed to get the velocity potential (dK/dtau)
            -song.k1, song.k2, song.k3: grie of mode
            -song.flatidx: Song output shape is weird ! see song/python/songy.py
            -dk12,dk3: step of the grid
        '''

    filename='sources_song_z000_{}.dat'.format(params.key_song)
    if not os.path.isfile(params.song_path+'output/'+filename) :
        print(params.song_path+'output/{} not found'.format(filename))
        print('===========================================================================================')
        run_song(hkmax,hkmin,opt)
    elif force:
        print('force running SONG')
        os.system("rm "+params.song_path+'output/{} not found'.format(filename))
        print('===========================================================================================')
        run_song(hkmax,hkmin,opt)

    print('===========================================================================================')
    print('loading '+params.song_path+'output/{}'.format(filename))
    song=s.FixedTauFile(params.song_path+'output/'+filename)

    if len(params.source)==0 and (len(song.k1)!=params.N_song_k12  \
            or np.min(song.k1)!=hkmin or np.max(song.k1)!=hkmax):
        print('The output '+params.song_path+'output/ found does not have the right shape or hkmax/hkmin')
        print('SONG N_song_k1={}, you ask {}'.format(len(song.k1),params.N_song_k12))
        print('SONG N_song_k3={}, you ask {}'.format(len(song.k3[0]),params.N_song_k3))
        print('SONG hkmin={}, you ask {}'.format(np.min(song.k1),hkmin))
        print('SONG hkmax={}, you ask {}'.format(np.max(song.k1),hkmax))
        
        print('===========================================================================================')
        #run_song(hkmax,hkmin,opt)
        #print('loading '+params.song_path+'output/{}'.format(filename))
        #song=s.FixedTauFile(params.song_path+'output/'+filename)

    dk12=song.k1[1]-song.k1[0]
    #dk3=np.diff(song.k3)[:,0]
     
    dk3=np.array([])
    for k1_ind,k1 in enumerate(song.k1):
        for k2_ind,k2 in enumerate(song.k2[k1_ind]):
            k3_ind=song.flatidx[k1_ind,k2_ind]
            dk3=np.append(dk3,song.k3[k3_ind][2]-song.k3[k3_ind][1])

    k3=np.concatenate(song.k3)
    k3sizes_cumsum = np.zeros(len(song.k3sizes_cumsum)+2,dtype=int)
    k3sizes_cumsum[1:-1]=song.k3sizes_cumsum
    k3sizes_cumsum[-1]  =len(k3)
    return np.concatenate(song.get_source(b'delta_cdm')),song.tau,song.k1,np.concatenate(song.k2),k3,song.flatidx,dk12,dk3,k3sizes_cumsum

def song_main(hkmax,hkmin,opt,force=False,dSONG=False):
    '''Main function for SONG '''

    source,tau,k1,k2,k3,flatidx,dk12,dk3,k3sizes_cumsum=song_output(hkmax,hkmin,opt,force)
    if dSONG:
        s2="sources_song_z001_{}.dat".format(params.key_song)
        source1,tau1,_,_,_,_,_,_,_=song_output(hkmax,hkmin,opt,force,s2)
        return source,k1/params.h,k2/params.h,k3/params.h,flatidx,dk12/params.h,dk3/params.h,source1,tau1-tau,k3sizes_cumsum
    else:
        return source,k1/params.h,k2/params.h,k3/params.h,flatidx,dk12/params.h,dk3/params.h,k3sizes_cumsum

####################################################################################### 
####################################################################################### first order transfert fct
def trans():
    '''
       This function returns the Primordial power spectrum, the transfer functions of delta_cdm and phi, and 
       derivative of the last transfer function. 

        -Primordial power spectrum: Primordial = A_s(k/k_0)**(ns-1) / (k**3/(2*np.pi**2)). 
        -delta_cdm transfer function: tr_delta_cdm(k,z)*zeta(k)=delta_cdm(k,z)
        -potential transfer function: tr_phi(z,k)*zeta(k)=phi(z,k)
        '''
    if len(params.source)==0:
        key='N1_{}_smart_kmin{:.1e}_kmax{:.1e}'.format(params.N_song_k12,params.kmin,params.kmax_song)
    else:
        key=params.source

    song=s.FixedTauFile(params.song_path+"output/sources_song_z000_{}.dat".format(key))
    k=song.first_order_sources['k']/params.h

    tr_delta_cdm=song.first_order_sources[b'delta_cdm']     
    tr_delta_b  =song.first_order_sources[b'delta_b']     
    tr_phi= (params.omega_b/(params.omega_b+params.omega_cdm)*tr_delta_b \
            + params.omega_cdm/(params.omega_b+params.omega_cdm)*tr_delta_cdm)*(-3*H**2/2)/(k**2+3*H**2) 
    
    dk=np.diff(np.append(k,k[-1]*2-k[-2]))
    dT=np.diff(np.append(tr_phi,tr_phi[-1]*2-tr_phi[-2]))

    return np.array([k,tr_delta_cdm]),np.array([k,tr_phi]),np.array([k,dT/dk])

####################################################################################### 
####################################################################################### First order stochastic potential 
def zeta_realisation(k_grid):
    '''
        Generate the linear curvature perturbation field (N//2+1,N,N) at redshift z in half of Fourier space. 
        The reality condition ensure the other half.
        The computation is in 3 steps:
            -compute the modulus of k in the grid (k)
            -interpolate transfer function and primordial power spectrum tr=T(k) and P=P(k)
            -randomly draw the real/imaginary part of the primordial curvature zeta following a Gaussian PDF with std=sqrt(P(k)/2)
        '''
    def random (k):
        P= params.A_s*(k/(params.k_pivot/params.h))**(params.n_s-1)/k**3*2*np.pi**2

        zeta_ini_Re=np.random.normal(0,N**3*np.sqrt(P/2*params.kmin**3/(2*np.pi)**3),k.shape) # https://nms.kcl.ac.uk/eugene.lim/AdvCos/lecture2.pdf
        zeta_ini_Im=np.random.normal(0,N**3*np.sqrt(P/2*params.kmin**3/(2*np.pi)**3),k.shape)  

        # equivalent :
        #rho =  np.random.normal(0,N**3*np.sqrt(P*dk**3/(2*np.pi)**3),k.shape)
        #phase = np.random.uniform(0,2*np.pi,k.shape)
        #zeta_ini_Re=rho*np.cos(phase)
        #zeta_ini_Im=rho*np.sin(phase)
        return  zeta_ini_Re+zeta_ini_Im*1j

    zeta = np.zeros((N//2+1,N,N),dtype=np.complex64)

    # fill density ignoring the plan z=0 (zeta[0])
    k=np.sqrt(k_grid[0][N//2+1:]**2+k_grid[1]**2+k_grid[2]**2)
    zeta[1:,:,:]= random (k) 
    # fill half of the plan z=0 ignoring the line (z=0,y=N//2)
    k=np.sqrt(k_grid[0][N//2+1:]**2+k_grid[1]**2)[:,:,0]
    zeta[0,N//2+1:]=random(k) 
    # fill half of the line (z=0,y=N//2)
    k=k_grid[0][:,0,0][N//2+1:]
    zeta[0,N//2,N//2+1:]=random(k) 


    # Even N in real space give a N+1 FFT grid with symmetries !
    zeta[1:-1,-1,1:-1]=zeta[1:-1,0,1:-1]  #z&x Plan
    zeta[1:-1,1:-1,-1]=zeta[1:-1,1:-1,0]  #z&y Plan

    # Zmax plan Surfaces 
    zeta[-1,1:N//2,1:N//2]   =np.conjugate(zeta[-1,N//2+1:-1,N//2+1:-1][::-1,::-1]) 
    zeta[-1,N//2+1:-1,1:N//2]=np.conjugate(zeta[-1,1:N//2,N//2+1:-1][::-1,::-1])     

    # Zmax plan lines X constant and Y constant
    zeta[-1,N//2,1:N//2]=np.conjugate(zeta[-1,N//2,N//2+1:-1][::-1])
    zeta[-1,1:N//2,N//2]=np.conjugate(zeta[-1,N//2+1:-1,N//2][::-1])

    r=zeta[:-1,-1,0] # All edges (x=0,y=0),(x=0,y=-1),(x=-1,y=0) and (x=-1,y=-1) are equal 
    zeta[:-1,-1,-1],zeta[:-1,0,0],zeta[:-1,0,-1]=r,r,r

    r=zeta[-1,0,1:N//2]  # Zmax edges sym with Y constant
    zeta[-1,-1,1:N//2],zeta[-1,0,N//2+1:-1],zeta[-1,-1,N//2+1:-1]=r,np.conjugate(r[::-1]),np.conjugate(r[::-1])

    r=zeta[-1,1:N//2,0]# Zmax edges sym with X constant
    zeta[-1,1:N//2,-1],zeta[-1,N//2+1:-1,0],zeta[-1,N//2+1:-1,-1]=r,np.conjugate(r[::-1]),np.conjugate(r[::-1])

    r=zeta[-1,0,0].real    # Zmax plan corners all equal and real 
    zeta[-1,0,0],zeta[-1,-1,0],zeta[-1,-1,-1],zeta[-1,0,-1]=r,r,r,r

    r=zeta[-1,N//2,0].real # Zmax plan: middle point of edges
    zeta[-1,N//2,0],zeta[-1,N//2,-1]=r,r
    r=zeta[-1,0,N//2].real 
    zeta[-1,0,N//2],zeta[-1,-1,N//2]=r,r

    # Zmax middle point real
    zeta[-1,N//2,N//2]=zeta[-1,N//2,N//2].real

    # z=0 Plan
    zeta[0,N//2,-1]=zeta[0,N//2,-1].real

    zeta[0,-1,N//2]=zeta[0,-1,N//2].real

    zeta[0,N//2:-1,-1]=zeta[0,N//2:-1,0]
    zeta[0,-1,N//2+1:-1]=np.conjugate(zeta[0,-1,1:N//2][::-1])

    r=zeta[0,-1,0].real
    zeta[0,-1,0],zeta[0,-1,-1]=r,r

    zeta[0,:N//2]     =np.conjugate(zeta[0,N//2+1:][::-1,::-1])
    zeta[0,N//2,:N//2]=np.conjugate(zeta[0,N//2,N//2+1:][::-1])
    return zeta

####################################################################################### 
####################################################################################### From initial potential to displacement field

@jit(nopython=True,fastmath=True,parallel=False)
def order1(k):
    '''Compute the first order quantities X1/delta1 at a given k. 
       X being: potential phi1 (==psi1), displacement field xi1, velocity v1
       See equation (36) of the note.
        '''
    phi1=-3*H**2 / 2 / (3*H**2+k**2)
    xi1= (1-3*phi1)/k**2
    v1=-2/3/H*phi1 
    return phi1,xi1,v1

@jit(nopython=True,fastmath=True,parallel=params.paral)
def song2xi_numba(H,song,k1,k3,flatidx,d2,xi2,phi2,phi2p,chi2,v2,k3sizes_cumsum):
    for ind1 in prange(len(k1)):
        if ind1%10==0: print(ind1)
        kk1=k1[ind1]
        for ind2 in prange(len(k1[:ind1+1])):
            kk2=k1[ind2]

            iii=k3sizes_cumsum[flatidx[ind1,ind2]-1]
            jjj=k3sizes_cumsum[flatidx[ind1,ind2]]
            kk3=k3[iii: jjj]
            
            #kk3 = k3[iii]

            phi1_k1,xi1_k1,v1_k1 = order1(kk1)
            phi1_k2,xi1_k2,v1_k2 = order1(kk2)


            k1dk2=(kk3**2-kk1**2-kk2**2)/2

            d2[iii: jjj] = song[iii: jjj] - k1dk2*v1_k1*v1_k2

            chi2[iii: jjj]=3./2/kk3**4*( 2*(kk1**2+kk2**2)*k1dk2/3+k1dk2**2/3+kk1**2*kk2**2 )\
                 *(3*H**2*v1_k1*v1_k2+2*phi1_k1*phi1_k2)

            phi2p[iii: jjj]=(-3*H**2*k1dk2*v1_k1*v1_k2-2*kk3**2*chi2[iii: jjj]\
                                            +k1dk2*phi1_k1*phi1_k2)/21/H

            phi2[iii: jjj] = (3*H**2/2/(3*H**2+kk3**2))*(-d2[iii: jjj]+2*chi2[iii: jjj]+ \
                    (2+k1dk2/3/H**2-2./3/H**2*(kk1**2+kk2**2))*phi1_k1*phi1_k2\
                    -2*phi2p[iii:jjj]/H )

            xi2[iii:jjj]  = 1/kk3**2*(d2[iii:jjj]-3*phi2[iii:jjj]-3./2*\
                    (kk2**2*phi1_k1*xi1_k2+kk1**2*phi1_k2*xi1_k1)\
                    +1./2*k1dk2*v1_k1*v1_k2-9./2*phi1_k1*phi1_k2+1./2*(kk1**2*kk2**2-k1dk2**2)*xi1_k1*xi1_k2)

            v2[iii:jjj]=2./3 * (-(phi2-chi2)[iii:jjj]/H - phi2p[iii:jjj]/H**2 + phi1_k1*phi1_k2/H + v1_k1*phi1_k2/2  \
                            -(k1dk2*(kk1**2+kk2**2)+ 2*kk1**2*kk2**2 )* xi1_k1*v1_k2/2/kk3**2 ) 
    return d2,xi2,phi2,phi2p,chi2, v2

def song2xi(song,k1,k2,k3,flatidx,H,k3sizes_cumsum):
    '''Compute the second order displacement field xi2:
       input: 
            -song is the output of SONG: the second order density kernel
            -k1,k2,k are the SONG output mode modulus k1,k2,k3
            -flatidx is the output of SONG flatidx
       See note for the detailed computation. 

            -k1dk2 is scalar product of k1 and k2
            -phi2p is the time derivative of the second order potential phi
        '''
    print('Computing all second order kernels')
    d2   =np.zeros_like(song)
    xi2  =np.zeros_like(song)
    phi2 =np.zeros_like(song)
    phi2p=np.zeros_like(song)
    chi2 =np.zeros_like(song)
    v2   =np.zeros_like(song)

    print(' Number of step {}'.format(N))
    d2,xi2,phi2,phi2p,chi2, v2=song2xi_numba(H,song,k1,k3,flatidx,d2,xi2,phi2,phi2p,chi2,v2,k3sizes_cumsum)
    return {'delta_song':song,'delta': d2,'xi':xi2,'phi':phi2,'phip':phi2p,'chi':chi2, 'v':v2}
    

def zeta2fields(field,zeta,k_grid,tr_delta_cdm=0,tr_phi=0):
    '''Compute the whole first order stocastic field from the first order density 
       computed in zeta_realisation().
       field can be 'phi','psi','xi','v','chi'.
        '''
    k=(np.sqrt(k_grid[0][N//2:]**2+k_grid[1]**2+k_grid[2]**2))

    tr_d=np.interp(k,tr_delta_cdm[0],tr_delta_cdm[1])
    tr_p=np.interp(k,tr_phi[0],tr_phi[1])

    if field in ['delta','delta_song']:
        return zeta*tr_d
    elif field=='xi':
        mask=np.where(k==0)
        k[mask]=1
        xi=zeta*(tr_d-3*tr_p)/k**2
        xi[mask]=0+0*1j
        return xi
    elif field=='v':
        return -zeta*2*tr_p/3/H
    elif field=='chi':
        return 0
    elif field=='phi':
        return tr_p*zeta
    else:
        print ('input field not understood')


def plot_staff():
    ## Phi transfer function comparision with CLASS
    #plt.figure()
    #claSS=np.loadtxt('../class_public/output/myclasstk.dat')
    #plt.loglog(claSS[:,0],np.abs(claSS[:,3]))
    #plt.loglog(k,np.abs(tr_delta_cdm))
    #plt.show()

    # Power spectrum comparison with Gevolution

    #plt.figure()
    #plt.loglog(k,Primordial*(omega_b/(omega_b+omega_cdm)*tr_delta_b + omega_cdm/(omega_b+omega_cdm)*tr_delta_cdm)**2)
    #claSS=np.loadtxt('../class_public/output/myclasspk.dat')
    #plt.loglog(claSS[:,0],np.abs(claSS[:,1]))
    ##gev=np.loadtxt('gevolution-1.2/output/lcdm_pk000_phi.dat')
    ##plt.loglog(gev[:,0],gev[:,1])
    #plt.show()

    #################################################################Pure class
    #claSS=np.loadtxt('../class_public/output/myclasstk.dat')
    #k=claSS[:,0]
    #Primordial= A_s*(k/(k_pivot/h))**(n_s-1)/k**3*2*np.pi**2
    #tr_delta_cdm=claSS[:,3]
    #tr_phi=claSS[:,7]

    fig, axs = plt.subplots(2)

    gev=np.loadtxt('gevolution-1.2/output/Pk_basic000_phi.dat')
    axs[0].loglog(gev[:,0],gev[:,1],label='basic')
    gev=np.loadtxt('gevolution-1.2/output/Pk_SONG000_phi.dat')
    axs[0].loglog(gev[:,0],gev[:,1],label='SONG')
    #axs[0].loglog(k,Primordial*k**3/2/np.pi**2 *tr_phi**2,label='Analytic')
    #axs[0].legend()
    axs[0].set_xlabel('k')
    axs[0].set_ylabel('Power spectra')
    #plt.show()

    #P=np.interp(gev[:,0],k,Primordial*k**3/2/np.pi**2 *tr_phi**2)
    #gev=np.loadtxt('gevolution-1.2/output/Pk_basic000_phi.dat')
    #axs[1].loglog(gev[:,0],np.abs(gev[:,1]-P)/P,label='basic')
    #gev=np.loadtxt('gevolution-1.2/output/Pk_SONG000_phi.dat')
    #axs[1].loglog(gev[:,0],np.abs(gev[:,1]-P)/P,label='SONG')
    #axs[1].legend()
    #axs[1].set_xlabel('k')
    #axs[1].set_ylabel('errors')
    #plt.show()


def we(kz,ky,kx,di,p):
    """debuging staff"""
    import pylab as plt
    from matplotlib.colors import LogNorm
    TT  = di[kz[0]:kz[1],ky[0]:ky[1],kx[0]:kx[1]].ravel()
    vu=np.zeros_like(di)
    ind=0
    for i in TT:
        if i in di:
            w=np.where(i==di)
            vu[w]=i

    #for i in range(N//2+1):
    for i in [p]:
        plt.figure()
        plt.imshow(np.abs(vu[i].real))#,norm=LogNorm())
    #plt.imshow(vu)
    plt.show()
    return vu

####################################################################################### 
####################################################################################### Kernels
Nopython=True
@jit(nopython=Nopython,fastmath=True)
def squeezed_term(k,trans,dTdk):
    '''
        Interpolate the squeezed term in the VR+ analytic approximation. 
        This function is used by Kernel() for the term gamma. To get the VR approximation,
        this function has to return 0
        '''
    T=np.interp(k,trans[0],trans[1])
    dT=np.interp(k,dTdk[0],dTdk[1])
    return k/T*dT

@jit(nopython=Nopython,fastmath=True)
def Kernel_analytic(k1 ,k2 ,k3 ,trans,dTdk ,kk11,kk3,   dk,     dk3,     dk12,k3sizes_cumsum):
    '''
        Analytic kernel VR+ approximation following equation 5.1 of 1602.05933.
        The term dividedby is implicit in 1602.05933 from equation 3.7 (VR definition)
        '''
    Hk3=H/k3
    alpha=2./7+59./14.*Hk3**2+45./2*Hk3**4
    beta =1.-0.5*Hk3**2+54.*Hk3**4
    gamma=-3./2*Hk3**2+9./2.*Hk3**4 
    gamma=gamma-5./4*(Hk3**2+3*Hk3**4)*squeezed_term(k3,trans,dTdk)
    k1k2=(k3**2-k1**2-k2**2)/2/k1/k2
    trk1=np.interp(k1,trans[0],trans[1])
    trk2=np.interp(k2,trans[0],trans[1])
    return 4*trk1*trk2*k1**2*k2**2*(beta-alpha+beta/2*k1k2*(k2/k1+k1/k2)+alpha*k1k2**2+gamma*(k1/k2-k2/k1)**2)/9/H**4

@jit(nopython=Nopython,fastmath=True)
def interp_nearest(kk1,kk2,kk3,K,flatidx,k1,k3,dk,dk3,dk12):
    out=np.zeros((len(kk1)))
    for ind in range(len(kk1)):
        if kk1[ind]==0 or kk2[ind]==0:
            continue
        pkk1,pkk2 = np.int(np.around((kk1[ind]-dk)/dk12)) , np.int(np.around((kk2[ind]-dk)/dk12))
        pkk1,pkk2 = max(pkk1,pkk2),min(pkk1,pkk2)
        if kk3>k3[flatidx[pkk1,pkk2]][-1]:
            pkk3=-1
        elif kk3<k3[flatidx[pkk1,pkk2]][0]:
            pkk3=0
        else:
            pkk3=np.int(np.around((kk3-k3[flatidx[pkk1,pkk2]][0])/dk3[flatidx[pkk1,pkk2]]))
        out[ind]=K[flatidx[pkk1,pkk2]][pkk3]
    return out 

@jit(nopython=Nopython,fastmath=True)
def interp_lin(kk1,kk2,kk3,K,flatidx,k1,k3,dk,dk3,dk12):

    def linear(pkk1,pkk2,kk3):
        pkk1,pkk2 = max(pkk1,pkk2),min(pkk1,pkk2)
        if kk3>k3[flatidx[pkk1,pkk2]][-1]:
            return K[flatidx[pkk1,pkk2]][-1]
        elif kk3<k3[flatidx[pkk1,pkk2]][0]:
            return K[flatidx[pkk1,pkk2]][0]
        else:
            pkk3=np.int((kk3-k3[flatidx[pkk1,pkk2]][0])/dk3[flatidx[pkk1,pkk2]])
            return ((k3[flatidx[pkk1,pkk2]][pkk3+1]-kk3)*K[flatidx[pkk1,pkk2]][pkk3]+\
                  (kk3-k3[flatidx[pkk1,pkk2]][pkk3])*K[flatidx[pkk1,pkk2]][pkk3+1])\
                       /dk3[flatidx[pkk1,pkk2]]

    out=np.zeros((len(kk1)))
    for ind in range(len(kk1)):
        if kk1[ind]==0 or kk2[ind]==0:
            continue
        pkk1,pkk2 = np.int((kk1[ind]-dk)/dk12) , np.int((kk2[ind]-dk)/dk12)

        K_lower_left =linear(pkk1  ,pkk2  ,kk3)
        K_lower_right=linear(pkk1+1,pkk2  ,kk3)
        K_upper_left =linear(pkk1  ,pkk2+1,kk3)
        K_upper_right=linear(pkk1+1,pkk2+1,kk3)
    
        A,B=k1[pkk1+1]-kk1[ind],kk1[ind]-k1[pkk1]
        K_lower= (A*K_lower_left+B*K_lower_right)/dk12
        K_upper= (A*K_upper_left+B*K_upper_right)/dk12

        out[ind]= ((k1[pkk2+1]-kk2[ind])*K_lower + (kk2[ind]-k1[pkk2])*K_upper)/dk12
    return out 

@jit(nopython=Nopython,fastmath=True)
def interp_linsmart(kk1,kk2,kk3,K,flatidx,k1,k3,dk,dk3,dk12,k3sizes_cumsum):

    def linear(pkk1,pkk2,kk3):
        pkk1,pkk2 = max(pkk1,pkk2),min(pkk1,pkk2)
        k3_ind=flatidx[pkk1,pkk2]
        iii=k3sizes_cumsum[k3_ind]
        jjj=k3sizes_cumsum[k3_ind+1]
        k3_list=k3[iii: jjj]
        K_list=K[iii: jjj]

        if kk3>=k3_list[-1]:
            return K_list[-1]
        elif kk3<=k3_list[0]:
            return K_list[0]
        elif kk3>k3_list[-2]:
            pkk3=len(k3_list)-2
            dkk3=k3_list[-1] - k3_list[-2]
        elif kk3<k3_list[1]:
            pkk3=0
            dkk3=k3_list[1]-k3_list[0]
        else:
            pkk3=np.int((kk3-k3_list[1])/dk3[k3_ind])+1
            dkk3=dk3[k3_ind]
        return ((k3_list[pkk3+1]-kk3)*K_list[pkk3]+\
                (kk3-k3_list[pkk3])*K_list[pkk3+1])/dkk3

    out=np.zeros((len(kk1)))
    for ind in range(len(kk1)):
        if kk1[ind]==0 or kk2[ind]==0:
            continue
        pkk1,pkk2 = np.int((kk1[ind]-dk)/dk12) , np.int((kk2[ind]-dk)/dk12)

        K_lower_left =linear(pkk1  ,pkk2  ,kk3)
        K_lower_right=linear(pkk1+1,pkk2  ,kk3)
        K_upper_left =linear(pkk1  ,pkk2+1,kk3)
        K_upper_right=linear(pkk1+1,pkk2+1,kk3)
    
        A,B=k1[pkk1+1]-kk1[ind],kk1[ind]-k1[pkk1]
        K_lower= (A*K_lower_left+B*K_lower_right)/dk12
        K_upper= (A*K_upper_left+B*K_upper_right)/dk12

        out[ind]= ((k1[pkk2+1]-kk2[ind])*K_lower + (kk2[ind]-k1[pkk2])*K_upper)/dk12
    return out 


####################################################################################### 
####################################################################################### intergration
@jit(nopython=Nopython,fastmath=True,parallel=False)
def compute(zeta,cc,kz,ky,kx,kmkz,kmky,kmkx,k_modulus,kern,k_norm,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum,reverse=True):
    zeta_k   = zeta[kz[0]:kz[1]    ,ky[0]:ky[1]    ,kx[0]:kx[1]].ravel()
    if reverse:
        kernel=kern(k_modulus[kz[0]:kz[1], ky[0]:ky[1],kx[0]:kx[1]].ravel()\
                     ,k_modulus[kmkz[0]:kmkz[1],kmky[0]:kmky[1],kmkx[0]:kmkx[1]][::-1,::-1,::-1].ravel()\
                     ,k_norm,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum)
        zeta_kmk = zeta[kmkz[0]:kmkz[1],kmky[0]:kmky[1],kmkx[0]:kmkx[1]]\
                             [::-1,::-1,::-1].ravel()
    else:
        kernel=kern(k_modulus[kz[0]:kz[1], ky[0]:ky[1],kx[0]:kx[1]].ravel()\
                     ,k_modulus[kmkz[0]:kmkz[1],kmky[0]:kmky[1],kmkx[0]:kmkx[1]].ravel()\
                     ,k_norm,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum)
        zeta_kmk = zeta[kmkz[0]:kmkz[1],kmky[0]:kmky[1],kmkx[0]:kmkx[1]].ravel()
    if cc==1:
        zeta_kmk=np.conjugate(zeta_kmk)
    elif cc==2:
        zeta_k=np.conjugate(zeta_k)
    elif cc==3:
        zeta_kmk=np.conjugate(zeta_kmk)
        zeta_k=np.conjugate(zeta_k)
    return np.nansum(zeta_k*zeta_kmk*kernel)


@jit(nopython=Nopython,fastmath=True,parallel=False)
def I2_out(N,kx,ky,kz,k_modulus,k_max,zeta,kern,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,\
            Nd2,k_norm,kz_min_lim,kz_max_lim1,kz_max_lim2,kx_min_lim,kx_max_lim,ky_min_lim,\
                        ky_max_lim,kym1,kym2,kxm1,kxm2,k3sizes_cumsum):
    Xpos,Xneg,Ypos,Yneg,Xd,Yd,XYpos,XYneg,XYd,Z,ZX,ZY,ZXY=0,0,0,0,0,0,0,0,0,0,0,0,0
    if kx-k_max<0 or kx+k_max>=N:

        if kx-k_max<0:
            kx_min,kx_max=N-(k_max-kx)-1,N-1
            kmkx_min,kmkx_max=Nd2+kx+1,Nd2+k_max+1
            kmkx_min_neg,kmkx_max_neg=Nd2-k_max,Nd2-kx

        if kx+k_max>=N:
            kx_min,kx_max=1,kx+k_max-N+2
            kmkx_min,kmkx_max=Nd2-k_max,kx-Nd2
            kmkx_min_neg,kmkx_max_neg=Nd2-kx+N,Nd2+k_max+1

        Xpos=compute(zeta,0,[kz_min_lim,kz_max_lim2],[ky_min_lim,ky_max_lim],
                       [kx_min,kx_max],[0,kz_max_lim2-kz_min_lim],[Nd2-kym2,Nd2+kym1+1],
                       [kmkx_min,kmkx_max],k_modulus,kern,k_norm,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum)

        Xneg=compute(zeta,1,[kz_max_lim2,kz_max_lim1],[ky_min_lim,ky_max_lim],
                       [kx_min,kx_max],[1,-kz_max_lim2+kz_max_lim1+1],[Nd2-kym1,Nd2+kym2+1],
                       [kmkx_min_neg,kmkx_max_neg],k_modulus,kern,k_norm,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum,reverse=False)
 
        if kz-k_max<0:
            if kx<Nd2:
                kx_min_kzneg,kx_max_kzneg=1,k_max-kx+1
            else:
                kx_min_kzneg,kx_max_kzneg=N-(kx+k_max-N)-2,N-1

            Xd=compute(zeta,2,[1,k_max-kz+1],[N-ky_max_lim,N-ky_min_lim],
                       [kx_min_kzneg,kx_max_kzneg],[kz_max_lim2-kz_min_lim,k_max+1],[Nd2-kym2,Nd2+kym1+1],
                       [kmkx_min,kmkx_max],k_modulus,kern,k_norm,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum,reverse=False)

    ###########################################################################################################################"
    if ky-k_max<0 or ky+k_max>=N:

        if ky-k_max<0:
            ky_min,ky_max=N-(k_max-ky)-1,N-1
            kmky_min,kmky_max=Nd2+ky+1,Nd2+k_max+1
            kmky_min_neg,kmky_max_neg=Nd2-k_max,Nd2-ky

        if ky+k_max>=N:
            ky_min,ky_max=1,ky+k_max-N+2
            kmky_min,kmky_max=Nd2-k_max,ky-N//2
            kmky_min_neg,kmky_max_neg=Nd2-ky+N,Nd2+k_max+1

        Ypos=compute(zeta,0,[kz_min_lim,kz_max_lim2],[ky_min,ky_max],
                       [kx_min_lim,kx_max_lim],[0,kz_max_lim2-kz_min_lim],[kmky_min,kmky_max],
                       [Nd2-kxm2,Nd2+kxm1+1],k_modulus,kern,k_norm,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum)

        Yneg=compute(zeta,1,[kz_max_lim2,kz_max_lim1],[ky_min,ky_max],
                       [kx_min_lim,kx_max_lim],[1,-kz_max_lim2+kz_max_lim1+1],[kmky_min_neg,kmky_max_neg],
                       [Nd2-kxm1,Nd2+kxm2+1],k_modulus,kern,k_norm,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum,reverse=False)

        if kz-k_max<0:
            if ky<Nd2:
                ky_min_kzneg,ky_max_kzneg=1,k_max-ky+1
            else:
                ky_min_kzneg,ky_max_kzneg=N-(ky+k_max-N)-2,N-1

            Yd=compute(zeta,2,[1,k_max-kz+1],[ky_min_kzneg,ky_max_kzneg],
                       [N-kx_max_lim,N-kx_min_lim],[kz_max_lim2-kz_min_lim,k_max+1],[kmky_min,kmky_max],
                       [Nd2-kxm2,Nd2+kxm1+1],k_modulus,kern,k_norm,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum,reverse=False)

    #########################################################################################################################
    if ky-k_max<0:
        ky_min_corner,ky_max_corner=1,k_max-ky+1
        kmky_min_corner,kmky_max_corner=Nd2-kym2,Nd2-kym2+k_max-ky
    if ky+k_max>=N:
        ky_min_corner,ky_max_corner=N-(k_max+ky-N)-2,N-1
        kmky_min_corner,kmky_max_corner=Nd2+kym1-(ky+k_max-N),Nd2+kym1+1
    if kx-k_max<0:
        kx_min_corner,kx_max_corner=1,k_max-kx+1
        kmkx_min_corner,kmkx_max_corner=Nd2-kxm2,Nd2-kxm2+k_max-kx
    if kx+k_max>=N:
        kx_min_corner,kx_max_corner=N-(k_max+kx-N)-2,N-1
        kmkx_min_corner,kmkx_max_corner=Nd2+kxm1-(kx+k_max-N),Nd2+kxm1+1

    if (kx-k_max<0 or kx+k_max>=N) and (ky-k_max<0 or ky+k_max>=N):
        # x&y side pos 
        XYpos=compute(zeta,0,[kz_min_lim,kz_max_lim2],[N-ky_max_corner,N-ky_min_corner],
                     [N-kx_max_corner,N-kx_min_corner],[0,kz_max_lim2-kz_min_lim],[kmky_min,kmky_max],
                     [kmkx_min,kmkx_max],k_modulus,kern,k_norm,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum)

        if kz<Nd2+1:
            # x&y side neg
            XYneg=compute(zeta,1,[kz_max_lim2,kz_max_lim1],[ky_min,ky_max],
                     [kx_min,kx_max],[1,-kz_max_lim2+kz_max_lim1+1],[N-kmky_max,N-kmky_min],
                     [N-kmkx_max,N-kmkx_min],k_modulus,kern,k_norm,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum,reverse=False)

        if kz-k_max<0:
            # x&y side d
            XYd=compute(zeta,2,[1,k_max-kz+1],[N-ky_max,N-ky_min],
                     [N-kx_max,N-kx_min],[kz_max_lim2-kz_min_lim,k_max+1],[kmky_min,kmky_max],
                     [kmkx_min,kmkx_max],k_modulus,kern,k_norm,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum,reverse=False)

    if kz+k_max>Nd2:
        ky_min,ky_max=N-(ky_max_lim),N-(ky_min_lim)
        kx_min,kx_max=N-(kx_max_lim),N-(kx_min_lim)
        kz_min,kz_max=N-kz-k_max-1,N//2

        # pure z>N//2+1
        Z=compute(zeta,3,[kz_min,kz_max],[ky_min,ky_max],
                       [kx_min,kx_max],[1,kz+k_max-Nd2+1],[Nd2-kym1,Nd2+kym2+1],
                       [Nd2-kxm1,Nd2+kxm2+1],k_modulus,kern,k_norm,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum)

        if (kx-k_max<0 or kx+k_max>=N):
            ZX=compute(zeta,3,[kz_min,kz_max],[ky_min,ky_max],
                       [kx_min_corner,kx_max_corner],[1,kz+k_max-Nd2+1],[N-(Nd2+kym1+1),N-(Nd2-kym2)],
                       [kmkx_min_corner,kmkx_max_corner],k_modulus,kern,k_norm,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum)

        if (ky-k_max<0 or ky+k_max>=N):
            ZY=compute(zeta,3,[kz_min,kz_max],[ky_min_corner,ky_max_corner],
                       [kx_min,kx_max],[1,kz+k_max-N//2+1],[kmky_min_corner,kmky_max_corner],
                       [N-(Nd2+kxm1+1),N-(Nd2-kxm2)],k_modulus,kern,k_norm,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum)

        if (kx-k_max<0 or kx+k_max>=N) and (ky-k_max<0 or ky+k_max>=N):
            ZXY=compute(zeta,3,[kz_min,kz_max],[ky_min_corner,ky_max_corner],
                       [kx_min_corner,kx_max_corner],[1,kz+k_max-Nd2+1],[kmky_min_corner,kmky_max_corner],
                       [kmkx_min_corner,kmkx_max_corner],k_modulus,kern,k_norm,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum)

    return Xpos+Xneg+Ypos+Yneg+Xd+Yd+XYpos+XYneg+XYd+Z+ZX+ZY+ZXY

@jit(nopython=Nopython,fastmath=True,parallel=params.paral)
def I2_LL(Int,N,xidx,yidx,zidx,k_modulus,k_max,zeta,kern,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum):
    '''
        Compute LL integral with cut-off k_max. Here k_max is an index defined in 
        function integre(): k_max=int(np.where(klin==k_max)[0])-N//2.
        If k_max=N//2, this function return the raw integral. 
        input:
            -Int: integral result array(N//2+1,N,N)
            -N: size of grid 
            -xidx,yidx,zidx: index of the k-grid where the integral is computed. As for zeta computed in zeta_realisation, 
                             this splitting allow this integral to be computed in different part (Volume+plan+line). 
                             More details in function integre().
            -k_modulus: (half)whole grid modulus
            -k_max: cut-off index
            -zeta: linear density
            -kern,kern_arg1,kern_arg2: see integre() intput description

        Computation is split in two parts. For zeta(k1)*zeta(k-k1):

            -if 0<=k1_z<=k_z 
                zeta(k1)=zeta_pos (pos=positive because then k_z-k1_z>=0).
                In this particular case: zeta(k-k1)=zeta_pos[::-1]
            -if k1_z>k_z 
                zeta(k1)  =zeta_neg_k 
                zeta(k-k1)=np.conjugate(zeta_neg_kmk) (kmk for k minus k1) 
                    (Thanks to reality and because we only have in memory density for k_z>=0)

            a third part would be considered:
                -if k1_z<0:
                    let us call zeta_d_k and zeta_d_kmk the corresponding density
                    We can show that zeta_d_k=zeta_neg_kmk and zeta_neg_k=zeta_d_kmk

            Taking into account reality condition, we finally get:
            Int[kz,ky,kx]=(np.nansum(             zeta_pos     *zeta_pos[::-1]*kern_pos)\
                        +2*np.nansum(np.conjugate(zeta_neg_kmk)*zeta_neg_k    *kern_neg))

        '''
    for iikx in prange(len(xidx)):
        if iikx%10==0: print(iikx)
        kx=xidx[iikx]
        for iiky in  prange(len(yidx)):
            ky=yidx[iiky]
            for iikz in prange(len(zidx)):
                kz=zidx[iikz]

                k_norm=k_modulus[kz,ky,kx]

                kz_min_lim =max(0,kz-k_max)
                kz_max_lim1=min(kz+1,k_max+1)

                kx_min_lim=max(N//2-k_max,kx-k_max)
                kx_max_lim=min(kx+k_max+1,N//2+k_max+1)

                ky_min_lim=max(N//2-k_max,ky-k_max)
                ky_max_lim=min(ky+k_max+1,N//2+k_max+1)


                zeta_pos=np.copy(zeta[kz_min_lim:kz_max_lim1,\
                                      ky_min_lim:ky_max_lim,\
                                      kx_min_lim:kx_max_lim])

                #we([kz_min_lim,kz_max_lim1],[ky_min_lim,ky_max_lim],[kx_min_lim,kx_max_lim],zeta,kz)

                kk_pos_k=k_modulus[kz_min_lim:kz_max_lim1, ky_min_lim:ky_max_lim, kx_min_lim:kx_max_lim]

                #if kz_min_lim<=kz<kz_max_lim1 and ky_min_lim<=ky<ky_max_lim and kx_min_lim<=kx<kx_max_lim:
                #    zeta_pos [kz-kz_min_lim,ky-ky_min_lim,kx-kx_min_lim]=0

                kern_pos=kern(kk_pos_k.ravel(),kk_pos_k[::-1,::-1,::-1].ravel(),\
                              k_norm,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum)
                I_pos=np.nansum(zeta_pos.ravel()*zeta_pos[::-1,::-1,::-1].ravel()*kern_pos)

                # if kz>k_max, the zeta_neg part vanishes
                if k_max-kz+1>0:
                    kz_max_lim2=k_max+1
                    zeta_neg_k=zeta[kz_max_lim1:kz_max_lim2,\
                                             ky_min_lim:ky_max_lim,\
                                             kx_min_lim:kx_max_lim].ravel()

                    kk_neg_k   =k_modulus   [kz_max_lim1:kz_max_lim2,\
                                             ky_min_lim:ky_max_lim,\
                                             kx_min_lim:kx_max_lim].ravel()

                    zeta_neg_kmk=zeta[1:k_max-kz+1,N-ky_max_lim:N-ky_min_lim,N-kx_max_lim:N-kx_min_lim].ravel()

                    kk_neg_kmk   =k_modulus   [1:k_max-kz+1,N-ky_max_lim:N-ky_min_lim,N-kx_max_lim:N-kx_min_lim].ravel()

                    kern_neg=kern(kk_neg_k,kk_neg_kmk\
                                         ,k_norm,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum)
                    
                    Int[kz,ky,kx]+=I_pos+2*np.nansum(np.conjugate(zeta_neg_kmk)\
                                             *zeta_neg_k*kern_neg)
                else:
                    Int[kz,ky,kx]+=I_pos
    return Int

@jit(nopython=Nopython,fastmath=True,parallel=params.paral)
def I2_LLout(Int,N,xidx,yidx,zidx,k_modulus,k_max,zeta,kern,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum):
    Nd2=N//2
    for iikx in prange(len(xidx)):
        if iikx%10==0: print(iikx)
        kx=xidx[iikx]
        for iiky in prange(len(yidx)):
            ky=yidx[iiky]
            for iikz in prange(len(zidx)):
                kz=zidx[iikz]

                k_norm=k_modulus[kz,ky,kx]

                kz_min_lim =max(0,kz-k_max)
                kz_max_lim1=min(kz+1,k_max+1)
                kz_max_lim2=kz+1

                kx_min_lim=max(N//2-k_max,kx-k_max)
                kx_max_lim=min(kx+k_max+1,N//2+k_max+1)
                ky_min_lim=max(N//2-k_max,ky-k_max)
                ky_max_lim=min(ky+k_max+1,N//2+k_max+1)

                kym1=-ky_min_lim+ky
                kym2= ky_max_lim-ky-1
                kxm1=-kx_min_lim+kx
                kxm2= kx_max_lim-kx-1

                Int[kz,ky,kx]+=I2_out(N,kx,ky,kz,k_modulus,k_max,zeta,kern,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,\
                        Nd2,k_norm,kz_min_lim,kz_max_lim1,kz_max_lim2,kx_min_lim,kx_max_lim,ky_min_lim,\
                        ky_max_lim,kym1,kym2,kxm1,kxm2,k3sizes_cumsum)
    return Int


@jit(nopython=Nopython,fastmath=True,parallel=params.paral)
def I2_SL(Int,N,xidx,yidx,zidx,k_modulus,k_max,zeta,kern,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum):
    '''
        Compute SL integral, see function I2_LL() for description of inputs
        
        Computation is split in three parts. For zeta(k1)*zeta(k-k1):

            -if 0<=k1_z<=k_z 
                zeta(k1)=zeta_pos (pos=positive because then k_z-k1_z>=0).
                In this particular case: zeta(k-k1)=zeta_pos[::-1]
            -if k1_z>k_z 
                zeta(k1)  =zeta_neg_k 
                zeta(k-k1)=np.conjugate(zeta_neg_kmk) (kmk for k minus k1) 
                    (Thanks to reality and because we only have in memory density for k_z>=0)
            -if k1_z<0:
                zeta(k1)  =np.conjugate(zeta_d_k )
                    (Thanks to reality and because we only have in memory density for k_z>=0)
                zeta(k-k1)=zeta_d_kmk 

            Taking into account reality condition, we finally get:
 
            Int[kz,ky,kx]=(np.nansum(             zeta_pos_kmk[::-1]*zeta_pos_k            *kern_bis_pos)\
                          +np.nansum(np.conjugate(zeta_neg_kmk)     *zeta_neg_k            *kern_bis_neg)\
                          +np.nansum(             zeta_d_kmk        *np.conjugate(zeta_d_k)*kern_bis_d))
        '''
    Nd2=N//2
    for iikx in prange(len(xidx)):
        if iikx%10==0: print(iikx)
        kx=xidx[iikx]
        for iiky in prange(len(yidx)):
            ky=yidx[iiky]
            for iikz in prange(len(zidx)):
                kz=zidx[iikz]

                k_norm=k_modulus[kz,ky,kx]

                kz_min_lim=max(0,kz-k_max)
                kz_max_lim1=min(kz+k_max+1,Nd2+1)
                kz_max_lim2=kz+1

                kx_min_lim=max(0,kx-k_max)
                kx_max_lim=min(kx+k_max+1,N)
                ky_min_lim=max(0,ky-k_max)
                ky_max_lim=min(ky+k_max+1,N)

                kym1=-ky_min_lim+ky
                kym2= ky_max_lim-ky-1
                kxm1=-kx_min_lim+kx
                kxm2= kx_max_lim-kx-1

                I_out=I2_out(N,kx,ky,kz,k_modulus,k_max,zeta,kern,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,\
                        Nd2,k_norm,kz_min_lim,kz_max_lim1,kz_max_lim2,kx_min_lim,kx_max_lim,ky_min_lim,\
                        ky_max_lim,kym1,kym2,kxm1,kxm2,k3sizes_cumsum)

                zeta_pos_k=np.copy(zeta[kz_min_lim:kz_max_lim2,\
                                                 ky_min_lim:ky_max_lim,
                                                 kx_min_lim:kx_max_lim]).ravel()
                kk_pos_k=k_modulus[kz_min_lim:kz_max_lim2,\
                                   ky_min_lim:ky_max_lim,\
                                   kx_min_lim:kx_max_lim].ravel()

                zeta_neg_k=np.copy(zeta[kz_max_lim2:kz_max_lim1,ky_min_lim:ky_max_lim,kx_min_lim:kx_max_lim]).ravel()
                kk_neg_k  =k_modulus[kz_max_lim2:kz_max_lim1,ky_min_lim:ky_max_lim,kx_min_lim:kx_max_lim].ravel()

                kzlower= k_max+kz+1-kz_max_lim1 

                zeta_pos_kmk =np.copy(zeta[:kz_max_lim2-kz_min_lim,\
                                                  Nd2-kym2:Nd2+kym1+1,\
                                                  Nd2-kxm2:Nd2+kxm1+1][::-1,::-1,::-1]).ravel()
                kk_pos_kmk    =        k_modulus   [:kz_max_lim2-kz_min_lim,\
                                                  Nd2-kym2:Nd2+kym1+1,\
                                                  Nd2-kxm2:Nd2+kxm1+1][::-1,::-1,::-1].ravel()
                zeta_neg_kmk =np.copy(zeta[1:k_max+1-kzlower,\
                                                  Nd2-kym1:Nd2+kym2+1,
                                                  Nd2-kxm1:Nd2+kxm2+1]).ravel()
                kk_neg_kmk    =        k_modulus   [1:k_max+1-kzlower,\
                                                  Nd2-kym1:Nd2+kym2+1,\
                                                  Nd2-kxm1:Nd2+kxm2+1].ravel()

                if kz-k_max<k_max+1:
                    
                    if Nd2-2*k_max<=kx<Nd2+2*k_max+1 and Nd2-2*k_max<=ky<Nd2+2*k_max+1:

                        kxupper=min(Nd2+k_max-(N-kx_max_lim)+1,kx_max_lim-kx_min_lim)
                        kxlower=max(Nd2-k_max-(N-kx_max_lim),0)
                        kyupper=min(Nd2+k_max-(N-ky_max_lim)+1,ky_max_lim-ky_min_lim)
                        kylower=max(Nd2-k_max-(N-ky_max_lim),0)
                        kzupper=min(2*k_max-kz+1,k_max+1)

                        zeta_pos_k.reshape((kz_max_lim2-kz_min_lim,\
                                                ky_max_lim-ky_min_lim,\
                                                kx_max_lim-kx_min_lim))[:kzupper,kylower:kyupper,kxlower:kxupper]=0
                        if kz<k_max+1:
                            zeta_neg_k.reshape((kz_max_lim1-kz_max_lim2,ky_max_lim-ky_min_lim,kx_max_lim-kx_min_lim))\
                                            [:k_max-kz,kylower:kyupper,kxlower:kxupper]=0 

                kern_bis_pos=kern(kk_pos_k,kk_pos_kmk,k_norm,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum)

                if kz==Nd2:
                    Int[kz,ky,kx]=(np.nansum(zeta_pos_kmk*zeta_pos_k*kern_bis_pos))+I_out#*dk**3
                elif kz-k_max>=0:
                    kern_bis_neg=kern(kk_neg_k,kk_neg_kmk      ,k_norm,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum)
                    Int[kz,ky,kx]=(np.nansum(zeta_pos_kmk*zeta_pos_k*kern_bis_pos)\
                       +np.nansum(np.conjugate(zeta_neg_kmk)*zeta_neg_k*kern_bis_neg))+I_out
                else:
                    kern_bis_neg=kern(kk_neg_k,kk_neg_kmk      ,k_norm,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum)

                    zeta_d_k  =np.copy(zeta[1:k_max-kz+1,N-ky_max_lim:N-ky_min_lim,N-kx_max_lim:N-kx_min_lim].ravel())
                    zeta_d_kmk =np.copy(zeta[1:k_max-kz+1,Nd2-kym2:Nd2+kym1+1,Nd2-kxm2:Nd2+kxm1+1]).ravel()
                    kk_d_k  =k_modulus[1:k_max-kz+1,N-ky_max_lim:N-ky_min_lim,N-kx_max_lim:N-kx_min_lim].ravel()
                    kk_d_kmk=k_modulus[1:k_max-kz+1,Nd2-kym2:Nd2+kym1+1,Nd2-kxm2:Nd2+kxm1+1].ravel()
                    kern_bis_d=kern(kk_d_k,kk_d_kmk,k_norm,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum)

                    if Nd2-2*k_max<=kx<Nd2+2*k_max+1 and Nd2-2*k_max<=ky<Nd2+2*k_max+1:
                        kxupper=min(Nd2+k_max-(N-kx_max_lim)+1,kx_max_lim-kx_min_lim)
                        kxlower=max(Nd2-k_max-(N-kx_max_lim),0)
                        kyupper=min(Nd2+k_max-(N-ky_max_lim)+1,ky_max_lim-ky_min_lim)
                        kylower=max(Nd2-k_max-(N-ky_max_lim),0)
                        kzupper=min(2*k_max-kz+1,k_max+1)
                        
                        zeta_d_k.reshape((k_max-kz,ky_max_lim-ky_min_lim,kx_max_lim-kx_min_lim))\
                                               [:kzupper,kylower:kyupper,kxlower:kxupper]=0 

                    Int[kz,ky,kx]=(np.nansum(zeta_pos_kmk*zeta_pos_k*kern_bis_pos)\
                              +np.nansum(np.conjugate(zeta_neg_kmk)*zeta_neg_k*kern_bis_neg)\
                              +np.nansum(zeta_d_kmk*np.conjugate(zeta_d_k)*kern_bis_d))+I_out
    return Int

def integre(which,klin,N,k_max,kmax,klin_grid,zeta,kern,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum):
    '''
        inputs:
        -which: whether it is 'TT','LL', 'SL' or 'SS' to be computed. The 'SS' is not implemented yet
        -klin: one dimensional list of k coordinate to be considered
        -N: size of the grid
        -k_max: cut-off in h/Mpc^-1
        -klin_grid: sparse grid of kx,ky,kz
        -zeta: Primordial curvature
        -kern: which kernel function to use. Either interp_nearest or Kernel_analytic
                if interp_nearest:
                    kern_arg1 has to be the SONG output song and kern_arg2 has to be flatidx
                if Kernel_analytic:
                    kern_arg1 and kern_arg2 have top be the two last output of function trans():
                                tr_pi=potential transfert function and dTdk its derivativ wrt k. 
                                It will be interpolated in squeezed_term() 

        Computation is split in three part in the space like in zeta_realisation() 
        in order to compute only independent quantity regarding reality conditions

        -kz>=1
        -kz=0,ky>N//2
        -kz=0,ky=N//2,kx>N//2
        '''
    Int=np.zeros(zeta.shape,dtype=np.complex64)
    k_modulus=np.sqrt(klin_grid[0][N//2:]**2+klin_grid[1]**2+klin_grid[2]**2)
    k_max=int(np.where(klin==k_max)[0])-N//2
    kmax=int(np.where(klin==kmax)[0])-N//2
    print(' k_lambda_index={}'.format(k_max))
    klin=np.arange(len(klin))-N//2

    if which=='SL':
        print(' integre SL ...')
        for opt in range(3):
            if opt==0:
                print('     compute volume z in [1,N//2+1]')
                xidx=np.arange(N)
                yidx=np.arange(N)
                zidx=np.arange(1,N//2+1)
            elif opt==1:
                print('     compute plan z=0')
                xidx=np.arange(N)
                yidx=np.arange(N//2+1,N)
                zidx=np.arange(1)
            elif opt==2:
                print('     compute line z=0 y=N//2')
                xidx=np.arange(N//2+1,N)
                yidx=np.array([N//2])
                zidx=np.arange(1)
            print('     Number of step {}'.format(len(xidx)))

            I2_SL(Int,N,xidx,yidx,zidx,k_modulus,k_max,zeta,kern,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum)
    else:
        if which=='TT':
            print(' integre TT ...')
            upper=kmax*2+1
        elif which=='LL':
            print(' integre LL ...')
            upper=k_max*2+1
        elif which=='LL+':
            print(' integre LL+ ...')
            upper=k_max*2+1

        xidx1=np.where(np.abs(klin)<=upper)[0]
        yidx1=np.where(np.abs(klin)<=upper)[0]
        zidx1=np.where(np.logical_and(klin[N//2:]<=upper,klin[N//2:]>10**-15))[0]

        xidx2=np.where(np.abs(klin)<=upper)[0]
        yidx2=np.where(np.logical_and(np.abs(klin)<=upper,klin>0))[0]
        zidx2=np.arange(1)

        xidx3=np.where(np.logical_and(np.abs(klin)<=upper,klin>0))[0]
        yidx3=np.array([N//2])
        zidx3=np.arange(1)

        for opt in range(3):
            if opt==0:
                xidx=xidx1
                yidx=yidx1
                zidx=zidx1
                print('     compute volume z in [1,N//2+1]')
            elif opt==1:
                print('     compute plan z=0')
                xidx=xidx2
                yidx=yidx2
                zidx=zidx2
            elif opt==2:
                print('     compute line z=0 y=N//2')
                xidx=xidx3
                yidx=yidx3
                zidx=zidx3
            print('     Number of step {}'.format(len(xidx)))

            if  which=='LL':
                I2_LL(Int,N,xidx,yidx,zidx,k_modulus,k_max,zeta,kern,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum)
            elif  which=='TT' or which=='LL+':
                I2_LL(Int,N,xidx,yidx,zidx,k_modulus,kmax,zeta,kern,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum)
                I2_LLout(Int,N,xidx,yidx,zidx,k_modulus,kmax,zeta,kern,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum)
            else:
                print('what ?')
                return 0

    return Int/N**3

####################################################################################### 
####################################################################################### mode grid
def k_distrib(k_min,N,klbd,absolute=True):
    ''' Inputs:
            -k_min: Minimum mode to be consider. Setting k_min automatically set the step dk=k_min 
             because in order for k-k1 to be always on the grid k1, we need to include 0 and to have a 
             constant step dk. 
            -N size of the grid. In order to include 0. If it is not odd, we set N+=1 
                                            (the final fft return the right even N grid)
            -klbd: k_lambda:
                if absolute==True:
                    the function will return the closest in the grid
                else:
                    klbd is considered as being a ratio, return kL=k[N//2:][int(klbd*N//2)]
        output:
        klin_concat,kmax,N,dk,klambda
            -k: list of k coordinate
            -kmax: largest mode to be considered
            -N like input
            -k_min in float32
            -kL: actual k_lambda
                    
        '''
    if N%2==0:
        print('N has to be odd to include 0: N+=1')
        N+=1
    k=np.linspace(-(N//2)*k_min,N//2*k_min,N,dtype=np.float32)
    if absolute:
        idxL=np.where(np.abs(klbd-k[N//2:])==np.min(np.abs(klbd-k[N//2:])))[0]
        kL=k[N//2:][idxL][0]
    else:
        kL=k[N//2:][int(klbd*N//2)]
    return k,np.float32(N//2*k_min),N,np.float32(k_min),kL

def fft(field):
    '''This function performs the inverse Fourier transform. It uses the numpy function irfftn.
       The input array has first to be re-organized. 
       In this code, the array filed is organized this way field=(z=0:Nyquist,y=-Nyquist:0:Nyquist,x=-Nyquist:0:Nyquist)
       which means shape(field)=(N//2+1,N,N) (Reminder: in the code, N is always odd while N_input is even, N=N_input+1). 
       The python modules takes as an input an array organized as follow: 
            field=(x=0:Nyquist-1:-1:-Nyquist, y=0:Nyquist-1:-1:-Nyquist, z=0:Nyquist) which means shape(field)=(N//2+1,N-1,N-1)
            Note that -Nyquist=+Nyquist since N_input is even. 
        '''
    new_field=np.zeros((N//2+1,N-1,N-1),dtype=np.complex64)

    field[0,N//2,:N//2]=np.conjugate(field[0,N//2,N//2+1:][::-1])
    field[0,:N//2,:]   =np.conjugate(field[0,N//2+1:,:][::-1,::-1])

    new_field[:,N//2+1:,N//2+1:]=field[:,1:N//2,1:N//2]
    new_field[:,:N//2+1,:N//2+1]=field[:,N//2:,N//2:]

    new_field[:,:N//2+1,N//2+1:]=field[:,N//2:,1:N//2]
    new_field[:,N//2+1:,:N//2+1]=field[:,1:N//2,N//2:]

    return np.fft.irfftn(new_field.transpose(),(N-1,N-1,N-1)) 

def read_h5(run,field):
    f1 = h5py.File('{}/{}_{}1_0.h5'.format(params.gev_path,run,field), 'r')
    f2 = h5py.File('{}/{}_{}2_0.h5'.format(params.gev_path,run,field), 'r')
    dat1=np.array(f1['data'],dtype=np.float32)
    dat2=np.array(f2['data'],dtype=np.float32)
    return dat1,dat2

####################################################################################### 
####################################################################################### Gevolution place

def settings(kmin,setting_name='msetting.ini',ICg='SONG',disp_file='displacement.h5',vel_file='velocitypotential.h5',pot_file='phi.h5'):
    '''create a setting file with the global parameters used in the programme'''
    setx=r"""
template file = sc1_crystal.dat    
Tk file = class_tk.dat              
baryon treatment = ignore
seed = 42                         
correct displacement = yes       
k-domain = cube                
Courant factor      = 48.0  
T_cmb       = 2.7255
N_ur        = 3.046
time step limit     = 0.04         
gravity theory      = GR            
vector method       = parabolic    
output path         = output/
generic file base   = lcdm
Pk outputs          = phi
Pk bins             = 1024
output              = mPk
gauge               = Newtonian
P_k_ini type        = analytic_Pk
P_k_max_h/Mpc       = 192           
z_pk                = 100          
root                = class_
background_verbose  = 1
spectra_verbose     = 1
output_verbose      = 1

Pk redshifts        = {}
Pk file base        = {}
IC generator      = {}
displacement file = {}
velocity file     = {}
metric file       = {}
tiling factor =     {}                  
k_pivot =           {}           
A_s =               {}
n_s =               {}
h           =       {}
omega_b     =       {}
omega_cdm   =       {}
initial redshift    = {}
boxsize             = {} 
Ngrid               = {}"""

    file = open(params.gev_path+setting_name, "w")
    file.write(setx.format(params.z,'Pk_'+ICg,ICg,disp_file,vel_file,pot_file,params.N//4,params.k_pivot\
                    ,params.A_s,params.n_s,params.h,params.omega_b*params.h**2,params.omega_cdm*params.h**2\
                            ,params.z,2*np.pi/params.kmin,params.N))
    file.close()

########################################################################################################################## 
########################################################################################################################## 

def computePower(filename):
    ''' Measure Powerspectrum. The output is saved in a txt file in the folder
        specified by output in initializeGlobals.

        delta : array_like
            input data, shape = (gridsize, gridsize, gridsize)
        realization: integer

        file format: k | P0(k)
    '''
    MAS     = 'None' # Mass-assignment scheme, possible choices are: 'CIC' 'NGP' 'TSC' 'PCS' if the density field has not been generated with any of these set it to 'None'
    axis    = 0           # Axis along which redshift-space distortions will be implemented (only needed if do_RSD=True): 0, 1 or 2 for x-axis, y-axis or z-axis, respectively.
    BoxSize = 2*np.pi/params.kmin   # Mpc/h , This fixed the all the units in pylians3 measurements (k:{h/Mpc})
    #filename = params.song_path+'powerspectrum.dat'
    threads = 4           # Number of openmp threads to be used.
    verbose = False        # Whether print information on the status/progress of the calculation: True or False

    print('\n Computing power')

    delta1=h5py.File(params.gev_path+filename+'.h5', 'r')
    delta1=np.array(delta1['data'],dtype=np.float32)
    Pkpyl = PKL.Pk(delta1, BoxSize, axis, MAS, threads, verbose)
    np.savetxt(params.song_path+'ps_'+filename+'.dat',np.vstack([Pkpyl.k3D, Pkpyl.Pk[:,0]]).T)

########################################################################################################################## Main

def main():
    global N,H

    if params.bug:
        os.environ["NUMBA_DISABLE_JIT"] = "1"

    H=100*np.sqrt((params.omega_cdm+params.omega_b)*(1+params.z))/3/10**5 #song provide a slightly different H

    try:
        N=int(sys.argv[1])
        print('sys N={}'.format(N))
        klin_concat,kmax,N,dk,klambda=k_distrib(params.kmin,N,params.kl)      # Generate the list of mode coordinate (assume h/Mpc)
        print('re-define N_song and kmax_song')
        print(N,params.N_song,params.N)
        params.N_song=N*(params.N_song//params.N)                # Song grid precision
        params.kmax_song=2*N*params.kmin                       # maximum value for SONG
    except IndexError:
        print('params N={}'.format(params.N))
        klin_concat,kmax,N,dk,klambda=k_distrib(params.kmin,params.N,params.kl)  # Generate the list of mode coordinate (assume h/Mpc)
        
    print('kmin={:.10f}, kmax={:.10f}, klambda={:.10f}'.format(dk,kmax,klambda))
    print('kmin_song={:.10f}, kmax_song={:.10f}'.format(params.kmin,params.kmax_song))
    hkmax,hkmin=np.float32(params.kmax_song*params.h),np.float32(params.kmin*params.h)

    if params.run=='TT':
        params.key= params.run+'_N{}_kmin{:.1e}_kmax{:.1e}'.format(N,params.kmin,kmax)
    elif params.run=='ps':
        if params.measurement=='TT':
            params.key= params.measurement+'_N{}_kmin{:.1e}_kmax{:.1e}'.format(N,params.kmin,kmax)
        else:
            params.key= params.measurement+'_N{}_kmin{:.1e}_kmax{:.1e}_kl{:.1e}'.format(N,params.kmin,kmax,klambda)
    else:
        params.key= params.run+'_N{}_kmin{:.1e}_kmax{:.1e}_kl{:.1e}'.format(N,params.kmin,kmax,klambda)

    if len(params.source)==0:
        params.key_song='N1_{}_smart_kmin{:.1e}_kmax{:.1e}'.format(params.N_song_k12,params.kmin,params.kmax_song)
    else:
        params.key_song=params.source
    
    if params.run=='song':
        song,k1,k2,k3,flatidx,dk12,dk3,k3sizes_cumsum=song_main(hkmax,hkmin,'lin',True)  # Get song outputs
    else:
        song,k1,k2,k3,flatidx,dk12,dk3,k3sizes_cumsum=song_main(hkmax,hkmin,'lin',False)  # Get song outputs

    if params.run=='kernel':
        kernel_dict=song2xi(song,k1,k2,k3,flatidx,H,k3sizes_cumsum)             
        for f in params.field:                                                      
            np.save(params.song_path+params.key_song+'_{}'.format(f),kernel_dict[f])

    elif params.run in ['d3','d3+','SL','LL','TT','LL+']:
        transcdm,transphi,dTdk=trans()                                       # Get transfer functions and primordial power spectrum 
        k_grid_lin=np.array(np.meshgrid(klin_concat,klin_concat,klin_concat\
                             ,sparse=True,indexing='ij'),dtype=object)       # mode Grid with 'ij' indexing

        for realisation in range(params.N_realization):
            print('**************************************************')
            print('realisation {}'.format(realisation))
            zeta=zeta_realisation(k_grid_lin)                                    # first order stocastic density

            for f in params.field:                                                      #
                print('computing first order {}'.format(f))
                f1 = fft(zeta2fields(f,zeta,k_grid_lin,transcdm,transphi))              # first order displacement field
                hf = h5py.File(params.gev_path+params.key+'_{}1_{}.h5'.format(f,realisation), 'w')  # Save in h5 format 
                hf.create_dataset('data', data=f1)                                      # 
                hf.close()                                                              #

            if params.order==2:
                if params.field==['delta_song']:
                    kernel_dict={'delta_song':song}
                else:
                    kernel_dict=song2xi(song,k1,k2,k3,flatidx,H,k3sizes_cumsum)             

                if params.numeric:
                    kern_arg2 = flatidx
                    print('numeric kernel (song) interpolation {}'.format(params.interp))
                    if params.interp=='nearest':
                        kernel=interp_nearest
                    elif params.interp=='lin':
                        kernel=interp_linsmart
                    else:
                        print('interpolation do not exists: lin or nearest')
                else:
                    print('analytic kernel')
                    kernel=Kernel_analytic
                    kern_arg1,kern_arg2 = transphi,dTdk

                for f in params.field:
                    print('computing second order {}'.format(f))   
                    if params.numeric:
                        kern_arg1 = kernel_dict[f]

                    if params.run in ['SL','LL','TT','LL+']:
                        res=integre(params.run,klin_concat,N,klambda,kmax,k_grid_lin,\
                                zeta,kernel,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum)#Integration
                    elif params.run=='d3':
                        f_LL=integre('LL',klin_concat,N,klambda,kmax,k_grid_lin,\
                                zeta,kernel,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum)#Integration
                        f_SL=integre('SL',klin_concat,N,klambda,kmax,k_grid_lin,\
                                zeta,kernel,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum)#integration
                        res=f_LL+2*f_SL
                    elif params.run=='d3+':
                        f_LL=integre('LL+',klin_concat,N,klambda,kmax,k_grid_lin,\
                                zeta,kernel,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum)#Integration
                        f_SL=integre('SL',klin_concat,N,klambda,kmax,k_grid_lin,\
                                zeta,kernel,kern_arg1,kern_arg2,k1,k3,dk,dk3,dk12,k3sizes_cumsum)#integration
                        res=f_LL+2*f_SL

                    hf = h5py.File(params.gev_path+params.key+'_{}2_{}.h5'.format(f,realisation), 'w')  # Save in h5 format 
                    hf.create_dataset('data', data=fft(res))                                           # 
                    hf.close()

    if params.run=='gevolution':
        settings(params.kmin,'msetting.ini',ICg='SONG',disp_file='d3_xi1_0.h5',vel_file='d3_v1_0.h5',pot_file='d3_phi1_0.h5')
        settings(params.kmin,'gsetting.ini',ICg='basic',disp_file='',vel_file='',pot_file='')

    if params.run=='ps' or params.measurement!='':
        for realisation in range(params.N_realization):
            filename=params.key+'_{}{}_{}'.format(params.field[0],params.order,realisation)
            #if not os.path.isfile(params.song_path+'ps_'+filename+'.dat') :
            computePower(filename)
            dat=np.loadtxt(params.song_path+'ps_'+filename+'.dat')
            if realisation==0:
                ps=dat[:,1]
            else:
                ps+=dat[:,1]
            plt.loglog(dat[:,0],dat[:,1],label='{} order={} N={}'.format(params.measurement,params.order,N))

        clem1=np.loadtxt('order'+str(params.order)+'.dat')
        plt.loglog(clem1[:,0],clem1[:,1],label='clem'+str(params.order))

if __name__ == "__main__":
    main()
