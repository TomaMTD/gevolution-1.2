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
from numba import jit,prange
song_path='/home/thomas/song/'                  # set your one path to song
gev_path ='/home/thomas/song/gevolution-1.2/'   # set your one path to gevolution
sys.path.insert(0, song_path+'python')          # path to python module of song
import songy as s
import h5py

####################################################################################### 
####################################################################################### Parameters
h=0.67556
omega_b=0.022032/h**2
omega_cdm=0.12038/h**2
omega_k=0
H0= 100*h/3/10**5
z_song=[100,101] # second one for time derivative (velocity)
z=z_song[0]
H=100*np.sqrt((omega_b+omega_cdm)*(1+z))/3/10**5 #song provide a slightly different H
fnl=0
A_s = 2.215e-9
n_s = 0.9619
k_pivot = 0.05 # 1/Mpc


####################################################################################### 
####################################################################################### SONG wrapper
def run_song(kmax,kmin,N,opt):
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
k3_size = {}
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

    k_modulus_max=2*kmax

    ini="./m{}matter.ini".format(N)
    pre="./m{}matter.pre".format(N)
    
    file = open(ini, "w")
    if len(z_song)>1: file.write(ini_file.format(A_s,n_s,h,omega_b*h**2,omega_cdm*h**2,omega_k,fnl,str(z_song[0])+','+str(z_song[1])))
    else: file.write(ini_file.format(A_s,n_s,h,omega_b*h**2,omega_cdm*h**2,omega_k,fnl,z))
    file.close()

    file = open(pre, "w")
    file.write(pre_file.format(opt,N,opt,kmin,k_modulus_max,int(N*2)))
    file.close()

    from subprocess import call
    call(['./song', ini, pre])

    os.system("cp "+song_path+'output/sources_song_z000.dat '+song_path+"output/sources_song_z000_N{}.dat".format(N))
    os.system("cp "+song_path+'output/sources_song_z001.dat '+song_path+"output/sources_song_z001_N{}.dat".format(N))

def song_output(kmax,kmin,N,opt,filename='sources_song_z000.dat'):
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

    if not os.path.isfile(song_path+'output/'+filename):
        print(song_path+'output/sources_song_z000.dat not found')
        print('Run song ...')
        run_song(kmax,kmin,N,opt)
    song=s.FixedTauFile(song_path+'output/'+filename)

    if len(song.k1)!=2*N:
        print('The output '+song_path+'output/ found does not not has the right shape')
        print('Run song ...')
        run_song(kmax,kmin,N,opt)
        song=s.FixedTauFile(song_path+'output/'+filename)
        
    dk12=song.k1[1]-song.k1[0]
    dk3=np.diff(song.k3)[:,0]
    return np.array(song.get_source(b'delta_cdm')),song.tau,song.k1,np.array(song.k2,dtype=object),np.array(song.k3),song.flatidx,dk12,dk3

def song_main(kmax,kmin,N,opt):
    '''Main function for SONG '''

    s1="sources_song_z000_N{}.dat".format(N)
    s2="sources_song_z001_N{}.dat".format(N)
    source,tau,k1,k2,k3,flatidx,dk12,dk3=song_output(kmax,kmin,N,opt,s1)
    source1,tau1,_,_,_,_,_,_=song_output(kmax,kmin,N,opt,s2)
    return source,k1/h,k2/h,k3/h,flatidx,dk12/h,dk3/h,source1,tau1-tau

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
    #import pylab as plt

    #song=s.FixedTauFile(song_path+"output/sources_song_z000_N{}.dat".format(N))
    #k=song.first_order_sources['k']/h**2

    #Primordial= A_s*(k/(k_pivot*h))**(n_s-1)/k**3*2*np.pi**2 
    #
    #tr_delta_cdm=song.first_order_sources[b'delta_cdm']     
    #tr_phi=tr_delta_cdm*(-3*H**2/2) /(k**2+3*H**2) 
    #
    ## Phi transfer function comparision with CLASS
    #plt.figure()
    #plt.loglog(k,np.abs(tr_delta_cdm))
    #claSS=np.loadtxt('../class_public/output/myclasstk.dat')
    #plt.loglog(claSS[:,0]/h,np.abs(claSS[:,3]))
    #plt.show()

    # Power spectrum comparison with Gevolution
    #plt.figure()
    #plt.loglog(k,Primordial*k**3/2/np.pi**2*tr_phi**2)

    #Pclass= A_s*(claSS[:,0]/(k_pivot*h))**(n_s-1)
    #plt.loglog(claSS[:,0],Pclass*claSS[:,7]**2)

    #gev=np.loadtxt('gevolution-1.2/output/lcdm_pk000_phi.dat')
    #plt.loglog(gev[:,0],gev[:,1])
    #plt.show()

    #################################################################Pure class
    claSS=np.loadtxt('../class_public/output/myclasstk.dat')
    k=claSS[:,0]
    Primordial= A_s*(k/(k_pivot*h))**(n_s-1)/k**3*2*np.pi**2
    tr_delta_cdm=claSS[:,3]
    tr_phi=claSS[:,7]

    #plt.figure()
    #plt.loglog(k,Primordial*k**3/2/np.pi**2 *tr_phi**2,label='Analytic')
    #gev=np.loadtxt('gevolution-1.2/output/mylcdm_pk000_phi.dat')
    #plt.loglog(gev[:,0],gev[:,1],label='SONG')
    #gev=np.loadtxt('gevolution-1.2/output/lcdm_pk000_phi.dat')
    #plt.loglog(gev[:,0],gev[:,1],label='basic')
    #plt.legend()
    #plt.show()

    dk=np.diff(np.append(k,k[-1]*2-k[-2]))
    dT=np.diff(np.append(tr_phi,tr_phi[-1]*2-tr_phi[-2]))

    return np.array([k,Primordial]),np.array([k,tr_delta_cdm]),np.array([k,tr_phi]),np.array([k,dT/dk])

####################################################################################### 
####################################################################################### First order stocastic potential 
def zeta_realisation(k_grid,Primordial):
    '''
        Generate the linear curvature perturbation field (N//2+1,N,N) at redshift z in half of Fourier space. 
        The reality condition ensure the other half.
        The computation is in 3 steps:
            -compute the modulus of k in the grid (k)
            -interpolate transfer function and primordial power spectrum tr=T(k) and P=P(k)
            -randomly draw the real/imaginary part of the primordial curvature zeta following a Gaussian PDF with std=sqrt(P(k)/2)
        '''
    def random (k):
        P=np.interp(k,Primordial[0],Primordial[1])

        zeta_ini_Re=np.random.normal(0,np.sqrt(P/2*(2*np.pi)**3),k.shape) # see https://nms.kcl.ac.uk/eugene.lim/AdvCos/lecture2.pdf
        zeta_ini_Im=np.random.normal(0,np.sqrt(P/2*(2*np.pi)**3),k.shape) # for the (2pi)^3 factor (around eq 16) 
                                                                         # /d3x
                                                                          
        # equivalent :
        #rho =  np.random.normal(0,np.sqrt(P*(2*np.pi)**3),k.shape)
        #phase = np.random.uniform(0,2*np.pi,k.shape)
        #zeta_ini_Re=rho*np.cos(phase)
        #zeta_ini_Im=rho*np.sin(phase)
        return zeta_ini_Re+zeta_ini_Im*1j

    zeta = np.zeros((N//2+1,N,N),dtype=np.complex64)

    # fill density ignoring the plan z=0 (zeta[0])
    k=np.sqrt(k_grid[0]**2+k_grid[1][N//2+1:]**2+k_grid[2]**2)
    zeta[1:,:,:]= random (k) 
    # fill half of the plan z=0 ignoring the line (z=0,y=N//2)
    k=np.sqrt(k_grid[0]**2+k_grid[1][N//2+1:]**2)[:,:,0]
    zeta[0,N//2+1:]=random(k) 
    # fill half of the line (z=0,y=N//2)
    k=k_grid[0][0,:,0][N//2+1:]
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
def order1(k):
    '''Compute the first order quantities X1/delta1 at a given k. 
       X being: potential phi1 (==psi1), displacement field xi1, velocity v1
       See equation (36) of the note.
        '''
    phi1=-3*H**2 / 2 / (3*H**2+k**2)
    xi1= (1-3*phi1)/k**2
    v1=-2/3/H*phi1 
    return phi1,xi1,v1

def song2xi(song,k1,k2,k,flatidx):
    '''Compute the second order displacement field xi2:
       input: 
            -song is the output of SONG: the second order density kernel
            -k1,k2,k are the SONG output mode modulus k1,k2,k3
            -flatidx is the output of SONG flatidx
       See note for the detailed computation. 

            -k1dk2 is scalar product of k1 and k2
            -phi2p is the time derivative of the second order potential phi
        '''
    xi2  =np.copy(song)
    phi2 =np.copy(song)
    phi2p=np.copy(song)
    chi2 =np.copy(song)
    for ind1,kk1 in enumerate(k1):
        for ind2,kk2 in enumerate(k2[ind1]):

            kk3 = k3[flatidx[ind1,ind2]]

            phi1_k1,xi1_k1,v1_k1 = order1(kk1)
            phi1_k2,xi1_k2,v1_k2 = order1(kk2)
            d2 = song[flatidx[ind1,ind2]]

            k1dk2=(kk3**2-kk1**2-kk2**2)/2

            chi2[flatidx[ind1,ind2]]=3./2/kk3**4*( 2*(kk1**2+kk2**2)*k1dk2/3+k1dk2**2/3+kk1**2*kk2**2 )\
                 *(3*H**2*v1_k1*v1_k2+2*phi1_k1*phi1_k2)

            phi2p[flatidx[ind1,ind2]]=(-3*H**2*k1dk2*v1_k1*v1_k2-2*kk3**2*chi2[flatidx[ind1,ind2]]\
                                            +k1dk2*phi1_k1*phi1_k2)/21/H

            phi2[flatidx[ind1,ind2]] = (3*H**2/2/(3*H**2+kk3**2))*(-d2+2*chi2[flatidx[ind1,ind2]]+ \
                    (2+k1dk2/3/H**2-2./3/H**2*(kk1**2+kk2**2))*phi1_k1*phi1_k2\
                    -2*phi2p[flatidx[ind1,ind2]]/H )

            xi2[flatidx[ind1,ind2]]  = 1/kk3**2*(d2-3*phi2[flatidx[ind1,ind2]]-3./2*\
                    (kk2**2*phi1_k1*xi1_k2+kk1**2*phi1_k2*xi1_k1)\
                    +1./2*k1dk2*v1_k1*v1_k2-9./2*phi1_k1*phi1_k2+1./2*(kk1**2*kk2**2-k1dk2**2)*xi1_k1*xi1_k2)

    return xi2,phi2,phi2p,chi2

def zeta2fields(field,zeta,k_grid,tr_delta_cdm=0,tr_phi=0):
    '''Compute the whole first order stocastic field from the first order density 
       computed in zeta_realisation().
       field can be 'phi','psi','xi','v','chi'.
        '''
    k=(np.sqrt(k_grid[0][N//2:]**2+k_grid[1]**2+k_grid[2]**2))

    tr_d=np.interp(k,tr_delta_cdm[0],tr_delta_cdm[1])
    tr_p=np.interp(k,tr_phi[0],tr_phi[1])

    if field=='delta':
        return zeta*tr_d
    elif field=='xi':
        xi=zeta*(tr_d-3*tr_p)/k**2
        xi[np.where(np.isnan(xi))]=0+0*1j
        return xi
    elif field=='v':
        return -zeta*2*tr_p/3/H
    elif field=='chi':
        return 0
    elif field=='phi':
        return tr_p*zeta
    else:
        return 'input field not understood'

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
def Kernel_analytic(k1,k2,k3,trans,dTdk):
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
def Kernel_song(kk1,kk2,kk3,K,flatidx):
    out=np.zeros((len(kk1)))
    count=0
    for ind in range(len(kk1)):
        if kk1[ind]==0 or kk2[ind]==0:
            continue
        pkk1=np.int(np.around((kk1[ind]-dk)/dk12))
        pkk2=np.int(np.around((kk2[ind]-dk)/dk12))
        pkk3=np.int(np.around((kk3-k3[flatidx[pkk1,pkk2]][0])/dk3[flatidx[pkk1,pkk2]]))
        if pkk3>=30:
            pkk3=29
        out[ind]=K[flatidx[pkk1,pkk2]][pkk3]
    return out 

def we(kz,ky,kx,di,p):
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
####################################################################################### intergration
@jit(nopython=Nopython,fastmath=True,parallel=False)
def I2_TT_LL(Int,N,xidx,yidx,zidx,k_modulus,k_max,zeta,kern,kern_arg1,kern_arg2):
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
                        +2*np.nansum(np.conjugate(zeta_neg_kmk)*zeta_neg_k    *kern_neg))*dk**3

        '''
    for iikx in prange(len(xidx)):
        kx=xidx[iikx]
        for iiky in prange(len(yidx)):
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

                kk_pos_k=k_modulus[kz_min_lim:kz_max_lim1, ky_min_lim:ky_max_lim, kx_min_lim:kx_max_lim]

                #if kz_min_lim<=kz<kz_max_lim1 and ky_min_lim<=ky<ky_max_lim and kx_min_lim<=kx<kx_max_lim:
                #    zeta_pos [kz-kz_min_lim,ky-ky_min_lim,kx-kx_min_lim]=0

                kern_pos=kern(kk_pos_k.ravel(),kk_pos_k[::-1,::-1,::-1].ravel(),k_norm,kern_arg1,kern_arg2)
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
                                         ,k_norm,kern_arg1,kern_arg2)
                    
                    Int[kz,ky,kx]=I_pos+2*np.nansum(np.conjugate(zeta_neg_kmk)\
                                             *zeta_neg_k*kern_neg)
                else:
                    Int[kz,ky,kx]=I_pos
                
    return Int

@jit(nopython=Nopython,fastmath=True,parallel=False)
def compute(zeta,cc,kz,ky,kx,kmkz,kmky,kmkx,k_modulus,kern,k_norm,kern_arg1,kern_arg2,reverse=True):

    zeta_k   = zeta[kz[0]:kz[1]    ,ky[0]:ky[1]    ,kx[0]:kx[1]].ravel()
    if reverse:
        kernel=kern(k_modulus[kz[0]:kz[1], ky[0]:ky[1],kx[0]:kx[1]].ravel()\
                     ,k_modulus[kmkz[0]:kmkz[1],kmky[0]:kmky[1],kmkx[0]:kmkx[1]][::-1,::-1,::-1].ravel()\
                     ,k_norm,kern_arg1,kern_arg2)
        zeta_kmk = zeta[kmkz[0]:kmkz[1],kmky[0]:kmky[1],kmkx[0]:kmkx[1]]\
                             [::-1,::-1,::-1].ravel()
    else:
        kernel=kern(k_modulus[kz[0]:kz[1], ky[0]:ky[1],kx[0]:kx[1]].ravel()\
                     ,k_modulus[kmkz[0]:kmkz[1],kmky[0]:kmky[1],kmkx[0]:kmkx[1]].ravel()\
                     ,k_norm,kern_arg1,kern_arg2)
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
def I2_SL(Int,N,xidx,yidx,zidx,k_modulus,k_max,zeta,kern,kern_arg1,kern_arg2):
    '''
        Compute SL integral, see function I2_TT_LL() for description of inputs
        
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
                          +np.nansum(             zeta_d_kmk        *np.conjugate(zeta_d_k)*kern_bis_d))*dk**3
        '''
    Nd2=N//2
    for iikx in prange(len(xidx)):
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



                ###########################################################################################################################
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
                                   [kmkx_min,kmkx_max],k_modulus,kern,k_norm,kern_arg1,kern_arg2)

                    Xneg=compute(zeta,1,[kz_max_lim2,kz_max_lim1],[ky_min_lim,ky_max_lim],
                                   [kx_min,kx_max],[1,-kz_max_lim2+kz_max_lim1+1],[Nd2-kym1,Nd2+kym2+1],
                                   [kmkx_min_neg,kmkx_max_neg],k_modulus,kern,k_norm,kern_arg1,kern_arg2,reverse=False)
 
                    if kz-k_max<0:
                        if kx<Nd2:
                            kx_min_kzneg,kx_max_kzneg=1,k_max-kx+1
                        else:
                            kx_min_kzneg,kx_max_kzneg=N-(kx+k_max-N)-2,N-1

                        Xd=compute(zeta,2,[1,k_max-kz+1],[N-ky_max_lim,N-ky_min_lim],
                                   [kx_min_kzneg,kx_max_kzneg],[kz_max_lim2-kz_min_lim,k_max+1],[Nd2-kym2,Nd2+kym1+1],
                                   [kmkx_min,kmkx_max],k_modulus,kern,k_norm,kern_arg1,kern_arg2,reverse=False)

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
                                   [Nd2-kxm2,Nd2+kxm1+1],k_modulus,kern,k_norm,kern_arg1,kern_arg2)

                    Yneg=compute(zeta,1,[kz_max_lim2,kz_max_lim1],[ky_min,ky_max],
                                   [kx_min_lim,kx_max_lim],[1,-kz_max_lim2+kz_max_lim1+1],[kmky_min_neg,kmky_max_neg],
                                   [Nd2-kxm1,Nd2+kxm2+1],k_modulus,kern,k_norm,kern_arg1,kern_arg2,reverse=False)

                    if kz-k_max<0:
                        if ky<Nd2:
                            ky_min_kzneg,ky_max_kzneg=1,k_max-ky+1
                        else:
                            ky_min_kzneg,ky_max_kzneg=N-(ky+k_max-N)-2,N-1

                        Yd=compute(zeta,2,[1,k_max-kz+1],[ky_min_kzneg,ky_max_kzneg],
                                   [N-kx_max_lim,N-kx_min_lim],[kz_max_lim2-kz_min_lim,k_max+1],[kmky_min,kmky_max],
                                   [Nd2-kxm2,Nd2+kxm1+1],k_modulus,kern,k_norm,kern_arg1,kern_arg2,reverse=False)

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
                                 [kmkx_min,kmkx_max],k_modulus,kern,k_norm,kern_arg1,kern_arg2)

                    if kz<Nd2+1:
                        # x&y side neg
                        XYneg=compute(zeta,1,[kz_max_lim2,kz_max_lim1],[ky_min,ky_max],
                                 [kx_min,kx_max],[1,-kz_max_lim2+kz_max_lim1+1],[N-kmky_max,N-kmky_min],
                                 [N-kmkx_max,N-kmkx_min],k_modulus,kern,k_norm,kern_arg1,kern_arg2,reverse=False)

                    if kz-k_max<0:
                        # x&y side d
                        XYd=compute(zeta,2,[1,k_max-kz+1],[N-ky_max,N-ky_min],
                                 [N-kx_max,N-kx_min],[kz_max_lim2-kz_min_lim,k_max+1],[kmky_min,kmky_max],
                                 [kmkx_min,kmkx_max],k_modulus,kern,k_norm,kern_arg1,kern_arg2,reverse=False)

                if kz+k_max>Nd2:
                    ky_min,ky_max=N-(ky_max_lim),N-(ky_min_lim)
                    kx_min,kx_max=N-(kx_max_lim),N-(kx_min_lim)
                    kz_min,kz_max=N-kz-k_max-1,N//2

                    # pure z>N//2+1
                    Z=compute(zeta,3,[kz_min,kz_max],[ky_min,ky_max],
                                   [kx_min,kx_max],[1,kz+k_max-Nd2+1],[Nd2-kym1,Nd2+kym2+1],
                                   [Nd2-kxm1,Nd2+kxm2+1],k_modulus,kern,k_norm,kern_arg1,kern_arg2)

                    if (kx-k_max<0 or kx+k_max>=N):
                        ZX=compute(zeta,3,[kz_min,kz_max],[ky_min,ky_max],
                                   [kx_min_corner,kx_max_corner],[1,kz+k_max-Nd2+1],[N-(Nd2+kym1+1),N-(Nd2-kym2)],
                                   [kmkx_min_corner,kmkx_max_corner],k_modulus,kern,k_norm,kern_arg1,kern_arg2)

                    if (ky-k_max<0 or ky+k_max>=N):
                        ZY=compute(zeta,3,[kz_min,kz_max],[ky_min_corner,ky_max_corner],
                                   [kx_min,kx_max],[1,kz+k_max-N//2+1],[kmky_min_corner,kmky_max_corner],
                                   [N-(Nd2+kxm1+1),N-(Nd2-kxm2)],k_modulus,kern,k_norm,kern_arg1,kern_arg2)

                    if (kx-k_max<0 or kx+k_max>=N) and (ky-k_max<0 or ky+k_max>=N):
                        ZXY=compute(zeta,3,[kz_min,kz_max],[ky_min_corner,ky_max_corner],
                                   [kx_min_corner,kx_max_corner],[1,kz+k_max-Nd2+1],[kmky_min_corner,kmky_max_corner],
                                   [kmkx_min_corner,kmkx_max_corner],k_modulus,kern,k_norm,kern_arg1,kern_arg2)

                I_out=Xpos+Xneg+Ypos+Yneg+Xd+Yd+XYpos+XYneg+XYd+Z+ZX+ZY+ZXY

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

                kern_bis_pos=kern(kk_pos_k,kk_pos_kmk,k_norm,kern_arg1,kern_arg2)

                if kz==Nd2:
                    Int[kz,ky,kx]=(np.nansum(zeta_pos_kmk*zeta_pos_k*kern_bis_pos))+I_out#*dk**3
                elif kz-k_max>=0:
                    kern_bis_neg=kern(kk_neg_k,kk_neg_kmk      ,k_norm,kern_arg1,kern_arg2)
                    Int[kz,ky,kx]=(np.nansum(zeta_pos_kmk*zeta_pos_k*kern_bis_pos)\
                       +np.nansum(np.conjugate(zeta_neg_kmk)*zeta_neg_k*kern_bis_neg))+I_out
                else:
                    kern_bis_neg=kern(kk_neg_k,kk_neg_kmk      ,k_norm,kern_arg1,kern_arg2)

                    zeta_d_k  =np.copy(zeta[1:k_max-kz+1,N-ky_max_lim:N-ky_min_lim,N-kx_max_lim:N-kx_min_lim].ravel())
                    zeta_d_kmk =np.copy(zeta[1:k_max-kz+1,Nd2-kym2:Nd2+kym1+1,Nd2-kxm2:Nd2+kxm1+1]).ravel()
                    kk_d_k  =k_modulus[1:k_max-kz+1,N-ky_max_lim:N-ky_min_lim,N-kx_max_lim:N-kx_min_lim].ravel()
                    kk_d_kmk=k_modulus[1:k_max-kz+1,Nd2-kym2:Nd2+kym1+1,Nd2-kxm2:Nd2+kxm1+1].ravel()
                    kern_bis_d=kern(kk_d_k,kk_d_kmk,k_norm,kern_arg1,kern_arg2)

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

def integre(which,klin,N,k_max,klin_grid,zeta,kern,kern_arg1,kern_arg2):
    '''
        inputs:
        -which: whether it is 'TT','LL', 'SL' or 'SS' to be computed. The 'SS' is not implemented yet
        -klin: one dimensional list of k coordinate to be considered
        -N: size of the grid
        -k_max: cut-off in h/Mpc^-1
        -klin_grid: sparse grid of kx,ky,kz
        -zeta: Primordial curvature
        -kern: which kernel function to use. Either Kernel_song or Kernel_analytic
                if Kernel_song:
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
    k_modulus=(np.sqrt(klin_grid[0][N//2:]**2+klin_grid[1]**2+klin_grid[2]**2))
    k_max=int(np.where(klin==k_max)[0])-N//2
    print('k_max={}'.format(k_max))
    klin=np.arange(len(klin))-N//2

    if which=='SL':
        print('integre SL ...')
        for opt in range(3):
            if opt==0:
                xidx=np.arange(N)
                yidx=np.arange(N)
                zidx=np.arange(1,N//2+1)
                #zidx=np.arange(N)
            elif opt==1:
                xidx=np.arange(N)
                yidx=np.arange(N//2+1,N)
                zidx=np.arange(1)
            elif opt==2:
                xidx=np.arange(N//2+1,N)
                yidx=np.array([N//2])
                zidx=np.arange(1)

            I2_SL(Int,N,xidx,yidx,zidx,k_modulus,k_max,zeta,kern,kern_arg1,kern_arg2)

    else:
        if which=='TT' or which=='LL':
            print('integre LL or TT ...')
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
        elif which=='SS':
            print('not yet')
            yidx1=np.arange(N)
            zidx1=np.arange(1,N//2+1)

            xidx2=np.arange(N)
            yidx2=np.arange(N//2+1,N)
            zidx2=np.array([0])

            xidx3=np.arange(N//2+1,N)
            yidx3=np.array([N//2])
            zidx3=np.array([0])
            return 0

        for opt in range(3):
            if opt==0:
                xidx=xidx1
                yidx=yidx1
                zidx=zidx1
            elif opt==1:
                xidx=xidx2
                yidx=yidx2
                zidx=zidx2
            elif opt==2:
                xidx=xidx3
                yidx=yidx3
                zidx=zidx3

            I2_TT_LL(Int,N,xidx,yidx,zidx,k_modulus,k_max,zeta,kern,kern_arg1,kern_arg2)
    return Int*dk**3/2**3/np.pi**3

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
    print('klambda='+str(kL))
    return k,np.float32(N//2*k_min),N,np.float32(k_min),kL

def fft(field):
    new_field=np.zeros((N//2+1,N,N),dtype=np.complex64)

    field[0,N//2,:N//2]=np.conjugate(field[0,N//2,N//2+1:][::-1])
    field[0,:N//2,:]   =np.conjugate(field[0,N//2+1:,:][::-1,::-1])

    new_field[:,:N//2+1,:N//2+1]=field[:,N//2:,N//2:]
    new_field[:,N//2+1:,N//2+1:]=field[:,:N//2,:N//2]

    new_field[:,:N//2+1,N//2+1:]=field[:,N//2:,:N//2]
    new_field[:,N//2+1:,:N//2+1]=field[:,:N//2,N//2:]

    return np.fft.irfftn(new_field,(N-1,N-1,N-1))

####################################################################################### 
####################################################################################### Gevolution place

def settings(kmin,ICg='SONG',disp_file='displacement.h5',vel_file='velocitypotential.h5',pot_file='phi.h5'):
    '''create a setting file with the global parameters used in the programme'''
    setx=r"""
template file = sc1_crystal.dat    
Tk file = class_tk.dat              
baryon treatment = blend          
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
snapshot file base  = lcdm_snap
snapshot redshifts  = 
snapshot outputs    = phi
Pk file base        = lcdm_pk
Pk redshifts        = 100
Pk outputs          = phi
Pk bins             = 1024
lightcone file base = lcdm_lightcone
lightcone outputs   = Gadget2, phi
lightcone 0 vertex    = 0, 0, 0       # in units of Mpc/h
lightcone 0 direction = 1, 1, 1
lightcone 0 distance  = 100           # in units of Mpc/h
lightcone 1 vertex    = 0, 0, 0       # in units of Mpc/h
lightcone 1 direction = 1, 1, 1
lightcone 1 distance  = 100, 450      # in units of Mpc/h
lightcone 1 opening half-angle = 30   # degrees
output              = mPk, dTk, vTk
gauge               = Newtonian
P_k_ini type        = analytic_Pk
P_k_max_h/Mpc       = 192           
z_pk                = 100          
root                = class_
background_verbose  = 1
spectra_verbose     = 1
output_verbose      = 1

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

    file = open(gev_path+"/msetting.ini", "w")
    file.write(setx.format(ICg,disp_file,vel_file,pot_file,N//4,k_pivot,A_s,n_s,h,omega_b*h**2,omega_cdm*h**2\
                            ,z,2*np.pi/kmin,N-1))
    file.close()



########################################################################################################################## 
########################################################################################################################## 
########################################################################################################################## Main

N=64                                                 # Number of mode to be considered (on laptop N=10 is quick enough)
print('N='+str(N))                                   #
kmin=np.float32(0.01)                                # The non-zero smallest mode  
dk=kmin                                              # which has to be equal to the step of the grid
klin_concat,kmax,N,dk,klambda=k_distrib(kmin,N,0.1) # Generate the list of mode coordinate (assume h/Mpc)
                                                     ##########################
song,k1,k2,k3,flatidx,dk12,dk3,song1,d_eta=song_main(kmax*h,kmin*h,N,'lin')  # Get song outputs
                                                                             #
Primordial,transcdm,transphi,dTdk=trans()                                    # Get transfer functions and primordial power spectrum 
k_grid_lin=np.array(np.meshgrid(klin_concat,klin_concat,\
            klin_concat,sparse=True,indexing='xy'),dtype=object)             # mode Grid 
zeta=zeta_realisation(k_grid_lin,Primordial)                                 # first order stocastic density
k_grid_lin=np.array(np.meshgrid(klin_concat,klin_concat,klin_concat\
            ,sparse=True,indexing='ij'),dtype=object)                        # mode Grid with 'ij' indexing


#TT =integre('TT',klin_concat,N,kmax   ,k_grid_lin,phi,transphi,dTdk)                 # total integrale  
#LL =integre('LL',klin_concat,N,klambda,k_grid_lin,phi,Kernel_analytic,transphi,dTdk) # LL analytic approx int
#SL =integre('SL',klin_concat,N,klambda,k_grid_lin,phi,Kernel_analytic,transphi,dTdk) # SL analytic approx int

#####debug
#k_modulus=(np.sqrt(k_grid_lin[0][N//2:]**2+k_grid_lin[1]**2+k_grid_lin[2]**2))
#xi2,phi2,_,_=song2xi(song,k1,k2,k3,flatidx)             
#G=np.ones(np.shape(k_modulus))
#SL=integre('SL',klin_concat,N,klambda,k_grid_lin,G,Kernel_song,xi2,flatidx)   
#LL=integre('LL',klin_concat,N,klambda,k_grid_lin,G,Kernel_song,xi2,flatidx)   
#####

#################################################################################### displacement field computation
print('delta:')                                                                    #
delta_LL=integre('LL',klin_concat,N,klambda,k_grid_lin,zeta,Kernel_song,song,flatidx)  # Integration
delta_SL=integre('SL',klin_concat,N,klambda,k_grid_lin,zeta,Kernel_song,song,flatidx)  #
delta1 = zeta2fields('delta',zeta,k_grid_lin,transcdm,transphi)                        # first order displacement field
                                                                                   #
hf = h5py.File(gev_path+'density1.h5', 'w')                                      # Save in h5 format 
hf.create_dataset('data', data=fft(delta1))                                        # 
hf.close()                                                                         #
                                                                                   #
hf = h5py.File(gev_path+'density2.h5', 'w')                                      # Save in h5 format 
hf.create_dataset('data', data=fft(delta_LL+2*delta_SL))                           # 
hf.close()                                  

#################################################################################### displacement field computation
print('displacement:')                                                             #
xi2,phi2,_,_=song2xi(song,k1,k2,k3,flatidx)                                        # Compute the second order kernels xi2 and phi2
xi_LL=integre('LL',klin_concat,N,klambda,k_grid_lin,zeta,Kernel_song,xi2,flatidx)  # Integration
xi_SL=integre('SL',klin_concat,N,klambda,k_grid_lin,zeta,Kernel_song,xi2,flatidx)  #
xi1    = zeta2fields('xi',zeta,k_grid_lin,transcdm,transphi)                       # first order displacement field
                                                                                   #
hf = h5py.File(gev_path+'displacement.h5', 'w')                                    # Save in h5 format 
hf.create_dataset('data', data=fft(xi1))                                           # 
hf.close()                                                                         #

print('potential:')                                                                #
phi_LL=integre('LL',klin_concat,N,klambda,k_grid_lin,zeta,Kernel_song,phi2,flatidx)# Integration
phi_SL=integre('SL',klin_concat,N,klambda,k_grid_lin,zeta,Kernel_song,phi2,flatidx)#
                                                                                   #
phi1 = zeta2fields('phi',zeta,k_grid_lin,transcdm,transphi)                        # first order displacement field
hf = h5py.File(gev_path+'phi.h5', 'w')                                             # Save in h5 format 
hf.create_dataset('data', data=fft(phi1))                                          # 
hf.close()                                                                         #

################################################################################ scalar velocity computation
print('Scalar velocity:')                                                      #
xi2_1,_,_,_=song2xi(song1,k1,k2,k3,flatidx)                                    # Compute the second order kernels velocity v2
v2 = (xi2_1-xi2)/d_eta                                                         #
v_LL=integre('LL',klin_concat,N,klambda,k_grid_lin,zeta,Kernel_song,v2,flatidx)# Integration
v_SL=integre('SL',klin_concat,N,klambda,k_grid_lin,zeta,Kernel_song,v2,flatidx)#
                                                                               #
v1  =zeta2fields('v',zeta,k_grid_lin,transcdm,transphi)                        # first order velocity
hf = h5py.File(gev_path+'velocitypotential.h5', 'w')                           # Save in h5 format 
hf.create_dataset('data', data=fft(v1))                                        # 
hf.close()                                                                     #

#################################################################################### Gevolution 
settings(kmin)
