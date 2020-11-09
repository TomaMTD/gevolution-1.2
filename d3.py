'''
code in python3
Needs to be in the same directory as SONG  (/.../song/) 
'''


####################################################################################### import
import numpy as np
import os
import sys
from numba import jit,prange
song_path='/home/thomas/song/'         # set your one path 
sys.path.insert(0, song_path+'python') # path to python module of song
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
H=100*h*np.sqrt((omega_b+omega_cdm)*(1+z))/3/10**5 #song provide a slightly different H
fnl=0
A_s = 2.215e-9
n_s = 0.9619
k_pivot = 0.05


####################################################################################### 
####################################################################################### SONG wrapper
def run_song(kmax,kmin,N,opt):
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

    
    file = open("./mmatter.ini", "w")
    if len(z_song)>1: file.write(ini_file.format(A_s,n_s,h,omega_b*h**2,omega_cdm*h**2,omega_k,fnl,str(z_song[0])+','+str(z_song[1])))
    else: file.write(ini_file.format(A_s,n_s,h,omega_b*h**2,omega_cdm*h**2,omega_k,fnl,z))
    file.close()

    file = open("./mmatter.pre", "w")
    file.write(pre_file.format(opt,N,opt,kmin,k_modulus_max,int(N*3)))
    file.close()

    from subprocess import call
    call(['./song', './mmatter.ini', './mmatter.pre'])

def song_output(kmax,kmin,N,opt,filename='sources_song_z000.dat'):

    if not os.path.isfile(song_path+'output/'+filename):
        print(song_path+'output/sources_song_z000.dat not found')
        print('Run song ...')
        run_song(kmax,kmin,N,opt)

    song=s.FixedTauFile(song_path+'output/'+filename)
    if len(song.k1)!=3*N:
        print('The output '+song_path+'output/ found does not not has the right shape')
        print('Run song ...')
        run_song(kmax,kmin,N,opt)
        song=s.FixedTauFile(song_path+'output/'+filename)
        
    dk12=song.k1[1]-song.k1[0]
    dk3=np.diff(song.k3)[:,0]
    return np.array(song.get_source(b'delta_cdm')),song.tau,song.k1,np.array(song.k2,dtype=object),np.array(song.k3),song.flatidx,dk12,dk3

def song_main(kmax,kmin,N,opt,filename=['sources_song_z000.dat','sources_song_z001.dat']):
    source,tau,k1,k2,k3,flatidx,dk12,dk3=song_output(kmax,kmin,N,opt,filename[0])
    try:
        source1,tau1,_,_,_,_,_,_=song_output(kmax,kmin,N,opt,filename[1])
        return source,k1,k2,k3,flatidx,dk12,dk3,source1,tau1-tau
    except IndexError:
        return source,k1,k2,k3,flatidx,dk12,dk3

####################################################################################### 
####################################################################################### first order transfert fct
def trans():
    '''
       This function returns the Primordial power spectrum, the transfer functions of delta_cdm and phi, and 
       derivative of the last transfer function. 

    -Primordial power spectrum : Primordial = A_s(k/k_0)**(ns-1) / (k**3/(2*np.pi**2)). 
     Used in first_order_fields() to generate the linear initial potential phi_ini

    -delta_cdm transfer function: tr_delta_cdm(k,z)*phi_ini(k)=delta_cdm(k,z)
     Used in first_order_fields() to transform phi_ini into density

    -potential transfer function: tr_phi(z,k)*phi_ini(k)=phi(z,k)
     Used in Kernel() (as its derivative) for the analytic approximation VR+
        '''

    song=s.FixedTauFile(song_path+'output/sources_song_z000.dat')
    k=song.first_order_sources['k']
    Primordial= A_s*(k/k_pivot)**(n_s-1)/k**3*2*np.pi**2
    tr_delta_cdm=song.first_order_sources[b'delta_cdm']
    tr_phi=tr_delta_cdm*(-3.*H0**2*(omega_cdm+omega_b)*(1+z)/2)\
              /(-k**2-3*H**2)

    dk=np.diff(np.append(k,k[-1]*2-k[-2]))
    dT=np.diff(np.append(tr_phi,tr_phi[-1]*2-tr_phi[-2]))
    return np.array([k,Primordial]),np.array([k,tr_delta_cdm]),np.array([k,tr_phi]),np.array([k,dT/dk])

####################################################################################### 
####################################################################################### First order stocastic potential 
def first_order_fields(k_grid,Primordial):
    '''
        Generate the linear density field (N//2+1,N,N) at redshift z in half of Fourier space. 
        The reality condition ensure the other half.
        The computation is in 3 steps:
            -compute the modulus of k in the grid (k)
            -interpolate transfer function and primordial power spectrum tr=T(k) and P=P(k)
            -randomly draw the real/imaginary part of the primordial potential phi following a Gaussian PDF with std=P(k)/2
            -compute the density contrast delta=phi*tr
        '''
    phi = np.zeros((N//2+1,N,N),dtype=np.complex64)

    # fill density ignoring the plan z=0 (delta[0])
    k=np.sqrt(k_grid[0]**2+k_grid[1][N//2+1:]**2+k_grid[2]**2)
    #tr=np.interp(k,transcdm[0],transcdm[1])
    P=np.interp(k,Primordial[0],Primordial[1])
    phi_ini_Re=np.random.normal(0,P/2,(N//2,N,N))
    phi_ini_Im=np.random.normal(0,P/2,(N//2,N,N))
    phi[1:,:,:]=(phi_ini_Re+phi_ini_Im*1j) #*tr

    # fill half of the plan z=0 ignoring the line (z=0,y=N//2)
    k=np.sqrt(k_grid[0]**2+k_grid[1][N//2+1:]**2)[:,:,0]
    #tr=np.interp(k,transcdm[0],transcdm[1])
    P=np.interp(k,Primordial[0],Primordial[1])
    phi_ini_Re_lin=np.random.normal(0,P/2,(N//2,N))
    phi_ini_Im_lin=np.random.normal(0,P/2,(N//2,N))
    phi[0,N//2+1:]=(phi_ini_Re_lin+phi_ini_Im_lin*1j)# *tr
    phi[0,:N//2]=np.conjugate(phi[0,N//2+1:][::-1,::-1])

    # fill half of the line (z=0,y=N//2)
    k=k_grid[0][0,:,0][N//2+1:]
    #tr=np.interp(k,transcdm[0],transcdm[1])
    P=np.interp(k,Primordial[0],Primordial[1])
    phi_ini_Re_lin=np.random.normal(0,P/2,(N//2))
    phi_ini_Im_lin=np.random.normal(0,P/2,(N//2))
    phi[0,N//2,N//2+1:]=(phi_ini_Re_lin+phi_ini_Im_lin*1j)# *tr
    phi[0,N//2,:N//2]=np.conjugate(phi[0,N//2,N//2+1:][::-1])
    return phi

####################################################################################### 
####################################################################################### From initial potential to displacement field
def order1(k):
    '''Compute the first order quantities X1/delta1 at a given k. 
       X being: potential phi1 (==psi1), displacement field xi1, velocity v1
        '''
    phi1=-3*H**2 / 2 / (3*H**2+k**2)
    xi1= (1-3*phi1)/k**2
    v1=-2/3/H*phi1 
    return phi1,xi1,v1

def deltacdm2xi(delta_cdm,k1,k2,k,flatidx):
    '''Compute the second order displacement field xi2:
       input: 
            -delta_cdm is the output of SONG: the second order density kernel
            -k1,k2,k are the SONG output mode modulus k1,k2,k3
            -flatidx is the output of SONG flatidx
       See note for the detailed computation. 

            -k1dk2 is scalar product of k1 and k2
            -phi2p is the time derivative of the second order potential phi
        '''
    xi2=delta_cdm*0
    for ind1,kk1 in enumerate(k1):
        for ind2,kk2 in enumerate(k2[ind1]):

            kk3 = k3[flatidx[ind1,ind2]]

            phi1_k1,xi1_k1,v1_k1 = order1(kk1)
            phi1_k2,xi1_k2,v1_k2 = order1(kk2)
            d2 = delta_cdm[flatidx[ind1,ind2]]

            k1dk2=(kk3**2-kk1**2-kk2**2)/2

            chi2=3./2/kk3**4*( (kk1**2+kk2**2)*k1dk2/3-k1dk2**2/3+kk1**2*kk2**2 )\
                 *(3*H**2*v1_k1*v1_k2+2*phi1_k1*phi1_k2)

            phi2p=(-3*H**2*k1dk2*v1_k1*v1_k2-2*kk3**2*chi2+k1dk2*phi1_k1*phi1_k2)/21/H

            phi2 = (3*H**2/2/(3*H**2+kk3**2))*(-d2+2*chi2+ \
                    (2+k1dk2/3/H**2-2./3/H**2*(kk1**2+kk2**2))*phi1_k1*phi1_k2\
                    -2*phi2p/H )

            xi2[flatidx[ind1,ind2]]  = 1/kk3**2*(d2-3*phi2-3./2*(kk2**2*phi1_k1*xi1_k2+kk1**2*phi1_k2*xi1_k1)\
                    +1./2*k1dk2*v1_k1*v1_k2-9./2*phi1_k1*phi1_k2+1./2*(kk1**2*kk2**2-k1dk2**2)*xi1_k1*xi1_k2)

    return xi2,phi2,phi2p,chi2

def phi2fields(field,phi,k_grid,tr_delta_cdm,tr_phi):
    '''Compute the whole first order stocastic field from the first order density 
       computed in first_order_fields().
       field can be 'phi','psi','xi','v','chi'.
        '''
    k=(np.sqrt(k_grid[0][N//2:]**2+k_grid[1]**2+k_grid[2]**2))
    #k= np.sqrt(k_grid[0]**2+k_grid[1][N//2:]**2+k_grid[2]**2)

    tr_d=np.interp(k,tr_delta_cdm[0],tr_delta_cdm[1])
    tr_p=np.interp(k,tr_phi[0],tr_phi[1])

    if field=='delta':
        return phi*tr_d
    elif field=='xi':
        return phi*(tr_d-3*tr_p)/k**2
    elif field=='v':
        return -phi*2*tr_p/3/H
    elif field=='chi':
        return 0
    else:
        return 'input field not understand'

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
def Kernel_song(kk1,kk2,kk3,delta_cdm,flatidx):
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
        out[ind]=delta_cdm[flatidx[pkk1,pkk2]][pkk3]
    return out 


####################################################################################### 
####################################################################################### intergration
@jit(nopython=Nopython,fastmath=True,parallel=False)
def delta2_TT_LL(Int,N,xidx,yidx,zidx,k_modulus,k_max,delta_ini,kern,kern_arg1,kern_arg2):
    '''
        Compute LL integral with cut-off k_max. Here k_max is an index defined in 
        function integre(): k_max=int(np.where(klin==k_max)[0])-N//2.
        If k_max=N//2, this function return the raw integral. 
        input:
            -Int: integral result array(N//2+1,N,N)
            -N: size of grid 
            -xidx,yidx,zidx: index of the k-grid where the integral is computed. As for delta_ini computed in first_order_fields, 
                             this splitting allow this integral to be computed in different part (Volume+plan+line). 
                             More details in function integre().
            -k_modulus: (half)whole grid modulus
            -k_max: cut-off index
            -delta_ini: linear density
            -kern,kern_arg1,kern_arg2: see integre() intput description

        Computation is split in two parts. For delta(k1)*delta(k-k1):

            -if 0<=k1_z<=k_z 
                delta(k1)=delta_pos (pos=positive because then k_z-k1_z>=0).
                In this particular case: delta(k-k1)=delta_pos[::-1]
            -if k1_z>k_z 
                delta(k1)  =delta_neg_k 
                delta(k-k1)=np.conjugate(delta_neg_kmk) (kmk for k minus k1) 
                    (Thanks to reality and because we only have in memory density for k_z>=0)

            a third part would be considered:
                -if k1_z<0:
                    let us call delta_d_k and delta_d_kmk the corresponding density
                    We can show that delta_d_k=delta_neg_kmk and delta_neg_k=delta_d_kmk

            Taking into account reality condition, we finally get:
            Int[kz,ky,kx]=(np.nansum(             delta_pos     *delta_pos[::-1]*kern_pos)\
                        +2*np.nansum(np.conjugate(delta_neg_kmk)*delta_neg_k    *kern_neg))*dk**3

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

                delta_pos=np.copy(delta_ini[kz_min_lim:kz_max_lim1,\
                                                 ky_min_lim:ky_max_lim,\
                                                kx_min_lim:kx_max_lim]).ravel()
                kk_pos_k=np.copy(k_modulus[kz_min_lim:kz_max_lim1,\
                                              ky_min_lim:ky_max_lim,\
                                              kx_min_lim:kx_max_lim]).ravel()

                if kz_min_lim<=kz<kz_max_lim1 and ky_min_lim<=ky<ky_max_lim and kx_min_lim<=kx<kx_max_lim:
                    delta_pos.reshape((kz_max_lim1-kz_min_lim,\
                                           ky_max_lim-ky_min_lim,\
                                           kx_max_lim-kx_min_lim))[kz-kz_min_lim,ky-ky_min_lim,kx-kx_min_lim]=0

                kern_pos=kern(kk_pos_k,kk_pos_k[::-1]\
                                   ,k_norm,kern_arg1,kern_arg2)

                # if kz>k_max, the delta_neg part vanishes
                if k_max-kz+1>0:
                    kz_max_lim2=k_max+1
                    delta_neg_k=delta_ini[kz_max_lim1:kz_max_lim2,\
                                             ky_min_lim:ky_max_lim,\
                                             kx_min_lim:kx_max_lim].ravel()
                    kk_neg_k   =k_modulus   [kz_max_lim1:kz_max_lim2,\
                                             ky_min_lim:ky_max_lim,\
                                             kx_min_lim:kx_max_lim].ravel()

                    delta_neg_kmk=delta_ini[1:k_max-kz+1,N-ky_max_lim:N-ky_min_lim,N-kx_max_lim:N-kx_min_lim].ravel()
                    kk_neg_kmk   =k_modulus   [1:k_max-kz+1,N-ky_max_lim:N-ky_min_lim,N-kx_max_lim:N-kx_min_lim].ravel()

                    kern_neg=kern(kk_neg_k,kk_neg_kmk\
                                         ,k_norm,kern_arg1,kern_arg2)
                    
                    Int[kz,ky,kx]=(np.nansum(delta_pos*delta_pos[::-1]*kern_pos)\
                       +2*np.nansum(np.conjugate(delta_neg_kmk)\
                                             *delta_neg_k\
                                             *kern_neg))*dk**3
                else:
                    Int[kz,ky,kx]=(np.nansum(delta_pos*delta_pos[::-1]*kern_pos))*dk**3
    return Int

@jit(nopython=Nopython,fastmath=True,parallel=False)
def delta2_SL(Int,N,xidx,yidx,zidx,k_modulus,k_max,delta_ini,kern,kern_arg1,kern_arg2):
    '''
        Compute SL integral, see function delta2_TT_LL() for description of inputs
        
        Computation is split in three parts. For delta(k1)*delta(k-k1):

            -if 0<=k1_z<=k_z 
                delta(k1)=delta_pos (pos=positive because then k_z-k1_z>=0).
                In this particular case: delta(k-k1)=delta_pos[::-1]
            -if k1_z>k_z 
                delta(k1)  =delta_neg_k 
                delta(k-k1)=np.conjugate(delta_neg_kmk) (kmk for k minus k1) 
                    (Thanks to reality and because we only have in memory density for k_z>=0)
            -if k1_z<0:
                delta(k1)  =np.conjugate(delta_d_k )
                    (Thanks to reality and because we only have in memory density for k_z>=0)
                delta(k-k1)=delta_d_kmk 

            Taking into account reality condition, we finally get:
 
            Int[kz,ky,kx]=(np.nansum(             delta_pos_kmk[::-1]*delta_pos_k            *kern_bis_pos)\
                          +np.nansum(np.conjugate(delta_neg_kmk)     *delta_neg_k            *kern_bis_neg)\
                          +np.nansum(             delta_d_kmk        *np.conjugate(delta_d_k)*kern_bis_d))*dk**3
        '''

    for iikx in prange(len(xidx)):
        kx=xidx[iikx]
        for iiky in prange(len(yidx)):
            ky=yidx[iiky]
            for iikz in prange(len(zidx)):
                kz=zidx[iikz]

                k_norm=k_modulus[kz,ky,kx]

                kz_min_lim=max(0,kz-k_max)
                kz_max_lim1=min(kz+k_max+1,N//2+1)
                kz_max_lim2=kz+1

                kx_min_lim=max(0,kx-k_max)
                kx_max_lim=min(kx+k_max+1,N)
                ky_min_lim=max(0,ky-k_max)
                ky_max_lim=min(ky+k_max+1,N)
                
                delta_pos_k=np.copy(delta_ini[kz_min_lim:kz_max_lim2,\
                                                 ky_min_lim:ky_max_lim,
                                                 kx_min_lim:kx_max_lim]).ravel()
                delta_pos_k.reshape((kz_max_lim2-kz_min_lim,\
                                        ky_max_lim-ky_min_lim,\
                                        kx_max_lim-kx_min_lim))[kz-kz_min_lim,ky-ky_min_lim,kx-kx_min_lim]=0
                kk_pos_k=k_modulus[kz_min_lim:kz_max_lim2,\
                                   ky_min_lim:ky_max_lim,\
                                   kx_min_lim:kx_max_lim].ravel()

                delta_neg_k=np.copy(delta_ini[kz_max_lim2:kz_max_lim1,ky_min_lim:ky_max_lim,kx_min_lim:kx_max_lim]).ravel()
                kk_neg_k   =        k_modulus   [kz_max_lim2:kz_max_lim1,ky_min_lim:ky_max_lim,kx_min_lim:kx_max_lim].ravel()

                kxm1    =-kx_min_lim+kx          
                kxm2    = kx_max_lim-kx-1        
                kym1    =-ky_min_lim+ky          
                kym2    = ky_max_lim-ky-1        

                kzlower= k_max+kz+1-kz_max_lim1 

                delta_pos_kmk =np.copy(delta_ini[:kz_max_lim2-kz_min_lim,\
                                                  N//2-kym2:N//2+kym1+1,\
                                                  N//2-kxm2:N//2+kxm1+1]).ravel()
                kk_pos_kmk    =        k_modulus   [:kz_max_lim2-kz_min_lim,\
                                                  N//2-kym2:N//2+kym1+1,\
                                                  N//2-kxm2:N//2+kxm1+1].ravel()
                delta_neg_kmk =np.copy(delta_ini[1:k_max+1-kzlower,\
                                                  N//2-kym1:N//2+kym2+1,
                                                  N//2-kxm1:N//2+kxm2+1]).ravel()
                kk_neg_kmk    =        k_modulus   [1:k_max+1-kzlower,\
                                                  N//2-kym1:N//2+kym2+1,\
                                                  N//2-kxm1:N//2+kxm2+1].ravel()

                if kz-k_max<k_max+1:
                    if N//2-k_max<kx<N//2+k_max+1 and N//2-k_max<ky<N//2+k_max+1 and N//2-k_max<kz<N//2+k_max+1:

                        kxupper=min(N//2+1+2*k_max-kx,kx_max_lim-kx_min_lim)
                        kxlower=max(N//2-kx,0)
                        kyupper=min(N//2+1+2*k_max-ky,ky_max_lim-ky_min_lim)
                        kylower=max(N//2-ky,0)
                        kzupper=min(2*k_max-kz+1,k_max+1)
                        delta_pos_k.reshape((kz_max_lim2-kz_min_lim,\
                                                ky_max_lim-ky_min_lim,\
                                                kx_max_lim-kx_min_lim))[:kzupper,kylower:kyupper,kxlower:kxupper]=0

                        klx=max(0,2*k_max-(kxupper-kxlower)+1)
                        kux=min(kxupper-kxlower,2*k_max+1)
                        kly=max(0,2*k_max-(kyupper-kylower)+1)
                        kuy=min(kyupper-kylower,2*k_max+1)

                        delta_pos_kmk.reshape((kz_max_lim2-kz_min_lim,kym1+1+kym2,kxm1+1+kxm2))\
                                    [k_max-kzupper+1:,kly:kuy,klx:kux]=0

                        if kz<k_max+1:
                            if kx<N//2+1 and ky<N//2+1:
                                delta_neg_k.reshape((kz_max_lim1-kz_max_lim2,ky_max_lim-ky_min_lim,kx_max_lim-kx_min_lim))\
                                                [:k_max-kz,N//2-ky:,N//2-kx:]=0
                                delta_neg_kmk.reshape((k_max-kzlower,kym2+1+kym1,kxm2+1+kxm1))\
                                                [:k_max-kz,N//2-ky:,N//2-kx:]=0
                            elif kx<N//2+1 and ky>N//2:
                                delta_neg_k.reshape((kz_max_lim1-kz_max_lim2,ky_max_lim-ky_min_lim,kx_max_lim-kx_min_lim))\
                                                [:k_max-kz,:N//2-ky,N//2-kx:]=0
                                delta_neg_kmk.reshape((k_max-kzlower,kym2+1+kym1,kxm2+1+kxm1))\
                                                [:k_max-kz,:N//2-ky,N//2-kx:]=0
                            elif kx>N//2 and ky<N//2+1:
                                delta_neg_k.reshape((kz_max_lim1-kz_max_lim2,ky_max_lim-ky_min_lim,kx_max_lim-kx_min_lim))\
                                                [:k_max-kz,N//2-ky:,:N//2-kx]=0
                                delta_neg_kmk.reshape((k_max-kzlower,kym2+1+kym1,kxm2+1+kxm1))\
                                                [:k_max-kz,N//2-ky:,:N//2-kx]=0
                            else:
                                delta_neg_k.reshape((kz_max_lim1-kz_max_lim2,ky_max_lim-ky_min_lim,kx_max_lim-kx_min_lim))\
                                                [:k_max-kz,:N//2-ky,:N//2-kx]=0
                                delta_neg_kmk.reshape((k_max-kzlower,kym2+1+kym1,kxm2+1+kxm1))\
                                                [:k_max-kz,:N//2-ky,:N//2-kx]=0

                kern_bis_pos=kern(kk_pos_k,kk_pos_kmk[::-1],k_norm,kern_arg1,kern_arg2)

                # splitting in three conditions since:
                #                   for kz==N//2: delta_neg_k=delta_d_k=0 
                #                   for kz>=k_max or kx==N//2 or ky==N//2: delta_d_k=0 
                #                   else: general formula splitting into different regions of the space (can probably be simplified)
                if kz==N//2:
                    Int[kz,ky,kx]=(np.nansum(delta_pos_kmk[::-1]*delta_pos_k*kern_bis_pos))*dk**3
                elif kz-k_max>=0 or kx==N//2 or ky==N//2:
                    kern_bis_neg=kern(kk_neg_k,kk_neg_kmk      ,k_norm,kern_arg1,kern_arg2)
                    Int[kz,ky,kx]=(np.nansum(delta_pos_kmk[::-1]*delta_pos_k*kern_bis_pos)\
                       +np.nansum(np.conjugate(delta_neg_kmk)*delta_neg_k*kern_bis_neg))*dk**3
                else:
                    kern_bis_neg=kern(kk_neg_k,kk_neg_kmk      ,k_norm,kern_arg1,kern_arg2)
                    if kx<N//2+1 :
                        if ky<N//2+1:
                            kxlower=N-min(kx+k_max+1,N//2-k_max)
                            kylower=N-min(ky+k_max+1,N//2-k_max)
                            delta_d_k=delta_ini[1:k_max-kz+1,kylower:N-ky_min_lim,kxlower:N-kx_min_lim].ravel()
                            kk_d_k   =k_modulus   [1:k_max-kz+1,kylower:N-ky_min_lim,kxlower:N-kx_min_lim].ravel()

                            kxlower=max(N//2-k_max,kx+k_max+1)
                            kxupper=min(N//2+k_max+1,N//2+kx+1)
                            kylower=max(N//2-k_max,ky+k_max+1)
                            kyupper=min(N//2+k_max+1,N//2+ky+1)
                            delta_d_kmk=delta_ini[kz+1:k_max+1,kylower:kyupper,kxlower:kxupper].ravel()
                            kk_d_kmk   =k_modulus   [kz+1:k_max+1,kylower:kyupper,kxlower:kxupper].ravel()
                            kern_bis_d=kern(kk_d_k,kk_d_kmk,k_norm,kern_arg1,kern_arg2)
 
                            Int[kz,ky,kx]=(np.nansum(delta_pos_kmk[::-1]*delta_pos_k*kern_bis_pos)\
                              +np.nansum(np.conjugate(delta_neg_kmk)*delta_neg_k*kern_bis_neg)\
                              +np.nansum(delta_d_kmk*np.conjugate(delta_d_k)*kern_bis_d))*dk**3


                        else:
                            kxlower=N-min(kx+k_max+1,N//2-k_max)
                            kyupper=N-max(ky-k_max,N//2+k_max+1)
                            delta_d_k=delta_ini[1:k_max-kz+1,N-ky_max_lim:kyupper,kxlower:N-kx_min_lim].ravel()
                            kk_d_k   =k_modulus   [1:k_max-kz+1,N-ky_max_lim:kyupper,kxlower:N-kx_min_lim].ravel()

                            kxlower=max(N//2-k_max,kx+k_max+1)
                            kxupper=min(N//2+k_max+1,N//2+kx+1)
                            kylower=max(N//2-k_max,ky-N//2)
                            kyupper=min(ky-k_max,N//2+k_max+1)
                            delta_d_kmk=delta_ini[kz+1:k_max+1,kylower:kyupper,kxlower:kxupper].ravel()
                            kk_d_kmk   =k_modulus   [kz+1:k_max+1,kylower:kyupper,kxlower:kxupper].ravel()
                            kern_bis_d=kern(kk_d_k,kk_d_kmk,k_norm,kern_arg1,kern_arg2)
 
                            Int[kz,ky,kx]=(np.nansum(delta_pos_kmk[::-1]*delta_pos_k*kern_bis_pos)\
                              +np.nansum(np.conjugate(delta_neg_kmk)*delta_neg_k*kern_bis_neg)\
                              +np.nansum(delta_d_kmk*np.conjugate(delta_d_k)*kern_bis_d))*dk**3
                            
                    else:
                        if ky>N//2:
                            kxupper=N-max(kx-k_max,N//2+k_max+1)
                            kyupper=N-max(ky-k_max,N//2+k_max+1)
                            delta_d_k=delta_ini[1:k_max-kz+1,N-ky_max_lim:kyupper,N-kx_max_lim:kxupper].ravel()
                            kk_d_k   =k_modulus   [1:k_max-kz+1,N-ky_max_lim:kyupper,N-kx_max_lim:kxupper].ravel()

                            kxlower=max(N//2-k_max,kx-N//2)
                            kxupper=min(kx-k_max,N//2+k_max+1)
                            kylower=max(N//2-k_max,ky-N//2)
                            kyupper=min(ky-k_max,N//2+k_max+1)
                            delta_d_kmk=delta_ini[kz+1:k_max+1,kylower:kyupper,kxlower:kxupper].ravel()
                            kk_d_kmk   =k_modulus   [kz+1:k_max+1,kylower:kyupper,kxlower:kxupper].ravel()
                            kern_bis_d=kern(kk_d_k,kk_d_kmk,k_norm,kern_arg1,kern_arg2)
 
                            Int[kz,ky,kx]=(np.nansum(delta_pos_kmk[::-1]*delta_pos_k*kern_bis_pos)\
                              +np.nansum(np.conjugate(delta_neg_kmk)*delta_neg_k*kern_bis_neg)\
                              +np.nansum(delta_d_kmk*np.conjugate(delta_d_k)*kern_bis_d))*dk**3

                        else:
                            kxupper=N-max(kx-k_max,N//2+k_max+1)
                            kylower=N-min(ky+k_max+1,N//2-k_max)
                            delta_d_k=delta_ini[1:k_max-kz+1,kylower:N-ky_min_lim,N-kx_max_lim:kxupper].ravel()
                            kk_d_k   =k_modulus   [1:k_max-kz+1,kylower:N-ky_min_lim,N-kx_max_lim:kxupper].ravel()

                            kxlower=max(N//2-k_max,kx-N//2)
                            kxupper=min(kx-k_max,N//2+k_max+1)
                            kylower=max(N//2-k_max,ky+k_max+1)
                            kyupper=min(N//2+k_max+1,N//2+ky+1)
                            delta_d_kmk=delta_ini[kz+1:k_max+1,kylower:kyupper,kxlower:kxupper].ravel()
                            kk_d_kmk   =k_modulus   [kz+1:k_max+1,kylower:kyupper,kxlower:kxupper].ravel()

                            kern_bis_d=kern(kk_d_k,kk_d_kmk,k_norm,kern_arg1,kern_arg2)
 
                            Int[kz,ky,kx]=(np.nansum(delta_pos_kmk[::-1]*delta_pos_k*kern_bis_pos)\
                              +np.nansum(np.conjugate(delta_neg_kmk)*delta_neg_k*kern_bis_neg)\
                              +np.nansum(delta_d_kmk*np.conjugate(delta_d_k)*kern_bis_d))*dk**3

                    # For some reason, if I set parallel=True, the following raise an error :
                            # TypeError: Failed in nopython mode pipeline (step: convert to parfors)
                            # object of type 'NoneType' has no len()
                    # To fix it, I had to writte into each previouse conditions Int[kz,ky,kx]=(...)
                    # I dont know why !
                    
                    #Int[kz,ky,kx]=(np.nansum(delta_pos_kmk[::-1]*delta_pos_k*kern_bis_pos)\
                    #   +np.nansum(np.conjugate(delta_neg_kmk)*delta_neg_k*kern_bis_neg))
                       #+np.nansum(delta_d_kmk*np.conjugate(delta_d_k)*kern_bis_d))*dk**3
    return Int

def integre(which,klin,N,k_max,klin_grid,delta_ini,kern,kern_arg1,kern_arg2):
    '''
        inputs:
        -which: whether it is 'TT','LL', 'SL' or 'SS' to be computed. The 'SS' is not implemented in this version of the code yet
        -klin: one dimensional list of k coordinate to be considered
        -N: size of the grid
        -k_max: cut-off in Mpc^-1
        -klin_grid: sparse grid of kx,ky,kz
        -delta_ini: linear density
        -kern: which kernel function to use. Either Kernel_song or Kernel_analytic
                if Kernel_song:
                    kern_arg1 has to be the SONG output delta_cdm and kern_arg2 has to be flatidx
                if Kernel_analytic:
                    kern_arg1 and kern_arg2 have top be the two last output of function trans():
                                tr_pi=potential transfert function and dTdk its derivativ wrt k. 
                                It will be interpolated in squeezed_term() 


        Computation is split in three part in the space like in first_order_fields() 
        in order to compute only independent quantity regarding reality conditions

        -kz>=1
        -kz=0,ky>N//2
        -kz=0,ky=N//2,kx>N//2
        '''
    Int=np.zeros(delta_ini.shape,dtype=np.complex64)
    k_modulus=(np.sqrt(klin_grid[0][N//2:]**2+klin_grid[1]**2+klin_grid[2]**2))
    k_max=int(np.where(klin==k_max)[0])-N//2
    klin=np.arange(len(klin))-N//2

    if which=='SL':
        print('integre SL ...')
        for opt in range(3):
            if opt==0:
                xidx=np.arange(N)
                yidx=np.arange(N)
                zidx=np.arange(1,N//2+1)
            elif opt==1:
                xidx=np.arange(N)
                yidx=np.arange(N//2+1,N)
                zidx=np.arange(1)
            elif opt==2:
                xidx=np.arange(N//2+1,N)
                yidx=np.array([N//2])
                zidx=np.arange(1)

            delta2_SL(Int,N,xidx,yidx,zidx,k_modulus,k_max,delta_ini,kern,kern_arg1,kern_arg2)

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

            delta2_TT_LL(Int,N,xidx,yidx,zidx,k_modulus,k_max,delta_ini,kern,kern_arg1,kern_arg2)
    return Int/2**3/np.pi**3

####################################################################################### 
####################################################################################### mode grid
def k_distrib(k_min,N,klbd,absolute=True):
    '''
        Inputs:
            -k_min: Minimum mode to be consider. Setting k_min automatically set also the step dk=k_min 
             because in order for k-k1 to be always on the grid k1, we need to include 0 and to have a 
             constant step dk. 
            -N size of the grid. In order to include 0, it has to be odd. If it is not,
             we automatically set N+=1
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

####################################################################################### 
####################################################################################### 
####################################################################################### Main

N=30                   # Number of mode to be considered (on laptop N=10 is quick enough)
print('N='+str(N))     #
kmin=np.float32(0.001) # The non-zero smallest mode  
dk=kmin                # which has to be equal to the step of the grid

klin_concat,kmax,N,dk,klambda=k_distrib(kmin,N,0.01) # Generate the list of mode coordinate 

delta_cdm,k1,k2,k3,flatidx,dk12,dk3,delta_cdm1,d_eta=song_main(kmax,kmin,N,'lin',\
              ['sources_song_z000.dat','sources_song_z001.dat']) # Get song outputs

Primordial,transcdm,transphi,dTdk=trans()                        # Get transfer functions and primordial power spectrum 
k_grid_lin=np.array(np.meshgrid(klin_concat,klin_concat,\
            klin_concat,sparse=True,indexing='xy'),dtype=object) # mode Grid 
phi=first_order_fields(k_grid_lin,Primordial)                    # first order stocastic density

# For reproducability tests use the same realisation of the first order density
#np.save('phi',phi)    
#phi=np.load('phi.npy')

k_grid_lin=np.array(np.meshgrid(klin_concat,klin_concat,klin_concat,sparse=True,indexing='ij'),dtype=object) # mode Grid with 'ij' indexing

#TT =integre('TT',klin_concat,N,klambda,k_grid_lin,phi,transphi,dTdk)                 # total integrale  
#LL =integre('LL',klin_concat,N,klambda,k_grid_lin,phi,Kernel_analytic,transphi,dTdk) # LL analytic approx int
#SL =integre('SL',klin_concat,N,klambda,k_grid_lin,phi,Kernel_analytic,transphi,dTdk) # SL analytic approx int

################################################################################## displacement field computation
print('displacement:')                                                           #
xi2,_,_,_=deltacdm2xi(delta_cdm,k1,k2,k3,flatidx)                                # Compute the second order kernels xi2
xi_LL=integre('LL',klin_concat,N,klambda,k_grid_lin,phi,Kernel_song,xi2,flatidx) # SONG Intergale
xi_SL=integre('SL',klin_concat,N,klambda,k_grid_lin,phi,Kernel_song,xi2,flatidx) #
xi1 = phi2fields('xi',phi,k_grid_lin,transcdm,transphi)                          # first order displacement field
                                                                                 #
hf = h5py.File('xi.h5', 'w')                                                     # Save in h5 format 
hf.create_dataset('xi', data=xi1+xi_LL+2*xi_SL)                                  # 
hf.close()                                                                       #
################################################################################## 

################################################################################# scalar velocity computation
print('Scalar velocity:')                                                       #
xi2_1,_,_,_=deltacdm2xi(delta_cdm1,k1,k2,k3,flatidx)                            # Compute the second order kernels velocity v2
v2 = (xi2_1-xi2)/d_eta                                                          #
v_LL=integre('LL',klin_concat,N,klambda,k_grid_lin,phi,Kernel_song,v2,flatidx)  # SONG Intergale
v_SL=integre('SL',klin_concat,N,klambda,k_grid_lin,phi,Kernel_song,v2,flatidx)  #
v1  =phi2fields('v',phi,k_grid_lin,transcdm,transphi)                           # first order velocity
                                                                                #
hf = h5py.File('v.h5', 'w')                                                     # Save in h5 format 
hf.create_dataset('v', data=v1+v_LL+2*v_SL)                                     # 
hf.close()                                                                      #
#################################################################################