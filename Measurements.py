#===============================================================================
 #Name        : Measurements.py
 #Authors     :
 #Version     :
 #Description :
#===============================================================================

import numpy as np
import h5py
import readgadget
import MAS_library as MASL
import Pk_library as PKL
from scipy import interpolate
import matplotlib.pyplot as plt


#===============================================================================
#                               PRELIMINARIES
#===============================================================================

def initializeGlobals():
    '''Initialize global parameters
    '''
    global gridSize, numFiles, filenamesArray
    global BoxSize, kf, ptypes, do_RSD, axis, MAS, threads, verbose
    global configuration, ihard, isoft, output
    global h, A_s, k_pivot, n_s, redshift

    ## Parameters for the density file:
    gridSize = 128                 # The density field will be a 3D float numpy array with grid**3 cells.
    numFiles = 4                   # Number of realizations of the overdensity.
    filenamesArray = []            # Array of paths to density files.
    for i in range(numFiles):
        filenamesArray.append('density1_%i.h5'%i)
        filenamesArray.append('density2_%i.h5'%i)

    ## Parameters for pylians3 usage:
    BoxSize =  628.3185447619725   # Mpc/h , This fixed the all the units in pylians3 measurements (k:{h/Mpc})
    #kf= 2*np.pi/BoxSize  # fundamental frequency:
    kf      = 0.01        # This is fixed for Thomas's files
    ptypes  = [1]         # [1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)
    do_RSD  = False       # If True, particles positions will be moved to redshift-space along the axis axis.
    axis    = 0           # Axis along which redshift-space distortions will be implemented (only needed if do_RSD=True): 0, 1 or 2 for x-axis, y-axis or z-axis, respectively.
    MAS     = 'None'      # Mass-assignment scheme, possible choices are: 'CIC' 'NGP' 'TSC' 'PCS' if the density field has not been generated with any of these set it to 'None'
    threads = 4           # Number of openmp threads to be used.
    verbose = False        # Whether print information on the status/progress of the calculation: True or False

    ## Triangle configuration for the bispectrum measurements:
    configuration = 'squeezed'  #possible choices are: squeezed' or 'all'
    ihard         = 40          #Hard modes to compute (It can be put from 1 to grid/3 in units of fundamental frequency. It gives the values for k1/kf and k2/kf)
    isoft         = 40          #Soft modes to compute (This are the values of k3/kf)

    ##output directory to save the powerspectrum and bispectrum measurements files:
    output   = 'output/'
    songpath = '/home/juan/z.Cosmo_tools/song/python'

    ## Cosmological parameters:
    h       = 0.67556
    A_s     = 2.215e-09
    k_pivot = 0.05     #in 1/Mpc
    n_s     = 0.9619
    redshift = 100

def read_delta_file():
    ''' Read the density file that contains overdensity delta.
        delta format: It should be a 3D float numpy array such delta = np.zeros((grid, grid, grid), dtype=np.float32)

        filenameArray: list of strings. global parameter, see initializeGlobals().
    Returns:
        deltaArray : array_like
            Size = number of realizations.
    '''
    deltaArray = np.empty((numFiles,gridSize,gridSize,gridSize), dtype=np.float32)

    for i,filename in enumerate(filenamesArray):
        print("Reading overdensity file: %s \t" %(filename,))
        try:
            with h5py.File(filename, "r") as f:
                # Find keys
                #print("Keys: %s" % f.keys())
                file_keys = list(f.keys())[0]
                # Get the data
                delta = np.array(f[file_keys], dtype=np.float32)
            ## This is to read delta^(1) + delta^(2):
            if i%2==0:
                deltaArray[(int)(i//2)] = delta
            else:
                deltaArray[(int)(i//2)] += delta
        except OSError:
            print("Overdensity File Not Found.\t")
            exit()

    return deltaArray

def measure_Overdensity(snapshot):
    ''' Measure delta from a snapshot using pylians3 mass-assignment scheme
    '''
    print("Reading snapshot file: %s \n" %(snapshot,))
    try:
        # read header
        header   = readgadget.header(snapshot)
        Boxsize  = header.boxsize#/1e3     #Mpc/h. For gadget2 snapshots uncomment /1e3
        Nall     = header.nall             #Total number of particles
        Masses   = header.massarr*1e10     #Masses of the particles in Msun/h format:[1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)
        ptype    = [Nall.nonzero()[0][0]] #[1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)
    except OSError:
        print("Error: Overdensity File Not Found.\t")
        exit()
    print('Checking snapshot header data with initializeGlobals() they should agree.\t')
    print('Boxsize:',BoxSize,Boxsize)
    print('Particle type', ptypes,ptype)

    rho = MASL.density_field_gadget(snapshot, ptypes, gridSize, MAS, do_RSD, axis)
    delta = rho/np.mean(rho, dtype=np.float64);  delta -= 1.0   #overdensity

    return delta

def makeDensityfile(delta):
    '''Transform real delta into phase space delta for py-pow code.
       The output files are saved in .npz file for each realization. format is a 3D array of size delta = (grid, grid, grid)

        delta : array_like
            input data, shape = (numfiles, gridsize, gridsize, gridsize)
    '''
    for ext in range(numFiles):
        #kf       = 0.01                        # h/Mpc
        #kf      = 2*np.pi/BoxSize             # h/Mpc
        kN       = kf*gridSize/2               # h/Mpc Nyquist frequency
        npticles = (16**3)*64                  #262144  (tiltfactor^3) * N_templete

        outdfile = 'output_pypow/density1bin_grid_%i_mm_z%i.00_gevolution_%i'%(gridSize,redshift,ext)

        print("Saving: %s"%outdfile)
        header = np.array([gridSize, npticles, kf, kN])
        deltaFFT = np.fft.fftn(delta[ext])

        np.savez(outdfile, header = header, data = deltaFFT)

def computePower(delta):
    ''' Measure Powerspectrum. The output is saved in a txt file in the folder
        specified by output in initializeGlobals().

        file format: k | P0(k)

        delta : array_like
            input data, shape = (numfiles, gridsize, gridsize, gridsize)
    '''
    for ext in range(numFiles):
        filename = output+'powerspectrum%i.dat'%ext
        print('\n Computing power, saving at '+filename)

        Pkpyl = PKL.Pk(delta[ext], BoxSize, axis, MAS, threads, verbose)

        np.savetxt(filename,np.vstack([Pkpyl.k3D, Pkpyl.Pk[:,0]]).T)

def pylians3Biscpetrum(delta,k1,k2,theta):
    ''' Get data from pylians3 Bispectrum measurements.
        returns: array_like
            k1/kF | k2/kF | k3/kF | P(k1) | P(k2) | P(k3) | B(k1,k2,k3) | Number of triangles

        delta : array_like
            input data, shape = (numfiles, gridsize, gridsize, gridsize)
        k1,k2: fixed wavenumber in units of the fundamental frequency.
            integer number.
        theta: angle between k1 and k2.
    '''
    Bk = PKL.Bk(delta, BoxSize, k1*kf, k2*kf, theta, MAS, threads)
    BSk_pyl = Bk.B        #bispectrm
    triangle_conf = Bk.triangles  #triangle counts
    k3_pyl  = Bk.k[2:]    #k-bins for power spectrum
    Pk1_pyl = Bk.Pk[0]    #power spectrum PS(k1)
    Pk2_pyl = Bk.Pk[1]    #         ´´    PS(k2)
    Pk3_pyl = Bk.Pk[2:]   #         ´´    PS(k3)

    lenn = len(theta)
    return  np.vstack([np.full(lenn,k1),np.full(lenn,k2), k3_pyl/kf, np.full(lenn,Pk1_pyl), np.full(lenn,Pk2_pyl), Pk3_pyl, BSk_pyl, triangle_conf])

def computeBispectrum(delta):
    '''Bispectrum measurement. The output is saved in a txt file  in the folder
       specified by output in initializeGlobals().

       BISPECTRUM FILES:
        Columns: k1/kF | k2/kF | k3/kF | P(k1) | P(k2) | P(k3) | B(k1,k2,k3) | Number of triangles
    '''

    k1 = np.arange(1,ihard,1)
    k2 = np.arange(1,ihard,1)
    k3_soft = np.arange(1,isoft,1)

    for ext in range(numFiles):
        filename = output+'bispectrum%i.dat'%ext
        myfile = open(filename, 'w')
        format = '%i %i %e %e %e %e %e %e'

        print('\n Computing %s triangle configuration and saving in file %s'%(configuration,filename))

        if configuration == 'squeezed':
            total = len(k1)
            count = 1
            for k11 in k1:
                print("\n Progress %i/%i:\t" %(count,total))
                theta = np.arccos( 0.5*(k3_soft/k11)**2 - 1)  ##use for k1=k2
                theta = theta[~np.isnan(theta)]
                lenn = len(theta)

                if lenn>0:
                    data = pylians3Biscpetrum(delta[ext], k11, k11, theta)
                    np.savetxt(myfile,data.T,fmt = format)
                    count+=1
                else:
                    count+=1
        else:
            total = len(k1)*len(k2)
            count = 1
            for k11 in k1:
                for k22 in k2:
                    print("\n file number:%i, Progress %i/%i:\t" %(ext,count,total))
                    #set the angle (k1,k2) ##use for all trinagle configuration
                    theta = np.arccos(0.5*(k3_soft**2 - k11**2 - k22**2)/k11/k22)
                    theta = theta[~np.isnan(theta)]
                    lenn = len(theta)
                    if lenn>0:
                        data = pylians3Biscpetrum(delta[ext], k11, k22, theta)
                        np.savetxt(myfile,data.T,fmt = format)
                        count+=1
                    else:
                        count+=1

        myfile.close()

def classPowerspectrum():
    ''' Compute the linear powerspectrum from the transfer function of class.
        needed in Bispectrum_song
    '''
    dataClass=np.loadtxt('class_tk.dat')

    k_lin = dataClass[:,0]  ##in h/Mpc
    Tk = dataClass[:,3]

    Pk_lin= 2*np.pi**2*A_s/k_lin**3*(k_lin/(k_pivot/h))**(n_s-1)*Tk**2

    return k_lin, Pk_lin

##F2 and Bispectrum_theory are taken from pylians3 libraries  ~/Pylians3/library/Pk_library
def F2(k1_vec, k2_vec):
    ''' Newtonian second order kernel F2(k1,k2)
    '''
    k1_mod = np.sqrt(np.dot(k1_vec, k1_vec))
    k2_mod = np.sqrt(np.dot(k2_vec, k2_vec))
    ctheta = np.dot(k1_vec, k2_vec)/(k1_mod*k2_mod)
    return 5.0/7.0 + 1.0/2.0*ctheta*(k1_mod/k2_mod + k2_mod/k1_mod) + 2.0/7.0*ctheta**2

def Bispectrum_theory(k,Pk,k1,k2):
    ''' Compute the tree level bispectrum F2N*PS*PS + perm.
    k,Pk: linear powerpsectrum
    k1,k2: wavenumber in h/Mpc
    '''

    bins = 50
    B = np.zeros(bins, dtype=np.float64)
    ks = np.zeros(bins, dtype=np.float64)

    thetas = np.linspace(0, np.pi, bins)
    k1_vec = np.array([0, 0, k1])

    Pk1 = np.interp(np.log(k1), np.log(k), Pk)
    Pk2 = np.interp(np.log(k2), np.log(k), Pk)

    for i,theta in enumerate(thetas):
        k2_vec = np.array([0,  k2*np.sin(theta), k2*np.cos(theta)])
        k3_vec = np.array([0, -k2*np.sin(theta),-k2*np.cos(theta)-k1])
        k3_norm = np.sqrt(np.dot(k3_vec, k3_vec))
        Pk3 = np.interp(np.log(k3_norm), np.log(k), Pk)

        F2_12 = F2(k1_vec, k2_vec)
        F2_13 = F2(k1_vec, k3_vec)
        F2_23 = F2(k2_vec, k3_vec)

        B[i] = 2.0*Pk1*Pk2*F2_12 + 2.0*Pk1*Pk3*F2_13 + 2.0*Pk2*Pk3*F2_23
        ks[i]=k3_norm

    return ks, B

def songBispectrum(k_lin,Pk_lin,k1,k2,k3_array):
    ''' Compute the bispectrum using song kernel K2*PS*PS + perm.
    k_lin,Pk_lin: linear powerpsectrum
    k1,k2,k3_array: wavenumber in h/Mpc
    '''

    sys.path.insert(1, songpath)  #song path
    import songy
    #read song output file
    data = songy.FixedTauFile('sources_song_z000_N129.dat')
    #get sources:
    T1 = data.first_order_sources[b'delta_cdm']       #first order transfer function
    K = np.array(data.get_source(b'delta_cdm'))  #second order kernel *T1 * T1
    #grid of wavenumber:
    k1_ker = np.array(data.k1)/h
    k2_ker = np.array(data.k2,dtype=object)/h
    k3_ker = np.array(data.k3)/h
    flatidx = data.flatidx
    #step of the grid:
    dk12=(data.k1[1]-data.k1[0])/h
    dk3=(np.diff(data.k3)[:,0])/h
    dk = np.float32(kf)

    def Kernel_song(kk1,kk2,kk3):
        '''kk1,kk2,kk3 in h/Mpc
        '''
        k1_idx=np.int(np.around((kk1-dk)/dk12))
        k2_idx=np.int(np.around((kk2-dk)/dk12))
        k3_idx=np.int(np.around((kk3-k3_ker[flatidx[k1_idx,k2_idx]][0])/dk3[flatidx[k1_idx,k2_idx]]))
        if kk3>k3_ker[flatidx[k1_idx,k2_idx]][-1]:
            k3_idx=-1

        return K[flatidx[k1_idx,k2_idx]][k3_idx]/T1[k1_idx]/T1[k2_idx]

    def Bispectrum_song(k_lin,Pk_lin,k1,k2,k3):
        '''k1,k2,k3 in h/Mpc
        '''
        Pk1 = np.interp(k1, k_lin, Pk_lin)
        Pk2 = np.interp(k2, k_lin, Pk_lin)
        Pk3 = np.interp(k3, k_lin, Pk_lin)

        K2_12 = Kernel_song(k1, k2, k3)
        K2_13 = Kernel_song(k1, k3, k2)
        K2_23 = Kernel_song(k2, k3, k1)

        return 2.*Pk1*Pk2*K2_12 + 2.*Pk1*Pk3*K2_13 + 2.*Pk2*Pk3*K2_23

    BS_song = [np.abs(Bispectrum_song(k_lin,Pk_lin,k1,k2,kk3)) for kk3 in k3_array]

    return k3_array, BS_song

def plotPowerSpectrum():
    ''' Tool to plot powerspectrum
    '''
    ps_array = []
    ks_array = []

    for ext in range(numFiles):
        filename = output+'powerspectrum%i.dat'%ext
        k, pk = np.loadtxt(filename, unpack=True)
        ps_array.append(pk)
        ks_array.append(k)

    Norm = numFiles
    Pk_pyl     = np.mean(np.array(ps_array), 0)
    Pk_pyl_err = np.std(np.array(ps_array), 0)/np.sqrt(Norm)
    k_pyl      = ks_array[0]

    k_lin,Pk_lin = classPowerspectrum()

    #Plot
    fig, (ax1, ax2)= plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})
    ax1.set_title("PowerSpectrum z=%s"%(redshift), fontsize = 24)
    ax1.plot(k_lin,Pk_lin,label = r"$PS_{Class}$")
    ax1.errorbar(k_pyl,Pk_pyl,Pk_pyl_err,linestyle = 'none',capsize=2, capthick=1,label = r"$PS_{Pylians}$")
    ax1.set_ylabel( "PS(k)", fontsize = 14)
    ax1.set_xlim(1e-2,2)
    ax1.set_ylim(1e-3,1e2)
    ax1.set_xticklabels([])
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()

    Pk_lin_int = interpolate.interp1d(k_lin, Pk_lin)

    ax2.semilogx(k_pyl,np.abs(1-Pk_pyl/Pk_lin_int(k_pyl)),label='$Lin/Pyl3$')
    ax2.set_xlabel(r'$k[h \; Mpc^{-1}]$' , fontsize = 14)
    ax2.set_ylabel(r'$\Delta P(\%)$',size="large")
    ax2.set_xlim(1e-2,2)
    ax2.set_ylim(0, 0.2)
    ax2.legend()

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('PS_contrast.svg')

def plotBisceptrum(k1):
    ''' Tool to plot the squeezed bispectrum  BS(k1,k1,k)
    k1: fixed wavenumber in units of the fundamental frequency.
        integer number.
    '''
    bs_array = []
    k3_array = []

    for ext in range(numFiles):
        filename = output+"bispectrum%i.dat" %ext
        k1_d, k2_d, k3_d, pk1, pk2, pk3, b123_d, cnts = np.loadtxt(filename, unpack=True)

        mask = (k1_d == (float)(k1))&(k2_d == (float)(k1))

        k3_array.append(k3_d[mask])
        bs_array.append(b123_d[mask])

    N = numFiles
    Bk_pyl     = np.mean(np.array(bs_array), 0)
    Bk_pyl_err = np.std(np.array(bs_array), 0)/np.sqrt(N)
    bk_pyl     = k3_array[0]*kf

    ## Theoretical Bispectrum Bispectrum from the data
    k_lin,Pk_lin         = classPowerspectrum()
    ks_exactN, BS_exactN = Bispectrum_theory(k_lin,Pk_lin,k1*kf,k1*kf)
    ks_song, BS_song     = songBispectrum(k_lin,Pk_lin,k1*kf,k1*kf,bk_pyl)

    #Plot:
    #fig, (ax1, ax2)= plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})
    fig, ax1= plt.subplots(1, 1)
    ax1.set_title( 'Bispectrum z=%s'%(redshift), fontsize = 24 )
    ax1.plot(ks_song,BS_song,'--',label = "BS_song")
    ax1.plot(ks_exactN,BS_exactN,label = "BS_Newtonian")
    ax1.errorbar(bk_pyl,np.abs(Bk_pyl),Bk_pyl_err,capsize=2, capthick=1,label = "BS_Pylians")

    ax1.set_xlabel( r'$k[h \; Mpc^{-1}]$' , fontsize = 14 )
    ax1.set_ylabel( r'$ | B(k_1,k_1,k) | $' , fontsize = 14 )
    ax1.set_xlim(bk_pyl[0],bk_pyl[-1])
    ax1.set_ylim(1e-2,1e3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    textstr = r'$k_1=%.2f$' % (k1*kf, )
    ax1.text(0.05, 0.05,textstr, transform=ax1.transAxes, fontsize = 12)

    ax1.legend()

    #plt.subplots_adjust(wspace=0, hspace=0)

    #ax2.set_xlabel(r'$k[h \; Mpc^{-1}]$',size="large",fontweight='bold')
    #ax2.set_ylabel(r'$\Delta B(\%)$',size="large")
    #ax2.set_xlim(bk_pyl[0],bk_pyl[-1])
    #ax2.set_xscale('log')

    plt.savefig("BS_Song_p%s.svg" %(str)(k1*kf)[2:4])

#===============================================================================
#                                   MAIN
#===============================================================================

if __name__ == '__main__':
    import sys

    n = sys.argv[1]

    initializeGlobals()

    if n=='compute':
        print('Measuring PowerSpectrum and Bispectrum')
        deltaArray = read_delta_file()
        #makeDensityfile(deltaArray)
        computePower(deltaArray)
        computeBispectrum(deltaArray)

    elif n=='plot':
        print('Making PowerSpectrum plot\t')
        plotPowerSpectrum()
        k1_range = np.arange(10,45,5)
        for k1 in k1_range:
            print('Making Bispectrum plot with k1 = %.2f fixed in the %s configuration\t'%(k1*kf,configuration))
            plotBisceptrum(k1)
    else:
        print('Error: Incorrect arguments.\t')
        print('Usage: Bispectrums.py <text>\t')
        print('<text> = "compute" or "plot". Use "compute" to make measurements of the powerspectrum and bispectrum \
            "plot" to make plots from measurement files (.dat).\t')
        exit()
