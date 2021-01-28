import numpy as np
####################################################################################### 
####################################################################################### Parameters

song_path='/home/thomas/song/'                  # set path to song
gev_path ='/home/thomas/song/gevolution-1.2/'   # set path to gevolution

h=0.67556
omega_b=0.022032/h**2
omega_cdm=0.12038/h**2 
omega_k=0
z_song=[100] # second one for time derivative (velocity)
z=z_song[0]
fnl=0
A_s = 2.215e-6
n_s = 0.9619
k_pivot = 0.05 # 1/Mpc

run='d3' # song: force computation of SONG only
         # SL or LL or d3: if SONG output is found with the right N_song it loads it
         #                 if not, it run SONG
         # LL: compute LL part (complexity almost constant wrt N)
         # SL: compute SL part (complexity N**3*Nl**3, parallelised if paral= True )
         # d3: compute SL and LL  
         # gevolution: Creates the right settings files (one with basic IC, one with SONG IC)
         # TT and LL+ (to be tested)
         
N=512                     # grid size                                
kmin=np.float32(0.0002)  # The non-zero smallest mode  h/Mpc

kl=np.float32(0.005)     #  cut off k_lambda    h/Mpc
paral= True             # Parallelization of SL and song2xi_numba
numeric=True             # True: uses SONG interpolation, False: uses analytic approx (only density)
interp='lin'             # 'lin' or 'nearest'
field=['delta']          # 'delta':song output,'xi': displacement field,'phi': potential, 
                         # 'phip': potential time derivative ,'chi': phi-psi , 'v': scalar velocity
                         

order=2                               # first-only or first+second order computation
N_realization=1                       # Number of realisation 
N_song_k12=N+1                        # Song grid precision
N_song_k3 =N+1                        # Song grid precision
kmax_song=np.float32(2*(N+1)*kmin)/h  # maximum value for SONG    h/Mpc

bug=False  # Disable Numba, refresh environment to return to False
           # If Numba bug: try to disable it, it is often a simple problem in python
           # If the code run well, still try to run without Numba, 
           #   you could see a problem that numba do not raise (eg out of range)
