###############################################################################
##############################-----IMPORTS-----################################
###############################################################################



import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import halfnorm
from scipy.stats import rayleigh
from scipy.stats import cumfreq
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.integrate import odeint
from scipy.integrate import simps
from scipy.interpolate import UnivariateSpline as US
from scipy import interpolate
import time
from datetime import date
import os
import copy
import warnings
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

plt.rcParams.update({'font.family':'sans-serif'})

pd.set_option('mode.chained_assignment', None)
warnings.filterwarnings('ignore', message="Pandas doesn't allow columns to be created via a new attribute name")



###############################################################################
#########################-----SAVING PREFERENCES-----##########################
###############################################################################



parent_dir = "/Users/mac/Documents/Working Research"
main_sims_dir = "/Users/mac/Documents/Working Research/Main Simulation Collection"
cf_sims_dir = "/Users/mac/Documents/Working Research/CF Simulation Collection"
sims_dir = "/Users/mac/Documents/Working Research/Miscellaneous Simulations"
figures_dir = "/Users/mac/Documents/Working Research/Figures for Paper"



###############################################################################
#############################-----CONSTANTS-----###############################
###############################################################################



R_sun = 6.957*10**8
M_sun = 1.99*10**30
M_earth = 5.972*10**24
G = 6.67*10**-11
R_star = 0.5*R_sun
M_star = 0.5*M_sun
T_star = 3250
k = (2*(6.371*10**6))/R_star
ihz = 663900000 ## Meters
ohz = 776100000 
m_per_AU = 1.496e11

m_dict = {0.1:{'slow':[-0.1392174705, -0.096173446], 'medium':[-0.141666667], 'fast':[-0.21651701]},
          0.2:{'slow':[-0.1092343614], 'medium':[-0.1432255834], 'fast':[-0.2249777586]},
          0.4:{'slow':[-0.11666667], 'medium':[-0.184286399, -0.0515942558], 'fast':[-0.227764652, -0.0225378778]}}
b_dict = {0.1:{'slow':[1.402957235, 0.9923985071], 'medium':[1.375000002], 'fast':[1.779102084]},
          0.2:{'slow':[1.18964053], 'medium':[1.370804667], 'fast':[2.349866552]},
          0.4:{'slow':[1.525], 'medium':[2.065718394, 0.8843483022], 'fast':[2.226587909, 0.6303030224]}}
c_dict = {0.1:{'slow':0.052, 'medium':0.09, 'fast':0.095},
          0.2:{'slow':0.2065312774, 'medium':0.225, 'fast':0.15},
          0.4:{'slow':0.38421565486, 'medium':0.41, 'fast':0.45}}
p_dict = {0.1:{'slow':[10**9.539, 6*10**9], 'medium':[10**9.071], 'fast':[6*10**7]},
          0.2:{'slow':[10**9], 'medium':[10**8], 'fast':[6*10**9]},
          0.4:{'slow':[6*10**9], 'medium':[8*10**8, 10**9.194], 'fast':[6*10**7, 10**8]}}
markers_dict = {'slow':'-', 'medium':'--', 'fast':':'}
colors_dict =  {0.1:'red', 0.2:'purple', 0.4:'blue'}

NSIMS = {'inherent':1000000, 'kepler':2500, 'tess':70000}
colors_master = {'Single':'green', 'Multi':'goldenrod', 'Disrupted':'firebrick', 'Intact':'cornflowerblue', 'Both':'grey'}

randomWalkNums = np.genfromtxt('eval_pdf_2.0post_equal_weights.dat')
firstPopulation = randomWalkNums[:,:2]
secondPopulation = randomWalkNums[:,2:4]

Y = np.loadtxt("ajaaf477t2_mrt.txt")
radii = Y[:,12]
masses = Y[:,11]

completenesshalfsunmassdata = np.load('Completeness_0.5Msun.npy')
def interpolation(a):
    x = np.logspace(-0.22,5.77,13, base=np.e) #Period
    y = np.logspace(-1.22,1.33,17, base=np.e) #Radii
    xx, yy = np.meshgrid(x, y)
    z = completenesshalfsunmassdata[a]
    f = interpolate.interp2d(x, y, z, kind='linear')
    return f
completenesshalfsunmassfunc = [interpolation(i) for i in range(len(completenesshalfsunmassdata))]

kepler_nt_counts = [76, 18, 12, 5, 4]



###############################################################################
###########################-----LoI FUNCTIONS-----#############################
###############################################################################



def LoI(IF, decay, A, A_bounds=(1e6,1e10)):
    
    if not hasattr(A, '__len__'): A = [A]    
    if isinstance(A, list): A = np.array(A)
    
    m = m_dict[IF][decay]                           ## Slope values
    b = b_dict[IF][decay]                           ## Intercept values
    c = c_dict[IF][decay]                           ## Constant values
    pivots = p_dict[IF][decay]                      ## Pivot locations
    n_segments = len(pivots) + 2
    
    L = np.repeat(np.nan, len(A))                   ## Likelihood values
    I = np.empty((n_segments, len(A)), dtype=bool)  ## Index array
    
    I[0] =  A < A_bounds[0]
    L[I[0]] = np.ones(sum(I[0]))
    
    I[1] = (A_bounds[0] <= A) & (A < pivots[0])
    L[I[1]] = m[0] * np.log10(A[I[1]]) + b[0]
    
    if len(pivots)==2: 
        I[2] = (pivots[0] <= A) & (A < pivots[1])
        L[I[2]] = m[1] * np.log10(A[I[2]]) + b[1]
        
    I[len(pivots)] = (pivots[-1] <= A)
    L[I[len(pivots)]] = c
    
    return L if len(L)>1 else L[0]



def LoI_Inverse(IF, decay, L, A_bounds=(1e6,1e10)):
    
    if not hasattr(L, '__len__'): L = [L]    
    if isinstance(L, list): L = np.array(L)
    
    m = m_dict[IF][decay]                           ## Slope values
    b = b_dict[IF][decay]                           ## Intercept values
    c = c_dict[IF][decay]                           ## Constant values
    pivots = p_dict[IF][decay]                      ## Pivot locations (ages)
    
    L_pivots = np.zeros(len(pivots) + 1)            ## Pivot locations (LoI values)
    L_pivots[0] = c
    if len(pivots)==2: L_pivots[1] = LoI(IF, decay, pivots[0], A_bounds=A_bounds)
    L_pivots[-1] = LoI(IF, decay, A_bounds[0], A_bounds=A_bounds)
    n_segments = len(L_pivots) + 1
    
    A = np.zeros(len(L))                            ## Ages
    I = np.empty((n_segments, len(L)), dtype=bool)  ## Index array
    
    I[0] = (L <= L_pivots[0])
    A[I[0]] = np.nan
    
    if len(L_pivots)==3:
        I[1] = (L >= L_pivots[0]) & (L <= L_pivots[1])
        A[I[1]] = 10**((L[I[1]]-b[1])/m[1])
        
        I[2] = (L >= L_pivots[1]) & (L <= L_pivots[2])
        A[I[2]] = 10**((L[I[2]]-b[0])/m[0])

    else:
        I[1] = (L >= L_pivots[0]) & (L <= L_pivots[1])
        A[I[1]] = 10**((L[I[1]]-b[0])/m[0])

    I[-1] = (L >= L_pivots[-1])
    A[I[-1]] = 1e6
    
    return A if len(A)>1 else A[0]



###############################################################################
############################-----SIMULATIONS-----##############################
###############################################################################



columns = ['ID#', 
           'Age', 
           'Intact', 
           'Age When Disrupted', 
           'Evolutionary Timescale', 
           'Number of Planets', 
           'Number of Transits', 
           'Number of Tess Detections', 
           'Mean Orbital Plane Inclination', 
           'Disruption Modes', 
           'Evolutionary Timescales', 
           'Periods', 
           'Radii', 
           'Masses', 
           'Inclinations', 
           'Eccentricities', 
           'Semi-Major Axes', 
           'Longitudes of Periastron', 
           'Impact Parameters', 
           'Equilibrium Temperatures', 
           'Transits?', 
           'Detected By Tess?', 
           'Runaway Greenhouse Effects?', 
           'In Habitable Zone?', 
           'Is Life Possible?']



def simulate(IF, decay, CF=0.5, N=2500, i_surface=None, timeit=False, **kwargs):
    
    A_bounds = kwargs.pop('A_bounds', (1e6, 1e10))
    logspace_ages = kwargs.pop('logspace_ages', False)
    
    systems = pd.DataFrame(columns=columns)
    
    t = [time.time()]
    col_order = ['']
    
    ## ID
    systems['ID#'] = np.arange(N)
    t.append(time.time())
    col_order.append('ID#')
    
    ## AGE
    if logspace_ages:
        systems['Age'] = 10**np.random.uniform(*np.log10(A_bounds), N)
    else:
        systems['Age'] = np.random.uniform(*A_bounds, N)
    t.append(time.time())
    col_order.append('Age')
    
    ## INTACT, AGE WHEN DISRUPTED
    L = LoI(IF, decay, systems['Age'])
    r = np.random.random(N)
    systems['Intact'] = r < L
    t.append(time.time())
    col_order.append('Intact')
    
    systems['Age When Disrupted'] = LoI_Inverse(IF, decay, r)
    t.append(time.time())
    col_order.append('Age When Disrupted')
    
    ## EVOLUTIONARY TIMESCALE
    A = systems['Age']
    DA = systems['Age When Disrupted']
    systems['Evolutionary Timescale'] = A - DA
    II = systems['Intact']
    systems['Evolutionary Timescale'][II] = A[II]
    t.append(time.time())
    col_order.append('Evolutionary Timescale')
    
    ## NUMBER OF PLANETS
    rand = np.random.choice(len(secondPopulation), size=N, replace=True)
    systems['Number of Planets'] = np.int_(np.round(secondPopulation[rand, 0])) + 1
    rand_II = np.random.choice(len(firstPopulation), size=N, replace=True)
    systems['Number of Planets'][II] = np.int_(np.round(firstPopulation[rand_II, 0])) + 1
    t.append(time.time())
    col_order.append('Number of Planets')
    
    ## DISRUPTION MODES
    DI = ~systems['Intact']
    NP = systems['Number of Planets']
    collision_numbers = np.random.random(size=(N,15))
    collision = collision_numbers < CF
    collision_letters = np.where(collision, 'C', 'M')
    collision_ragged = [collision_letters[i,:NP[i]] for i in range(N) if DI[i]]
    
    systems['Disruption Modes'][DI] = collision_ragged
    systems['Disruption Modes'][II] = [[np.nan]*NP[i] for i in range(N) if II[i]]
    t.append(time.time())
    col_order.append('Disruption Modes')
    
    ## EVOLUTIONARY TIMESCALES
    A_II = A[II]
    NP_II = systems['Number of Planets'][II]
    intact_ragged = [[A_II[i]]*NP_II[i] for i in range(N) if II[i]]
    systems['Evolutionary Timescales'][II] = intact_ragged
    
    ET_15 = np.repeat(np.array(systems['Evolutionary Timescale']).reshape((N,1)), 15, axis=1)
    A_15 = np.repeat(np.array(A).reshape((N,1)), 15, axis=1)
    disrupted_ET = np.where(collision, ET_15, A_15)
    disrupted_ragged = [disrupted_ET[i,:NP[i]] for i in range(N) if DI[i]]
    systems['Evolutionary Timescales'][DI] = disrupted_ragged
    t.append(time.time())
    col_order.append('Evolutionary Timescales')
    
    ## PERIODS
    period = np.exp(np.random.uniform(np.log(0.75), np.log(300), size=(N,15)))
    period_ragged = [period[i,:NP[i]] for i in range(N)]
    systems['Periods'] = period_ragged
    t.append(time.time())
    col_order.append('Periods')
    
    # RADII, MASSES
    rand = np.array(np.random.choice(len(radii), size=(N,15), replace=True))
    radii_rand = radii[rand]
    radius_ragged = [radii_rand[i,:NP[i]] for i in range(N)]
    systems['Radii'] = radius_ragged
    t.append(time.time())
    col_order.append('Radii')
    
    masses_rand = masses[rand]
    mass_ragged = [masses_rand[i,:NP[i]] for i in range(N)]
    systems['Masses'] = mass_ragged
    t.append(time.time())
    col_order.append('Masses')
    
    ## MEAN ORBITAL PLANE INCLINATION
    systems['Mean Orbital Plane Inclination'] = np.random.uniform(-90, 90, N)
    t.append(time.time())
    col_order.append('Mean Orbital Plane Inclination')
    
    ## INCLINATIONS
    rand_DI = np.random.choice(len(secondPopulation), size=N, replace=True)
    MOPI_DI = systems['Mean Orbital Plane Inclination'][DI]
    NP_DI = NP[DI]
    inclination_deviation_DI_ragged = [np.random.normal(MOPI_DI[i], np.float_(secondPopulation[rand_DI[i], 1]), size=NP_DI[i]) for i in range(N) if DI[i]]
    systems['Inclinations'][DI] = inclination_deviation_DI_ragged
    
    rand_II = np.random.choice(len(firstPopulation), size=N, replace=True)
    MOPI_II = systems['Mean Orbital Plane Inclination'][II]
    NP_II = NP[II]
    inclination_deviation_II_ragged = [np.random.normal(MOPI_II[i], np.float_(firstPopulation[rand_II[i], 1]), size=NP_II[i]) for i in range(N) if II[i]]
    systems['Inclinations'][II] = inclination_deviation_II_ragged
    
    t.append(time.time())
    col_order.append('Inclinations')
    
    ## ECCENTRICITIES
    eccentricity_II = halfnorm.rvs(scale=0.049, size=(N,15))
    eccentricity_II_ragged = [eccentricity_II[i,:NP_II[i]] for i in range(N) if II[i]]
    systems['Eccentricities'][II] = eccentricity_II_ragged
    
    eccentricity_DI = rayleigh.rvs(scale=0.26, size=(N,15))
    while np.any([i>1 for i in eccentricity_DI]):
        eccentricity_DI[eccentricity_DI>1] = rayleigh.rvs(scale=0.26, size=np.sum(eccentricity_DI>1))
    eccentricity_DI_ragged = [eccentricity_DI[i,:NP_DI[i]] for i in range(N) if DI[i]]
    systems['Eccentricities'][DI] = eccentricity_DI_ragged
    
    t.append(time.time())
    col_order.append('Eccentricities')
    
    ## SEMI-MAJOR AXIS
    systems['Semi-Major Axes'] = ((G*((86400*systems['Periods'])**2)*M_star)/(4*np.pi**2))**(1/3)
    t.append(time.time())
    col_order.append('Semi-Major Axes')
    
    ## LONGITUDE OF PERIASTRON
    longitude_periastron = np.random.uniform(0, 360, size=(N,15))
    longitude_periastron_ragged = [longitude_periastron[i,:NP[i]] for i in range(N)]
    systems['Longitudes of Periastron'] = longitude_periastron_ragged
    t.append(time.time())
    col_order.append('Longitudes of Periastron')
    
    ## IMPACT PARAMETER
    a = systems['Semi-Major Axes']
    inc = systems['Inclinations']
    ecc = systems['Eccentricities']
    omega = systems['Longitudes of Periastron']
    systems['Impact Parameters'] = [(a[i] * np.sin((np.pi/180) * inc[i])) * ((1 - ecc[i]**2)/(1 + ecc[i] * np.sin((np.pi/180) * omega[i]))) for i in range(N)]
    t.append(time.time())
    col_order.append('Impact Parameters')
    
    ## EQUILIBRIUM TEMPERATURE
    systems['Equilibrium Temperatures'] = (((R_star)/systems['Semi-Major Axes'])**(1/2))*(((1-0.3)/4)**(1/4))*T_star
    t.append(time.time())
    col_order.append('Equilibrium Temperatures')
    
    ## TRANSIT?, NUMBER OF TRANSIT
    transit = [(np.abs(systems['Impact Parameters'][i]) < R_star) for i in range(N)]
    systems['Transits?'] = transit
    t.append(time.time())
    col_order.append('Transits?')
    
    systems['Number of Transits'] = [np.sum(transit[i]) for i in range(N)]
    t.append(time.time())
    col_order.append('Number of Transits')
    
    ## DETECTED BY TESS?, NUMBER OF TESS DETECTION
    if i_surface is None: i_surface = np.random.choice(len(completenesshalfsunmassfunc))
    f = completenesshalfsunmassfunc[i_surface]
    radius = systems['Radii']
    period = systems['Periods']
    transit = systems['Transits?']
    rand = np.random.random(size=(N,15))
    completeness = [(np.diag( f(period[i], radius[i]) ) if NP[i]>1 else f(period[i], radius[i]) ) for i in range(N)]
    detect = [(rand[i,:NP[i]] < completeness[i]) & transit[i] for i in range(N)]
    systems['Detected By Tess?'] = detect
    t.append(time.time())
    col_order.append('Detected By Tess?')

    systems['Number of Tess Detections'] = [np.sum(detect[i]) for i in range(N)]
    t.append(time.time())
    col_order.append('Number of Tess Detections')
     
    ## RUNAWAY GREENHOUSE EFFECT?
    rg = [(ecc[i] > 0.915*np.exp(-0.003/(a[i]*6.685*10**-12)**2)) for i in range(N)]
    systems['Runaway Greenhouse Effects?'] = rg
    t.append(time.time())
    col_order.append('Runaway Greenhouse Effects?')
    
    ## IN HABITABLE ZONE?
    hz = [(a[i] > 0.2*1.496e11) & (a[i] < 0.38*1.496e11) for i in range(N)]
    systems['In Habitable Zone?'] = hz
    t.append(time.time())
    col_order.append('In Habitable Zone?')
    
    ## IS LIFE POSSIBLE?
    systems['Is Life Possible?'] = [(~rg[i]) & (hz[i]) for i in range(N)]
    t.append(time.time())
    col_order.append('Is Life Possible?')
    
    setattr(systems, 'IF', IF)
    setattr(systems, 'decay', decay)
    setattr(systems, 'CF', CF)
    setattr(systems, 'N', N)
    setattr(systems, 'A_bounds', A_bounds)
    setattr(systems, 'i_surface', i_surface)
    setattr(systems, 'logspace_ages', logspace_ages)

    if timeit:
        per_N = kwargs.pop('per_N', False)
        t = np.array(t)
        t = t[1:] - t[:-1]
        print(f"Total time: {np.sum(t)} seconds")
        if per_N: t = t/N
        xticklabels = col_order[1:]
        fig, ax = plt.subplots(figsize=(len(t)/3,3))
        ax.bar(np.arange(len(t)), t)
        ax.set_ylabel("Time [s]" + per_N*" (per System)")
        ax.set_xticks(np.arange(len(t)))
        ax.set_xticklabels(xticklabels, rotation=90)
        ax.grid()
    
    return systems



def save(df, filename=None, n=None, parent_dir=sims_dir, **kwargs):
    
    ext = kwargs.pop('ext', ".csv")
    
    IF = df.IF
    decay = df.decay
    CF = df.CF
    N = df.N
    
    folder = foldername(IF, decay, CF, N, parent_dir=parent_dir)
    if not os.path.isdir(folder): 
        os.mkdir(folder)
    
    if n is None and filename is None:
        current_ns = [int(filename[-8:-len(ext)]) for filename in os.listdir(folder) if filename[:3]=='sim']
        n = 1 if len(current_ns)==0 else max(current_ns)+1
    
    if filename is None: 
        filename = f'sim{n:04d}' + ext
    elif n is not None: 
        filename = f"{filename}{n:04d}" + ext
    else:
        filename = filename + ext
    
    path = os.path.join(folder, filename)
    df.to_pickle(path)
    
    return


    
def load(n, IF, decay, CF, N, filename=None, parent_dir=sims_dir, **kwargs):
    
    ext = kwargs.pop('ext', ".csv")
    
    folder = foldername(IF, decay, CF, N, parent_dir=parent_dir)
    
    if filename is None: 
        filename = f'sim{n:04d}' + ext
    elif n is not None: 
        filename = f"{filename}{n:04d}" + ext
    else:
        filename = filename + ext
    
    path = os.path.join(folder, filename)
    df = pd.read_pickle(path)
    
    setattr(df, 'IF', IF)
    setattr(df, 'decay', decay)
    setattr(df, 'CF', CF)
    setattr(df, 'N', N)
    
    return df



def collectData(n_sims, IFs, decays, CFs=0.5, Ns=2500, **kwargs):
    
    foldername = kwargs.pop('foldername', "Simulation Collection")
    parent = kwargs.pop('parent', parent_dir)
    i_surface = kwargs.pop('i_surface', [None]*n_sims)
    
    foldername = f"{date.today()} " + foldername
    folder = os.path.join(parent, foldername)
    if not os.path.isdir(folder): os.mkdir(folder)
    
    if isinstance(IFs, float): IFs = [IFs]
    if isinstance(decays, str): decays = [decays]
    if isinstance(CFs, float): CFs = [CFs]
    if isinstance(Ns, int): Ns = [Ns]
    
    for IF in IFs:
        for decay in decays:
            for CF in CFs:
                for N in Ns:
                    for n in range(n_sims):
                        df = simulate(IF, decay, CF=CF, N=N, i_surface=i_surface[n], **kwargs)
                        save(df, parent_dir=folder)
                
    return
            


###############################################################################
#########################-----HELPER FUNCTIONS-----############################
###############################################################################



def hstackColumn(systems, col, conditions=[], condition_states=None):
    
    """
    Returns elements of flattened column of systems that meet an optional set of conditions. 
    - systems: dataframe of planetary systems
    - col: column to flatten
    - conditions: list of conditions
    - condition_states: list of required states (true/false) of conditions
    """
    
    stack = np.hstack(systems[col])
    N, NS = len(systems), len(stack)
    keep = np.array([True]*NS)
    if isinstance(conditions, str): conditions = [conditions]
    for i, condition in enumerate(conditions):
        
        if isinstance(condition, str):
            if condition in systems.columns:
                condition = np.hstack(systems[condition])
        else:
            assert (len(condition)==N or len(condition)==NS), f"Manual condition must be length {N} or {NS}, but is length {len(condition)}."
        
        if len(condition)==N and N!=NS:
            NP = systems['Number of Planets']
            condition = np.repeat(condition, NP)
        
        if condition_states is not None:
            if not condition_states[i]: condition = ~condition
        
        keep = (keep & condition)
        
    return stack[keep]



def foldername(IF, decay, CF, N, parent_dir=sims_dir):
    return os.path.join(parent_dir, f"IF{int(IF*100)}-DR{decay}-CF{int(CF*100)}-N{N}")



def loadMany(IF, decay, CF, N, ns=None, parent_dir=main_sims_dir):
    
    folder = foldername(IF, decay, CF, N, parent_dir=parent_dir)
    if ns is None:
        subs = os.listdir(folder)
        ns = np.arange(len(subs)) + 1
    
    dfs = np.empty(len(ns), dtype=object)
    progress = 0
    for i, n in enumerate(ns):
        dfs[i] = load(n, IF, decay, CF, N, parent_dir=parent_dir)
        
        if (i+1)/len(ns) >= progress+(1/12):
            progress = (i+1)/len(ns)
            print(f"Loading ({IF}, {decay}, {CF}, {N}): {progress*100:.2f}%...")
    
    return dfs



def LoI_legend(ax, IFs, decays, **kwargs):
    
    bbox_to_anchor = kwargs.pop('bbox_to_anchor', (1,0.5))
    locs = kwargs.pop('locs', ('lower left', 'upper left'))
    
    red = mpatches.Patch(color='red', label='0.1')
    purple = mpatches.Patch(color='purple', label='0.2')
    blue = mpatches.Patch(color='blue', label='0.4')
    slow = mlines.Line2D([], [], color='black', linestyle='-',
                          markersize=15, label='Slow')
    medium = mlines.Line2D([], [], color='black', linestyle='--',
                          markersize=15, label='Medium')
    fast = mlines.Line2D([], [], color='black', linestyle=':',
                          markersize=15, label='Fast')
    
    handles_IF_dict = {0.1:red, 0.2:purple, 0.4:blue}
    handles_decay_dict = {'slow':slow, 'medium':medium, 'fast':fast}
    
    handles_IF = [handles_IF_dict[IF] for IF in IFs]
    handles_decay =[handles_decay_dict[decay] for decay in decays]
    
    legend_IF = ax.legend(handles=handles_IF, loc=locs[0], bbox_to_anchor=bbox_to_anchor)
    legend_IF.set_title("Total Intact Fraction")
    ax.add_artist(legend_IF)
    legend_decay = ax.legend(handles=handles_decay, loc=locs[1], bbox_to_anchor=bbox_to_anchor)
    legend_decay.set_title("Decay Rate")
    
    return



def P_tau_in_interval_disrupted(IF, decay, CF, tau_lower, tau_upper, **kwargs):
    
    L = lambda A : LoI(IF, decay, A)
    
    a = quad(lambda A: 1-L(A-tau_lower), tau_lower, tau_upper)[0]
    
    b = quad(lambda A: L(A-tau_lower) - L(A), tau_lower, tau_upper)[0]
    
    c = quad(lambda A: L(A-tau_upper) - L(A-tau_lower), tau_upper, 1e10)[0]
    
    d = quad(lambda A: 1-L(A), 1e6, 1e10)[0]
    
    return (a + (1-CF)*b + CF*c) / d



def PDF_tau_disrupted(IF, decay, CF, n_bins=100, **kwargs):
    
    taus = np.linspace(0, 1e10, n_bins+1)
    PDF = np.zeros(n_bins)
    
    for i in range(n_bins):
        PDF[i] = P_tau_in_interval_disrupted(IF, decay, CF, taus[i], taus[i+1])
    
    norm = np.sum(PDF*(1e10/n_bins))

    return PDF/norm



###############################################################################
#########################-----PLOTTING FUNCTIONS-----##########################
###############################################################################



def timeSimulate(IF=0.1, decay='slow', log10N_range=(1,4,100), fit=True, **kwargs):
    
    Ns = np.int32(np.logspace(*log10N_range))
    times = np.zeros(len(Ns))
    for i, N in enumerate(Ns):
        start = time.time()
        simulate(IF, decay, N=N)
        times[i] = time.time()-start
    
    fig, ax = plt.subplots()
    ax.loglog(Ns, times, '+')
    ax.set_xlabel(r"Number of Systems $N$")
    ax.set_ylabel(r"Simulation Time")
    ax.grid()
    
    if fit:
        def func(N, floor, A, n):
            return floor + A*N**n
        
        popt, pcov = curve_fit(func, Ns, times)
        ax.loglog(Ns, func(Ns, *popt), 'C0', alpha=0.4)
        return N, times, lambda N: func(N, *popt)
        
    
    return N, times



def LoI_Overlay(IFs=[0.1,0.4], decays=['fast','medium','slow'], show_all=False, ax=None, **kwargs):
    
    assert show_all or (len(IFs)!=0 and len(decays)!=0), 'Must supply intact fraction and decay or toggle show_all.'
    
    xlim = kwargs.pop('xlim', (1e6,1e10))
    ylim = kwargs.pop('ylim', (0,1))
    A = kwargs.pop('A', np.logspace(*np.log10(xlim), 100))
    save = kwargs.pop('save', False)
    parent_dir = kwargs.pop('parent_dir', figures_dir)
    filename = kwargs.pop('filename', "Likelihood of Intactness Functions")
    ext = kwargs.pop('ext', ".pdf")
    dpi = kwargs.pop('dpi', 200)
    if show_all: IFs, decays = [0.1, 0.2, 0.4], ['fast','medium','slow']
    
    if ax is None: fig, ax = plt.subplots(dpi=200)
    for IF in IFs:
        for decay in decays:
            ax.plot(A, LoI(IF, decay, A), markers_dict[decay], color=colors_dict[IF], **kwargs)
    
    LoI_legend(ax, IFs, decays, bbox_to_anchor=(0.74,1), locs=('upper right', 'upper left'))
    ax.set(xscale='log', ylabel='Likelihood of Intactness', xlabel='Age of Star [yr]', ylim=ylim, xlim=xlim)
    
    if save:
        plt.savefig(os.path.join(parent_dir, filename + ext), dpi=dpi, transparent=True)
    
    return



def planetPropertiesHistogram(sim, ax=None, **kwargs):
    
    save = kwargs.pop('save', False)
    parent_dir = kwargs.pop('parent_dir', figures_dir)
    filename = kwargs.pop('filename', "Planet Properties Histograms")
    ext = kwargs.pop('ext', ".pdf")
    dpi = kwargs.pop('dpi', 200)
    
    ecc_bins = np.linspace(0,1,100)
    inc_bins = np.linspace(-20,20,100)
    r_bins = np.linspace(0,5,10)
    m_bins = np.linspace(0,20,10)
    
    prop_bins = [ecc_bins, ecc_bins, inc_bins, inc_bins, r_bins, m_bins]
    prop_range = [(0,1),(0,1),(-20,20),(-20,20),(0,5),(0,20)]
    i_ax = [0, 0, 1, 1, 2, 3]
    colors = [colors_master['Intact'], colors_master['Disrupted'], colors_master['Intact'], colors_master['Disrupted'], 'k', 'k']
    labels = ['Intact','Disrupted','Intact','Disrupted', None, None]
    xlabels = ['Eccentricity', r'Inclination [$^{\circ}$]', r'Radius [$R_{\oplus}$]', r'Mass [$M_{\oplus}$]', None, None]
    
    if ax is None: fig, ax = plt.subplots(ncols=4, figsize=(18, 3), dpi=200)
        
    ecc = np.array(sim['Eccentricities'], dtype=object)
    inc = np.array(sim['Inclinations']-sim['Mean Orbital Plane Inclination'], dtype=object)
        
    II = sim['Intact']
    DI = ~sim['Intact']
        
    ecc_i = np.hstack(ecc[II])
    ecc_d = np.hstack(ecc[DI])
    inc_i = np.hstack(inc[II])
    inc_d = np.hstack(inc[DI])
    
    r = np.hstack(np.array(sim['Radii']))
    m = np.hstack(np.array(sim['Masses']))
    
    prop_categories = [ecc_i, ecc_d, inc_i, inc_d, r, m]
    
    for j in range(len(prop_categories)):
        n, bins = np.histogram(prop_categories[j], bins=len(prop_bins[j]), range=prop_range[j], density=True)
        ax[i_ax[j]].plot(prop_bins[j], n, color=colors[j], label=labels[j])
    
    ax[0].legend()
    ax[1].legend()
    ax[0].set_ylabel('Probability Density')
    for i in range(4):
        ax[i].set_xlabel(xlabels[i])
    
    if save:
        plt.savefig(os.path.join(parent_dir, filename + ext), dpi=dpi, transparent=True, bbox_inches='tight')

    return



def eccIncContour(sim, ax=None, **kwargs):
    
    save = kwargs.pop('save', False)
    parent_dir = kwargs.pop('parent_dir', figures_dir)
    filename = kwargs.pop('filename', "Inclinations Vs Eccentricities")
    ext = kwargs.pop('ext', ".pdf")
    dpi = kwargs.pop('dpi', 200)
    legend = kwargs.pop('legend', True)
    ylabel = kwargs.pop('ylabel', True)
    
    if ax is None: fig, ax = plt.subplots(1, 1, figsize=[7,5], dpi=200)
        
    inc = np.array(np.abs(sim['Inclinations'] - sim['Mean Orbital Plane Inclination']))
    ecc = np.array(sim['Eccentricities'])
    
    II = sim['Intact']
    DI = ~sim['Intact']
    
    inc_II = np.hstack(inc[II])
    ecc_II = np.hstack(ecc[II])
    inc_DI = np.hstack(inc[DI])
    ecc_DI = np.hstack(ecc[DI])
    
    kde_II = gaussian_kde(np.vstack([np.log(ecc_II), np.log(inc_II)]))
    kde_DI = gaussian_kde(np.vstack([np.log(ecc_DI), np.log(inc_DI)]))
    
    #ee_II, ii_II = np.meshgrid(ecc_II, inc_II)
    #ee_DI, ii_DI = np.meshgrid(ecc_DI, inc_DI)
    ee, ii = np.meshgrid(np.logspace(-3,0), np.logspace(-2,2))
    
    positions = np.vstack([np.log(ee).ravel(), np.log(ii).ravel()])
    density_II = np.reshape(kde_II(positions).T, ee.shape)
    density_DI = np.reshape(kde_DI(positions).T, ee.shape)
    density_II = (density_II - np.min(density_II))/(np.max(density_II) - np.min(density_II))
    density_DI = (density_DI - np.min(density_DI))/(np.max(density_DI) - np.min(density_DI))
    
    levels = 1 - np.array([0.675, 0.393, 0.118])
    ax.contour(ee, ii, density_II, levels=levels, colors=colors_master['Intact'])
    ax.contour(ee, ii, density_DI, levels=levels, colors=colors_master['Disrupted'])
    
    #ax.scatter(ecc_II, inc_II, s=0.5, color=colors_master['Intact'], alpha=0.3, label='Intact')
    #ax.scatter(ecc_DI, inc_DI, s=0.5, color=colors_master['Disrupted'], alpha=0.3, label='Disrupted')    
    ax.plot([0,1],[0,0],color=colors_master['Intact'],label='Intact')
    ax.plot([0,1],[0,0],color=colors_master['Disrupted'],label='Disrupted')
    ax.plot([0,1],[0,0],lw=0.8,color='black',label='He et al, 2020')
    ax.set_xlabel(r'$e$')
    if ylabel:
        ax.set_ylabel(r'$|i|\ [^{\circ}]$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.001,1)
    ax.set_ylim(0.01,100)
    if legend:
        ax.legend(loc='lower left')
    
    if save:
        plt.savefig(os.path.join(parent_dir, filename + ext), dpi=dpi, transparent=True)

    return



def drawSystems(sim, **kwargs):
    
    save = kwargs.pop('save', False)
    parent_dir = kwargs.pop('parent_dir', figures_dir)
    filename = kwargs.pop('filename', "Typical Systems")
    ext = kwargs.pop('ext', ".pdf")
    dpi = kwargs.pop('dpi', 200)
    
    a = np.array(sim['Semi-Major Axes']/m_per_AU)
    rel_inc = np.array(sim['Inclinations']-sim['Mean Orbital Plane Inclination'])
    b = np.array([a[i] * np.sin((np.pi/180) * rel_inc[i]) for i in range(len(a))], dtype=object)
    t = np.array(sim['Transits?'])
    nt = np.array(sim['Number of Transits'])
    n = np.array(sim['Number of Planets'])
    colors = np.append([None], [f'C{i}' for i in range(15)])
    
    II = sim['Intact']
    DI = ~sim['Intact']
    
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(12,6), dpi=200)
    ax[0,0].set_aspect('equal')
    ax[0,1].set_aspect('equal')
    ax[1,0].set_aspect('equal')
    ax[1,1].set_aspect('equal')
    [ax[0,0].plot(a[II][i], b[II][i], '.', c=colors[n[II][i]]) for i in range(sum(II))]
    [ax[0,1].plot(a[DI][i], b[DI][i], '.', c=colors[n[DI][i]]) for i in range(sum(DI))]
    ax[1,0].set_xlabel(r'$a$ [AU]')
    ax[1,0].set_ylabel(r'$a \cdot \sin(i)$ [AU]')
    ax[1,1].set_xlabel(r'$a$ [AU]')
    ax[0,0].set_ylabel(r'$a \cdot \sin(i)$ [AU]')
    ax[0,0].set_title('Intact')
    ax[0,1].set_title('Disrupted')
    
    for i in range(sum(II)):
        colors = np.where(t[II][i], (colors_master['Single'] if nt[II][i]==1 else colors_master['Multi']), 'grey')
        alphas = np.where(t[II][i], 0.8, 0)
        ax[1,0].scatter(a[II][i], b[II][i], marker='.', color=colors, alpha=alphas)
        
    for i in range(sum(DI)):
        colors = np.where(t[DI][i], (colors_master['Single'] if nt[DI][i]==1 else colors_master['Multi']), 'grey')
        alphas = np.where(t[DI][i], 0.8, 0)
        ax[1,1].scatter(a[DI][i], b[DI][i], marker='.', color=colors, alpha=alphas)
    
    fig.subplots_adjust(wspace=-0.3)
    
    norm = mpl.colors.Normalize(vmin=0.5, vmax=10.5)
    sm = plt.cm.ScalarMappable(cmap='tab10', norm=norm)
    cax = fig.add_axes([ax[0,1].get_position().x1, ax[0,1].get_position().y0, 0.01, ax[0,1].get_position().height])
    cbar = plt.colorbar(sm, ax=ax[0,1], cax=cax)
    cbar.set_ticks(np.arange(1,11,2))
    cbar.set_label('Number of Planets')
    
    labels = ['Single', 'Multi']
    colors = [colors_master['Single'], colors_master['Multi']]
    handles = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
    legend = fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.805, 0.48))
    legend.set_title("Transits in a ...")
    
    [a.grid() for a in ax.ravel()]
    
    if save:
        plt.savefig(os.path.join(parent_dir, filename + ext), dpi=dpi, transparent=True)
    
    return



def ntBarGraph(sims_k, sims_t, ax=None, **kwargs):
    
    save = kwargs.pop('save', False)
    parent_dir = kwargs.pop('parent_dir', figures_dir)
    filename = kwargs.pop('filename', "Number of Transits")
    ext = kwargs.pop('ext', ".pdf")
    dpi = kwargs.pop('dpi', 200)
    legend = kwargs.pop('legend', True)
    ylabel = kwargs.pop('ylabel', True)
    
    nt_range = np.arange(1,6)
    
    sim_k_nt_counts = np.empty((len(sims_k), 5))
    
    for i, sim in enumerate(sims_k):
        nt = sim['Number of Transits']
        sim_k_nt_counts[i] = np.array([len(np.where(nt == nt_range[i])[0]) for i in range(len(nt_range))])
        sim_k_nt_counts[i] = np.array(sim_k_nt_counts[i])/np.sum(sim_k_nt_counts[i])
        
    sim_t_nt_counts = np.empty((len(sims_t), 5))
    
    for i, sim in enumerate(sims_t):
        nt = sim['Number of Tess Detections']
        sim_t_nt_counts[i] = np.array([len(np.where(nt == nt_range[i])[0]) for i in range(len(nt_range))])
        sim_t_nt_counts[i] = np.array(sim_t_nt_counts[i])/np.sum(sim_t_nt_counts[i])
    
    if ax is None: fig, ax = plt.subplots(1, 1, figsize=[5,3.5], dpi=200)
    width = 0.25
    ax.bar(nt_range-width, np.array(kepler_nt_counts)/np.sum(kepler_nt_counts), width, color='deeppink', label='Actual Kepler')
    median = np.median(sim_k_nt_counts, axis=0)
    yerr = [median-np.percentile(sim_k_nt_counts, 16, axis=0), np.percentile(sim_k_nt_counts, 82, axis=0)-median]
    ax.bar(nt_range, median, width, yerr=yerr, color='hotpink', alpha=0.5, label='Simulated Kepler')
    median = np.median(sim_t_nt_counts, axis=0)
    yerr = [median-np.percentile(sim_t_nt_counts, 16, axis=0), np.percentile(sim_t_nt_counts, 82, axis=0)-median]
    ax.bar(nt_range+width, median, width, yerr=yerr, color='purple', label='Simulated TESS')
    ax.set_xticks(nt_range)
    ax.set_xlim(0.5,5.5)
    if ylabel:
        ax.set_ylabel('Fraction of Systems with Detected Planets')
    ax.set_xlabel('Number of Transiting Planets per System')
    if legend:
        ax.legend()
    
    if save:
        plt.savefig(os.path.join(parent_dir, filename + ext), dpi=dpi, transparent=True)
    
    return



def smdiComparison(sims, observer='kepler', ax=None, **kwargs):
    
    save = kwargs.pop('save', False)
    parent_dir = kwargs.pop('parent_dir', figures_dir)
    filename = kwargs.pop('filename', "Fraction of Observed Systems By Disruption State")
    ext = kwargs.pop('ext', ".pdf")
    dpi = kwargs.pop('dpi', 200)
    legend = kwargs.pop('legend', True)
    title = kwargs.pop('title', True)
    ylabel = kwargs.pop('ylabel', True)
    
    intact_singles = np.empty(len(sims))
    intact_multis = np.empty(len(sims))
    
    for i, sim in enumerate(sims):
        
        if observer == 'inherent':
            nt = sim['Number of Transits']
        if observer == 'kepler':
            nt = sim['Number of Transits']
        if observer == 'tess':
            nt = sim['Number of Tess Detections']
        
        II = sim['Intact']
        
        intact_singles[i] = (sum(nt[II] == 1)/sum(nt == 1))
        intact_multis[i] = (sum(nt[II] > 1)/sum(nt > 1))
    
    mean_is = np.mean(intact_singles)
    mean_im = np.mean(intact_multis)
    std_is = np.std(intact_singles)
    std_im = np.std(intact_multis)
    
    if ax is None: fig, ax = plt.subplots(1, 1, figsize=[3,5], dpi=200)
    
    ax.bar(['Singles', 'Multis'], [mean_is, mean_im], 0.7, yerr=[std_is, std_im], color=colors_master['Intact'], label='Intact')
    ax.bar(['Singles', 'Multis'], [1-mean_is,1-mean_im], 0.7, bottom=[mean_is, mean_im], color=colors_master['Disrupted'], label='Disrupted')
    if ylabel:
        ax.set_ylabel('Fraction of Systems')
    if title:
        ax.set_title(r'$\bar{\mathscr{L}}$ = ' + f'{sims[0].IF}', fontsize=10)
    if legend: ax.legend(loc='lower right')
    
    if save:
        plt.savefig(os.path.join(parent_dir, filename + ext), dpi=dpi, transparent=True, bbox_inches='tight')
    
    return



def eccIncHistogram(sims, observer='kepler', ecc_n=None, inc_n=None, ax=None, **kwargs):
    
    save = kwargs.pop('save', False)
    parent_dir = kwargs.pop('parent_dir', figures_dir)
    filename = kwargs.pop('filename', "Eccentricity and Inclination Histograms")
    ext = kwargs.pop('ext', ".pdf")
    dpi = kwargs.pop('dpi', 200)
    ylabel = kwargs.pop('ylabel', True)
    legend = kwargs.pop('legend', True)
    title = kwargs.pop('title', True)
    
    ecc_bins = np.linspace(0,1,100)
    inc_bins = np.linspace(-20,20,100)
    
    if ecc_n is None or inc_n is None:
    
        ecc_n = np.empty((len(sims),2,100))
        inc_n = np.empty((len(sims),2,100))
            
        for i, sim in enumerate(sims):
            
            ecc = np.array(sim['Eccentricities'], dtype=object)
            inc = np.array(sim['Inclinations']-sim['Mean Orbital Plane Inclination'], dtype=object)
            t = np.array(sim['Transits?'], dtype=object)
            
            if observer == 'inherent':
                
                title_name = 'Inherent'
                
                II = sim['Intact']
                DI = ~sim['Intact']
                
                ecc_i_or_m = np.hstack(ecc[II])
                ecc_d_or_s = np.hstack(ecc[DI])
                inc_i_or_m = np.hstack(inc[II])
                inc_d_or_s = np.hstack(inc[DI])
                
            if observer == 'kepler':
                
                title_name = 'Observed by Kepler'
            
                MI = sim['Number of Transits'] > 1
                SI = sim['Number of Transits'] == 1
                
            if observer == 'tess':
                
                title_name = 'Observed by TESS'
                
                MI = sim['Number of Tess Detections'] > 1
                SI = sim['Number of Tess Detections'] == 1
                
            if observer == 'kepler' or observer == 'tess':
            
                ecc_i_or_m = []
                inc_i_or_m = []
                for k in range(sum(MI)):
                    system_ecc = ecc[MI][k]
                    system_inc = inc[MI][k]
                    system_t = t[MI][k]
                    ecc_i_or_m = np.append(ecc_i_or_m, system_ecc[system_t])
                    inc_i_or_m = np.append(inc_i_or_m, system_inc[system_t])
                
                ecc_d_or_s = []
                inc_d_or_s = []
                for k in range(sum(SI)):
                    system_ecc = ecc[SI][k]
                    system_inc = inc[SI][k]
                    system_t = t[SI][k]
                    ecc_d_or_s = np.append(ecc_d_or_s, system_ecc[system_t])
                    inc_d_or_s = np.append(inc_d_or_s, system_inc[system_t])
            
            ecc_categories = [ecc_i_or_m, ecc_d_or_s]
            inc_categories = [inc_i_or_m, inc_d_or_s]
            #ecc_categories = [ecc_i_or_m, ecc_d_or_s, np.append(ecc_i_or_m,ecc_d_or_s)]
            #inc_categories = [inc_i_or_m, inc_d_or_s, np.append(inc_i_or_m,inc_d_or_s)]
            
            for j in range(len(ecc_categories)):
                n, bins = np.histogram(ecc_categories[j], bins=100, range=(0,1), density=True)
                ecc_n[i,j] = n
                
            for j in range(len(inc_categories)):
                n, bins = np.histogram(inc_categories[j], bins=100, range=(-20,20), density=True)
                inc_n[i,j] = n
            
    if ax is None: fig, ax = plt.subplots(nrows=2, figsize=(3, 4), dpi=200)
    
    if observer == 'inherent': labels = ['Intact','Disrupted']
    if observer == 'kepler' or observer == 'tess': labels = ['Multi','Single']
    
    for i in range(2):
        ax[0].plot(ecc_bins, np.median(ecc_n[:,i,:], axis=0), color=colors_master[labels[i]], label=labels[i])
        ax[0].fill_between(ecc_bins, y1=np.percentile(ecc_n[:,i,:], 16, axis=0), y2=np.percentile(ecc_n[:,i,:], 82, axis=0), color=colors_master[labels[i]], alpha=0.2)
        ax[1].plot(inc_bins, np.median(inc_n[:,i,:], axis=0), color=colors_master[labels[i]], label=labels[i])
        ax[1].fill_between(inc_bins, y1=np.percentile(inc_n[:,i,:], 16, axis=0), y2=np.percentile(inc_n[:,i,:], 82, axis=0), color=colors_master[labels[i]], alpha=0.2)
    
    if ylabel:
        ax[0].set_ylabel('Probability Density')
        ax[1].set_ylabel('Probability Density')
    if title:
        ax[0].set_title(title_name)
    if legend:
        ax[0].legend()
    
    if save:
        plt.savefig(os.path.join(parent_dir, filename + ext), dpi=dpi, transparent=True)

    return ecc_n



def tauHistogram(sims, observer='kepler', n_bins=100, ax=None, **kwargs):
    
    data = kwargs.pop('data', None)
    save = kwargs.pop('save', False)
    parent_dir = kwargs.pop('parent_dir', figures_dir)
    filename = kwargs.pop('filename', "Tau Histogram")
    ext = kwargs.pop('ext', ".pdf")
    dpi = kwargs.pop('dpi', 200)
    legend = kwargs.pop('legend', True)
    title = kwargs.pop('title', True)
    xlabel = kwargs.pop('xlabel', True)
    ylabel = kwargs.pop('ylabel', True)
    
    et_bins = np.linspace(0,1e10,n_bins)
    title_name = {'inherent':'Inherent', 'kepler':'Observed by Kepler', 'tess':'Observed by TESS'}[observer]
    
    if data is None:
    
        et_n = np.empty((len(sims),2,n_bins))
        cdf = np.empty((len(sims),2,n_bins))
        
        for i, sim in enumerate(sims):
            
            if observer == 'inherent':

                et_i_or_m = hstackColumn(sim, 'Evolutionary Timescales', conditions=['Intact'])
                et_d_or_s = hstackColumn(sim, 'Evolutionary Timescales', conditions=['Intact'], condition_states=[False])
            
            if observer == 'kepler':  
                
                et_i_or_m = hstackColumn(sim, 'Evolutionary Timescales', conditions=['Transits?', sim['Number of Transits']>1])
                et_d_or_s = hstackColumn(sim, 'Evolutionary Timescales', conditions=['Transits?', sim['Number of Transits']==1])
                
            if observer == 'tess':
                
                et_i_or_m = hstackColumn(sim, 'Evolutionary Timescales', conditions=['Detected By Tess?', sim['Number of Tess Detections']>1])
                et_d_or_s = hstackColumn(sim, 'Evolutionary Timescales', conditions=['Detected By Tess?', sim['Number of Tess Detections']==1])
            
            et_categories = [et_i_or_m, et_d_or_s]
            
            for k in range(len(et_categories)):
                hist = cumfreq(et_categories[k], numbins=n_bins, defaultreallimits=(0,1e10))
                et_n[i,k] = hist.cumcount
                cdf[i,k] = et_n[i,k]/et_n[i,k,-1]
                
    else: 
        cdf = data
    
    mean = np.nanmean(cdf, axis=0)
    std = np.nanstd(cdf, axis=0)
        
    if ax is None: fig, ax = plt.subplots(dpi=200)
    
    if observer == 'inherent': labels = ['Intact','Disrupted']
    if observer == 'kepler' or observer == 'tess': 
        labels = ['Multi','Single']
        ax.fill_between(et_bins/1e9, y1=mean[0]-std[0], y2=mean[0]+std[0], color=colors_master[labels[0]], alpha=0.1)
        ax.fill_between(et_bins/1e9, y1=mean[1]-std[1], y2=mean[1]+std[1], color=colors_master[labels[1]], alpha=0.1)
    ax.plot(et_bins/1e9, mean[0], color=colors_master[labels[0]], label=labels[0])
    ax.plot(et_bins/1e9, mean[1], color=colors_master[labels[1]], label=labels[1])
    if xlabel:
        ax.set_xlabel(r'$t$ [Gyr]')
    if ylabel:
        ax.set_ylabel(r'$P(\tau < t)$')
    if title:
        ax.set_title(title_name, fontsize=13)
    if legend:
        ax.legend()
    
    if save:
        plt.savefig(os.path.join(parent_dir, filename + ext), dpi=dpi, transparent=True)
    
    return cdf



def deltaTauMinAge(sims, observer='kepler', n_bins=100, log_bins=True, ax=None, **kwargs):
    
    data = kwargs.pop('data', None)
    save = kwargs.pop('save', False)
    parent_dir = kwargs.pop('parent_dir', figures_dir)
    filename = kwargs.pop('filename', "Delta Tau Vs Min Age")
    ext = kwargs.pop('ext', ".pdf")
    dpi = kwargs.pop('dpi', 200)
    title = kwargs.pop('title', True)
    xlabel = kwargs.pop('xlabel', True)
    ylabel = kwargs.pop('ylabel', True)
    color = kwargs.pop('color', None)
    marker = kwargs.pop('marker', None)
    error = kwargs.pop('error', False)
    
    title_name = {'inherent':'Inherent', 'kepler':'Observed by Kepler', 'tess':'Observed by TESS'}[observer]
    
    if data is None:
        
        min_age = np.logspace(6,10,n_bins) if log_bins else np.linspace(1e6,1e10,n_bins)
            
        et_i_or_m = np.empty((len(sims),n_bins))
        et_d_or_s = np.empty((len(sims),n_bins))
        
        if observer == 'inherent':
    
            for i, sim in enumerate(sims):
                
                for j in range(len(min_age)):
                    if j%10==0: print(f"Simulation {i+1}/{len(sims)}, age {j+1}/{len(min_age)}")
                    age_condition = sim['Age']>min_age[j]
                    
                    ##Below is the mean over ETs within one simulation given a minimum stellar age.
                    et_i_or_m[i,j] = np.nanmean(hstackColumn(sim, 'Evolutionary Timescales', conditions=['Intact', age_condition]))
                    et_d_or_s[i,j] = np.nanmean(hstackColumn(sim, 'Evolutionary Timescales', conditions=['Intact', age_condition], condition_states=[False, True]))
                    
        if observer == 'kepler':
            
            for i, sim in enumerate(sims):
            
                for j in range(len(min_age)):
                    if j%10==0: print(f"Simulation {i+1}/{len(sims)}, age {j+1}/{len(min_age)}")
                    age_condition = sim['Age']>min_age[j]
                
                    et_i_or_m[i,j] = np.nanmean(hstackColumn(sim, 'Evolutionary Timescales', conditions=['Transits?', sim['Number of Transits']>1, age_condition]))
                    et_d_or_s[i,j] = np.nanmean(hstackColumn(sim, 'Evolutionary Timescales', conditions=['Transits?', sim['Number of Transits']==1, age_condition]))
                    
        if observer == 'tess':
            
            for i, sim in enumerate(sims):
            
                for j in range(len(min_age)):
                    if j%10==0: print(f"Simulation {i+1}/{len(sims)}, age {j+1}/{len(min_age)}")
                    age_condition = sim['Age']>min_age[j]
                
                    et_i_or_m[i,j] = np.nanmean(hstackColumn(sim, 'Evolutionary Timescales', conditions=['Detected By Tess?', sim['Number of Tess Detections']>1, age_condition]))
                    et_d_or_s[i,j] = np.nanmean(hstackColumn(sim, 'Evolutionary Timescales', conditions=['Detected By Tess?', sim['Number of Tess Detections']==1, age_condition]))
        
    else:
        min_age, et_i_or_m, et_d_or_s = data
    
    ##Below is the mean of the curve of ETs versus minimum stellar age over multiple simulations.
    mean_et_i_or_m = np.nanmean(et_i_or_m, axis=0)
    mean_et_d_or_s = np.nanmean(et_d_or_s, axis=0)
    mean = mean_et_i_or_m - mean_et_d_or_s
    
    if ax is None: fig, ax = plt.subplots(dpi=200)
    
    if marker is None:
        marker = '-'
    if color is None:
        color = 'black'
    ax.plot(min_age, mean/1e9, marker, color=color)
    ax.set_xscale('log')
    if xlabel:
        ax.set_xlabel(r'Minimum Stellar Age [yr]')
    if error:
        if observer == 'kepler' or observer == 'tess': 
            std_et_i_or_m = np.nanstd(et_i_or_m, axis=0)
            std_et_d_or_s = np.nanstd(et_d_or_s, axis=0)
            std = np.sqrt(std_et_i_or_m**2 + std_et_d_or_s**2)
            ax.fill_between(min_age, y1=(mean-std)/1e9, y2=(mean+std)/1e9, color=color, alpha=0.1)
    if ylabel:
        if observer == 'inherent':
            ax.set_ylabel(r'$\bar{\tau}_{intact}$ - $\bar{\tau}_{disrupted}$ [Gyr]')
        if observer == 'kepler' or observer == 'tess':
            ax.set_ylabel(r'$\bar{\tau}_{multi}$ - $\bar{\tau}_{single}$ [Gyr]')
    if title:
        ax.set_title(title_name, fontsize=13)
        
    if save:
        plt.savefig(os.path.join(parent_dir, filename + ext), dpi=dpi, transparent=True)
    
    return min_age, et_i_or_m, et_d_or_s



def amdAgeScatter(sim, ax=None, **kwargs):
    
    save = kwargs.pop('save', False)
    parent_dir = kwargs.pop('parent_dir', figures_dir)
    filename = kwargs.pop('filename', "Age Vs AMD")
    ext = kwargs.pop('ext', ".pdf")
    dpi = kwargs.pop('dpi', 200)
    legend = kwargs.pop('legend', True)
    ylabel = kwargs.pop('ylabel', True)
    
    if ax is None: fig, ax = plt.subplots(1, 1, figsize=[7,5], dpi=200)
        
    age = np.array(sim['Age'])
    a = np.array(sim['Semi-Major Axes'])
    ecc = np.array(sim['Eccentricities'])
    inc = np.array(sim['Inclinations'] - sim['Mean Orbital Plane Inclination'])
    m = np.array(sim['Masses'])
    
    Lambda = np.empty(len(sim), dtype=object)
    L = np.empty(len(sim), dtype=object)
    L_deficit = np.empty(len(sim), dtype=object)
    mean_L_deficit = np.zeros(len(sim))
    for j in range(len(sim)):
        Lambda[j] = m[j]*M_earth * np.sqrt(G*M_star*a[j])
        L[j] = Lambda[j] * np.sqrt(1 - ecc[j]**2) * np.cos(inc[j]*np.pi/180)
        L_deficit[j] = Lambda[j] - L[j]
        mean_L_deficit[j] = np.mean(L_deficit[j])
    
    II = sim['Intact']
    DI = ~sim['Intact']
    
    age_II = age[II]
    age_DI = age[DI]
    mean_L_deficit_II = mean_L_deficit[II]
    mean_L_deficit_DI = mean_L_deficit[DI]
    
    bins = np.logspace(6,10,100)
    bin_centers = 10**((np.log10(bins[1:]) + np.log10(bins[:-1]))/2)
    j_bins = np.digitize(age, bins)
    mean_L_deficit_means = np.zeros(len(bins)-1)
    for j in range(len(bins)-1):
        if sum(j_bins==j)>0:
            mean_L_deficit_in_this_bin = np.hstack(mean_L_deficit[j_bins==j])
            mean_L_deficit_means[j] = np.mean(np.log10(mean_L_deficit_in_this_bin))
        else:
            mean_L_deficit_means[j] = np.nan
    
    ax.loglog(age_DI, mean_L_deficit_DI, '.', color=colors_master['Disrupted'], alpha=0.6, label='Disrupted')
    ax.loglog(age_II, mean_L_deficit_II, '.', color=colors_master['Intact'], alpha=0.6, label='Intact')
    ax.plot(bin_centers, 10**mean_L_deficit_means, color='k', lw=4, label='Overall Mean')
    if ylabel:
        ax.set_ylabel(r'Mean Planet AMD [J$\cdot$s]')
    ax.set_xlabel('Age of System [yr]')
    if legend:
        ax.legend()
    ax.grid()
        
    if save:
        plt.savefig(os.path.join(parent_dir, filename + ext), dpi=dpi, transparent=True)
        
    return



def areasExplanation(IF, decay, tau_lower, tau_upper, n_bins=500, ax=None, **kwargs):
    
    save = kwargs.pop('save', False)
    parent_dir = kwargs.pop('parent_dir', figures_dir)
    filename = kwargs.pop('filename', "Areas Explanation")
    ext = kwargs.pop('ext', ".pdf")
    dpi = kwargs.pop('dpi', 200)
    legend = kwargs.pop('legend', True)
    ylabel = kwargs.pop('ylabel', True)
    
    if ax is None: fig, ax = plt.subplots(dpi=200)
    ages = np.linspace(0,1e10,n_bins)
    ax.plot(ages/1e9, LoI(IF, decay, ages), 'r', label=r"$\mathscr{L}$")
    ax.plot((ages+tau_lower)/1e9, LoI(IF, decay, ages), 'r--')
    ax.plot((ages+tau_upper)/1e9, LoI(IF, decay, ages), 'r--')
    ax.vlines([tau_lower/1e9, tau_upper/1e9], 0, 1, colors='grey', label=r"$(\tau_l,\tau_u)$")
    
    tau_ = np.linspace(0, tau_upper-tau_lower, n_bins)
    tau__ = np.linspace(tau_upper, 1e10, n_bins)
    ax.fill_between((tau_lower + tau_)/1e9, np.ones(len(tau_)), LoI(IF,decay,tau_), color='green', alpha=0.5, label=r"$\mathscr{A}_{cm}$")
    ax.fill_between((tau_lower + tau_)/1e9, LoI(IF,decay,tau_), LoI(IF,decay,tau_lower+tau_), color='yellow', alpha=0.5, label=r"$\mathscr{A}_{m}$")
    ax.fill_between((tau__)/1e9, LoI(IF,decay,tau__-tau_upper), LoI(IF,decay,tau__-tau_lower), color='blue', alpha=0.5, label=r"$\mathscr{A}_{c}$")
    
    ax.set(xlim=(0,10), ylim=(0,1), xlabel=r"$A$ [Gyr]")
    ax.grid()
    if legend:
        ax.legend(loc='upper left')
    if ylabel:
        ax.set_ylabel(r"$r$")
    
    if save:
        plt.savefig(os.path.join(parent_dir, filename + ext), dpi=dpi, transparent=True)
    
    return



def areasExplanation2(IF, decay, CF, tau_lower, tau_upper, n_bins=500, ax=None, **kwargs):
    
    save = kwargs.pop('save', False)
    parent_dir = kwargs.pop('parent_dir', figures_dir)
    filename = kwargs.pop('filename', "Areas Explanation 2")
    ext = kwargs.pop('ext', ".pdf")
    dpi = kwargs.pop('dpi', 200)
    legend = kwargs.pop('legend', True)
    ylabel = kwargs.pop('ylabel', True)
    
    if ax is None: fig, ax = plt.subplots(dpi=200)
    ages = np.linspace(tau_lower, 1e10, n_bins)
    dA = (1e10-tau_lower)/n_bins
    L = lambda A : LoI(IF, decay, A)
    a = np.zeros((n_bins))
    b = np.zeros((n_bins))
    c = np.zeros((n_bins))
    for i in range(len(ages)):
        a[i] = quad(lambda A: 1-L(A-tau_lower), ages[i]-dA/2, ages[i]+dA/2)[0]
        if ages[i]>tau_upper:
            a[i] = 0
        b[i] = (1-CF)*quad(lambda A: L(A-tau_lower) - L(A), ages[i]-dA/2, ages[i]+dA/2)[0]
        if ages[i]>tau_upper:
            b[i] = 0
        c[i] = CF*quad(lambda A: L(A-tau_upper) - L(A-tau_lower), ages[i]-dA/2, ages[i]+dA/2)[0]
        if ages[i]<tau_upper:
            c[i] = 0
    
    d = a+b+c
    norm = simps(d, ages)/1e9
    ax.fill_between(ages/1e9, 0, a/norm, color='green', alpha=0.5, label=r"$\mathscr{A}_{cm}$")
    ax.fill_between(ages/1e9, a/norm, (a+b)/norm, color='yellow', alpha=0.5, label=r"$\mathscr{A}_{m}$")
    ax.fill_between(ages/1e9, (a+b)/norm, (a+b+c)/norm, color='blue', alpha=0.5, label=r"$\mathscr{A}_{c}$")
    ax.plot(np.concatenate(([tau_lower/1e9-0.001], ages/1e9)), np.concatenate(([0], (a+b+c)/norm)), color='black')
    ax.vlines([tau_lower/1e9, tau_upper/1e9], 0, 1.25, colors='grey', label=r"$(\tau_l,\tau_u)$")
    
    ax.set(xlim=(0,10), ylim=(0,1.25), xlabel=r"$A$ [Gyr]")
    ax.grid()
    if legend:
        ax.legend(loc='upper right')
    if ylabel:
        ax.set_ylabel(r"$P(A~|~ t_l \leq \tau \leq t_u~|~ S = D)$")
    
    if save:
        plt.savefig(os.path.join(parent_dir, filename + ext), dpi=dpi, transparent=True)
        
    return sum(a), sum(b), sum(c)



def areasExplanation3(IF, decay, CF, tau_lower, tau_upper, a, b, c, n_bins=500, ax=None, **kwargs):
    save = kwargs.pop('save', False)
    parent_dir = kwargs.pop('parent_dir', figures_dir)
    filename = kwargs.pop('filename', "Areas Explanation 2")
    ext = kwargs.pop('ext', ".pdf")
    dpi = kwargs.pop('dpi', 200)
    ylabel = kwargs.pop('ylabel', True)
    
    data = kwargs.pop('data', None)
    
    if ax is None: fig, ax = plt.subplots(dpi=200)
    
    ages = np.linspace(0, 1e10, n_bins)
    if data is None:
        data = PDF_tau_disrupted(IF, decay, CF, n_bins)
    x = np.concatenate(([0], ages/1e9, [10]))
    y = np.concatenate(([0], data, [0]))*1e9
    ax.plot(x, y, color='k')
    bounds = (x>tau_lower/1e9) & (x<tau_upper/1e9)
    ax.fill_between(x, 0, a/(a+b+c)*y, where=bounds, color='green', alpha=0.5, label=r"$\mathscr{A}_{cm}$")
    ax.fill_between(x, (a/(a+b+c))*y, (b/(a+b+c))*y+(a/(a+b+c))*y, where=bounds, color='yellow', alpha=0.5, label=r"$\mathscr{A}_{m}$")
    ax.fill_between(x, (b/(a+b+c))*y+(a/(a+b+c))*y, (c/(a+b+c))*y+(b/(a+b+c))*y+(a/(a+b+c))*y, where=bounds, color='blue', alpha=0.5, label=r"$\mathscr{A}_{c}$")
    ax.vlines(tau_lower/1e9, 0, y[np.where(x>tau_lower/1e9)[0][0]], colors='grey')
    ax.vlines(tau_upper/1e9, 0, y[np.where(x<tau_upper/1e9)[0][-1]], colors='grey')
    ax.set_xlabel(r'$t$ [Gyr]')
    if ylabel:
        ax.set_ylabel(r'$P(\tau = t~|~S=D)$')
    ax.set_xlim(0,10)
    ax.set_ylim(0,None)
    ax.grid()
    
    if save:
        plt.savefig(os.path.join(parent_dir, filename + ext), dpi=dpi, transparent=True)
    
    return data

    

def numericalPrediction(dist_type='PDF', n_bins=500, ax=None, **kwargs):
    
    data = kwargs.pop('data', None)
    legend = kwargs.pop('legend', False)
    ylim = kwargs.pop('ylim', True)
    
    ylabel = {'PDF': r'$P(\tau = t~|~S=D)$', 'CDF': r'$P(\tau < t~|~S=D)$',
              'CDF-U': r'$P(\tau < t~|~S=D) - P(u<t)$'}
    
    if ax is None: fig, ax = plt.subplots(dpi=200)
    IFs = [0.1, 0.4]
    decays = ['fast', 'slow']
    
    ages = np.linspace(0, 1e10, n_bins)
    
    if data is None:
        data = np.zeros((len(IFs),len(decays),n_bins))
        for i, IF in enumerate(IFs):
            for j, decay in enumerate(decays):
                if dist_type == 'PDF':
                    data[i,j] = PDF_tau_disrupted(IF, decay, 0.5, n_bins)
            
    for i, IF in enumerate(IFs):
        for j, decay in enumerate(decays):
            x = np.concatenate(([0], ages/1e9, [10]))
            y = np.concatenate(([0], data[i,j], [0]))
            if dist_type=='PDF':
                y = y*1e9
            if dist_type=='CDF': 
                y = y.cumsum()
                y = y/y[-1]
            if dist_type=='CDF-U':
                y = y.cumsum()
                y = y/y[-1] - np.linspace(0,1,n_bins+2)
            ax.plot(x, y, markers_dict[decay], color=colors_dict[IF])
    
            
    if legend:
        LoI_legend(ax, IFs, decays, bbox_to_anchor=(0.59,0), locs=('lower right','lower left'))
    ax.set_xlabel(r'$t$ [Gyr]')
    ax.set_ylabel(ylabel[dist_type])
    if ylim:
        ax.set_ylim(0, None)
    ax.grid()
    
    return data




###############################################################################
#######################-----PLOT-MERGING FUNCTIONS-----########################
###############################################################################



def ntBarGraph_(ns=None, **kwargs):
    
    save = kwargs.pop('save', False)
    parent_dir = kwargs.pop('parent_dir', figures_dir)
    filename = kwargs.pop('filename', "Number of Transits")
    ext = kwargs.pop('ext', ".pdf")
    dpi = kwargs.pop('dpi', 200)
    
    combos = [(0.1, 'slow'), (0.2, 'slow'), (0.4, 'slow')]
    I = len(combos)
    
    fig, ax = plt.subplots(ncols=I, sharey=True, figsize=[3.5*I,3], dpi=200)
    fig.subplots_adjust(wspace=0.1)
    
    for i in range(I):
        IF, decay = combos[i]
        print(f'{IF}, {decay}')
        dfs_k = loadMany(IF, decay, 0.5, NSIMS['kepler'], ns=ns)
        dfs_t = loadMany(IF, decay, 0.5, NSIMS['tess'], ns=ns)
        ntBarGraph(dfs_k, dfs_t, ax=ax[i], legend=(i==I-1), ylabel=(i==0))
        ax[i].set_title(r'$\bar{\mathscr{L}}$ = ' + f'{IF}')
        #ax[i].set_title(r'$\bar{\mathscr{L}}$ = ' + f'{IF}' + r', $\mathscr{D}$ = ' + f'{decay}')
        ax[i].set_xlabel(None)
    fig.supxlabel('Number of Transiting Planets per System', y=-0.05)
    
    if save:
        plt.savefig(os.path.join(parent_dir, filename + ext), dpi=dpi, transparent=True, bbox_inches='tight')
        
    return



def smdiComparison_(ns=None, **kwargs):
    
    save = kwargs.pop('save', False)
    parent_dir = kwargs.pop('parent_dir', figures_dir)
    filename = kwargs.pop('filename', "Fraction of Observed Systems By Disruption State")
    ext = kwargs.pop('ext', ".pdf")
    dpi = kwargs.pop('dpi', 200)
    
    observers = ['kepler', 'tess']
    combos = [(0.1, 'slow'), (0.2, 'slow'), (0.4, 'slow')]
    I = len(observers)
    J = len(combos)
    
    fig, ax = plt.subplots(nrows=I, ncols=J, sharex=True, sharey=True, figsize=[2*J,2*I], dpi=200)
    ax = ax.reshape(I,J)
    
    for i in range(I):
        observer = observers[i]
        for j in range(J):
            IF, decay = combos[j]
            print(f'{observer}, {IF}, {decay}')
            dfs = loadMany(IF, decay, 0.5, NSIMS[observer], ns=ns)
            smdiComparison(dfs, observer=observer, ax=ax[i,j], legend=(i==I-1 and j==J-1), title=(i==0), ylabel=(j==0))
    
    ax[0,0].text(-1.4, 0.5, 'Kepler', fontweight='bold', horizontalalignment='right')
    ax[1,0].text(-1.4, 0.5, 'TESS', fontweight='bold', horizontalalignment='right')
    
    if save:
        plt.savefig(os.path.join(parent_dir, filename + ext), dpi=dpi, transparent=True, bbox_inches='tight')
    
    return
    
    

def eccIncHistogram_(IF, decay, ns=None, **kwargs):
    
    save = kwargs.pop('save', False)
    parent_dir = kwargs.pop('parent_dir', figures_dir)
    filename = kwargs.pop('filename', "Eccentricity and Inclination Histograms")
    ext = kwargs.pop('ext', ".pdf")
    dpi = kwargs.pop('dpi', 200)
    
    #observers = ['kepler','tess']
    observers = ['inherent', 'kepler', 'tess']
    J = len(observers)
    
    fig, ax = plt.subplots(nrows=2, ncols=J, sharey='row', figsize=[3*J,4.5], dpi=200)
    fig.subplots_adjust(hspace=0.5)
    
    for j in range(J):
        observer = observers[j]
        print(f'{observer}, {IF}, {decay}')
        dfs = loadMany(IF, decay, 0.5, NSIMS[observer], ns=ns)
        eccIncHistogram(dfs, observer=observer, ax=ax[:,j], ylabel=(j==0), legend=True, title=True)
    ax[0,1].set_xlabel('Eccentricity')
    ax[1,1].set_xlabel(r'Inclination [$^{\circ}$]')
            
    if save:
        plt.savefig(os.path.join(parent_dir, filename + ext), dpi=dpi, transparent=True, bbox_inches='tight')
            
    return
    


def tauHistogram_(ns=None, **kwargs):
    
    data = kwargs.pop('data', None)
    save = kwargs.pop('save', False)
    parent_dir = kwargs.pop('parent_dir', figures_dir)
    filename = kwargs.pop('filename', "Tau Histogram")
    ext = kwargs.pop('ext', ".pdf")
    dpi = kwargs.pop('dpi', 200)
    observers = kwargs.pop('observers', ['inherent','kepler','tess'])
    combos = kwargs.pop('combos', [(0.1, 'slow'), (0.1, 'fast'), (0.4, 'slow'), (0.4, 'fast')])
    
    I = len(combos)
    J = len(observers)
    
    if ns is None: ns = [ns]*J
    if data is None: data = np.reshape([None]*I*J, (I,J))
    newdata = np.empty((I,J), dtype=object)
    
    fig, ax = plt.subplots(nrows=I, ncols=J, sharex=True, sharey=True, figsize=[4*J,3*I], dpi=200)
    ax = ax.reshape(I,J)
    
    for i in range(I):
        IF, decay = combos[i]
        ax[i,0].text(-0.3, 0.5, r'$\bar{\mathscr{L}}$ = ' + f'{IF}' + r', $\mathscr{D}$ = ' + f'{decay}', fontsize=13, verticalalignment='center', rotation=90, transform=ax[i,0].transAxes)
        for j in range(J):
            observer = observers[j]
            print(f'{observer}, {IF}, {decay}')
            dfs = loadMany(IF, decay, 0.5, NSIMS[observer], ns=ns[j], parent_dir=main_sims_dir) if data[i,j] is None else None
            newdata[i,j] = tauHistogram(dfs, observer=observer, ax=ax[i,j], data=data[i,j], legend=(i==I-1), title=(i==0), xlabel=(i==I-1), ylabel=(j==0))
    
    if save:
        plt.savefig(os.path.join(parent_dir, filename + ext), dpi=dpi, transparent=True, bbox_inches='tight')
        
    return newdata



def deltaTauMinAge_(ns=None, n_bins=100, log_bins=True, **kwargs):
    
    data = kwargs.pop('data', None)
    save = kwargs.pop('save', False)
    parent_dir = kwargs.pop('parent_dir', figures_dir)
    filename = kwargs.pop('filename', "Delta Tau Vs Min Age")
    ext = kwargs.pop('ext', ".pdf")
    dpi = kwargs.pop('dpi', 200)
    observers = kwargs.pop('observers', ['inherent','kepler','tess'])
    combos = kwargs.pop('combos', [(0.1, 'slow'), (0.1, 'fast'), (0.4, 'slow'), (0.4, 'fast')])
    
    IFs = list(set([combo[0] for combo in combos]))
    decays = list(set([combo[1] for combo in combos]))
    I = len(combos)
    J = len(observers)
    
    if data is None: data = np.reshape([None]*I*J, (I,J))
    newdata = np.zeros((I,J,2,n_bins))
    
    fig, ax = plt.subplots(ncols=J, figsize=[4.5*J,3.5], dpi=200)
    if J == 1: ax = [ax]
    fig.subplots_adjust(wspace=0.3)
    
    for i in range(I):
        IF, decay = combos[i]
        for j in range(J):
            observer = observers[j]
            print(f'{observer}, {IF}, {decay}')
            dfs = loadMany(IF, decay, 0.5, NSIMS[observer], ns=ns) if data[i,j] is None else None
            newdata[i,j] = deltaTauMinAge(dfs, observer=observer, n_bins=n_bins, log_bins=log_bins, data = data[i,j], marker=markers_dict[decay], color=colors_dict[IF], ax=ax[j])
    
    ylims = np.array([a.get_ylim() for a in ax])
    common_ylim = [np.min(ylims[:,0]), np.max(ylims[:,1])]
    [a.set_ylim(common_ylim) for a in ax]
    
    LoI_legend(ax[-1], IFs, decays, bbox_to_anchor=(0.64,0), locs=('lower right','lower left'))
    
    if save:
        plt.savefig(os.path.join(parent_dir, filename + ext), dpi=dpi, transparent=True, bbox_inches='tight')

    return newdata



def amdAgeScatter_(**kwargs):
    
    save = kwargs.pop('save', False)
    parent_dir = kwargs.pop('parent_dir', figures_dir)
    filename = kwargs.pop('filename', "Age Vs AMD")
    ext = kwargs.pop('ext', ".pdf")
    dpi = kwargs.pop('dpi', 200)
    
    combos = [(0.1, 'slow'), (0.1, 'fast'), (0.4, 'slow'), (0.4, 'fast')]
    I = len(combos)
    
    fig, ax = plt.subplots(ncols=I, sharey=True, figsize=[4.5*I,3], dpi=200)
    
    for i in range(I):
        IF, decay = combos[i]
        print(f'{IF}, {decay}')
        df = simulate(IF, decay, N=50000, logspace_ages=True)
        amdAgeScatter(df, ax=ax[i], legend=(i==I-1), ylabel=(i==0))
        ax[i].set_title(r'$\bar{\mathscr{L}}$ = ' + f'{IF}' + r', $\mathscr{D}$ = ' + f'{decay}')
    
    if save:
        plt.savefig(os.path.join(parent_dir, filename + ext), dpi=dpi, transparent=True, bbox_inches='tight')
    
    return



def tauHistogramCF_(ns=None, **kwargs):
    
    data = kwargs.pop('data', None)
    save = kwargs.pop('save', False)
    parent_dir = kwargs.pop('parent_dir', figures_dir)
    filename = kwargs.pop('filename', "Tau Histogram for Different CF")
    ext = kwargs.pop('ext', ".pdf")
    dpi = kwargs.pop('dpi', 200)
    observers = kwargs.pop('observers', ['inherent','kepler','tess'])
    CFs = kwargs.pop('CFs', [0, 1])
    
    I = len(CFs)
    J = len(observers)
    
    if ns is None: ns = [ns]*J
    if data is None: data = np.reshape([None]*I*J, (I,J))
    newdata = np.empty((I,J), dtype=object)
    
    fig, ax = plt.subplots(nrows=I, ncols=J, sharex=True, sharey=True, figsize=[4*J,3*I], dpi=200)
    ax = ax.reshape(I,J)
    
    for i in range(I):
        CF = CFs[i]
        ax[i,0].text(-0.3, 0.5, r'$f_c$ = ' + f'{CF}', fontsize=13, verticalalignment='center', rotation=90, transform=ax[i,0].transAxes)
        for j in range(J):
            observer = observers[j]
            print(f'{observer}')
            dfs = loadMany(0.1, 'slow', CF, NSIMS[observer], ns=ns[j], parent_dir=cf_sims_dir) if data[i,j] is None else None
            newdata[i,j] = tauHistogram(dfs, observer=observer, ax=ax[i,j], data=data[i,j], legend=(i==I-1), title=(i==0), xlabel=(i==I-1), ylabel=(j==0))
    
    if save:
        plt.savefig(os.path.join(parent_dir, filename + ext), dpi=dpi, transparent=True, bbox_inches='tight')
        
    return newdata



def areasExplanation_(**kwargs):
    
    data = kwargs.pop('data', None)
    save = kwargs.pop('save', False)
    parent_dir = kwargs.pop('parent_dir', figures_dir)
    filename = kwargs.pop('filename', "areasExplanationSarah")
    ext = kwargs.pop('ext', ".pdf")
    dpi = kwargs.pop('dpi', 200)
    
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=[8,10], sharey='row', dpi=200)
    fig.subplots_adjust(hspace=0.3, wspace=0.1)
    
    areasExplanation(0.1, 'slow', 4e9, 5e9, ax=ax[0,0])
    a,b,c = areasExplanation2(0.1, 'slow', 0.5, 4e9, 5e9, ax=ax[1,0], legend=False)
    data = areasExplanation3(0.1, 'slow', 0.5, 4e9, 5e9, a,b,c, ax=ax[2,0], data=data)
    
    areasExplanation(0.1, 'slow', 9e9, 10e9, ax=ax[0,1], legend=False, ylabel=False)
    a,b,c = areasExplanation2(0.1, 'slow', 0.5, 9e9, 10e9, ax=ax[1,1], legend=False, ylabel=False)
    data = areasExplanation3(0.1, 'slow', 0.5, 9e9, 10e9, a,b,c, ax=ax[2,1], data=data, ylabel=False)
    
    if save:
        plt.savefig(os.path.join(parent_dir, filename + ext), dpi=dpi, transparent=True, bbox_inches='tight')
    
    return data



def numericalPrediction_(**kwargs):
    
    data = kwargs.pop('data', None)
    save = kwargs.pop('save', False)
    parent_dir = kwargs.pop('parent_dir', figures_dir)
    filename = kwargs.pop('filename', "Numerical Prediction")
    ext = kwargs.pop('ext', ".pdf")
    dpi = kwargs.pop('dpi', 200)
    
    fig, ax = plt.subplots(ncols=3, figsize=[15,4], dpi=200)
    fig.subplots_adjust(wspace=0.4)
    
    
    data = numericalPrediction(dist_type='PDF', ax=ax[0], legend=True, data=data)
    numericalPrediction(dist_type='CDF', ax=ax[1], data=data)
    numericalPrediction(dist_type='CDF-U', ax=ax[2], ylim=False, data=data)
    
    if save:
        plt.savefig(os.path.join(parent_dir, filename + ext), dpi=dpi, transparent=True, bbox_inches='tight')
    
    return



def deltaTauMinAge__(ns=None, n_bins=100, log_bins=True, **kwargs):

    data = kwargs.pop('data', None)
    save = kwargs.pop('save', False)
    parent_dir = kwargs.pop('parent_dir', figures_dir)
    filename = kwargs.pop('filename', "Delta Tau Vs Min Age w Error")
    ext = kwargs.pop('ext', ".pdf")
    dpi = kwargs.pop('dpi', 200)
    observers = kwargs.pop('observers', ['inherent','kepler','tess'])
    #combos = kwargs.pop('combos', [(0.1, 'slow')])
    combos = kwargs.pop('combos', [(0.1, 'slow'), (0.1, 'fast'), (0.4, 'slow'), (0.4, 'fast')])
    
    #IFs = list(set([combo[0] for combo in combos]))
    #decays = list(set([combo[1] for combo in combos]))
    I = len(combos)
    J = len(observers)
    
    if ns is None: ns = [ns]*J
    if data is None: data = np.reshape([None]*I*J, (I,J))
    newdata = np.empty((I,J), dtype=object)
    
    fig, ax = plt.subplots(nrows=I, ncols=J, sharex=True, figsize=[4.5*J,3.5*I], dpi=200)
    if I ==1 or J == 1: ax = np.array([ax])
    fig.subplots_adjust(wspace=0.3)
    
    for i in range(I):
        IF, decay = combos[i]
        ax[i,0].text(-0.3, 0.5, r'$\bar{\mathscr{L}}$ = ' + f'{IF}' + r', $\mathscr{D}$ = ' + f'{decay}', fontsize=13, verticalalignment='center', rotation=90, transform=ax[i,0].transAxes)
        for j in range(J):
            observer = observers[j]
            print(f'{observer}, {IF}, {decay}')
            dfs = loadMany(IF, decay, 0.5, NSIMS[observer], ns=ns[j]) if data[i,j] is None else None
            newdata[i,j] = deltaTauMinAge(  dfs, 
                                            observer=observer, 
                                            n_bins=n_bins, 
                                            log_bins=log_bins, 
                                            data=data[i,j], 
                                            marker=markers_dict[decay], 
                                            color=colors_dict[IF], 
                                            ax=ax[i,j], 
                                            error=True,
                                            title=(i==0),
                                            xlabel=(i==(I-1)))
            #newdata[i,j] = deltaTauMinAge(dfs, observer=observer, n_bins=n_bins, log_bins=log_bins, data = data[i,j], marker=markers_dict[decay], color=colors_dict[IF], ax=ax[i,j], error=True)
    
    ylims = np.array([a.get_ylim() for a in ax.ravel()])
    common_ylim = [np.min(ylims[:,0]), np.max(ylims[:,1])]
    [a.set_ylim(common_ylim) for a in ax.ravel()]
    
    if save:
        plt.savefig(os.path.join(parent_dir, filename + ext), dpi=dpi, transparent=True, bbox_inches='tight')

    return newdata



    
    
    
    
