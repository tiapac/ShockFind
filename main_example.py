import pickle
import gc
import AnalysisPlayground.analysis_tools as an
# import  shock_finder as css
# try:
from ShockFind import shock_finder
from ShockFind.utils.logger import setup_logger, loglevels
# except ImportError:
#     from src.shockfind_interface import shock_finder
#     from utils.logger import setup_logger, loglevels
    
import numpy as np
import matplotlib.pyplot as plt
import os

def _mpi_rank():
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank()
    except ImportError:
        return 0

_rank = _mpi_rank()
logger = setup_logger("main_example", level=loglevels.INFO)
setup_logger("ShockFind",            level=loglevels.INFO)   # show ShockFind library messages



level= 8
outnumb = 25

analysis_name = "LB_double"
analysis_name += "lvl_%02d_%05d"%(level, outnumb)


SNAPSHOT_PREFIX = f"output_{outnumb:05d}"



DATA_PATH = "/mnt/beegfs/projects/hpcc250512a2/mpacicco/simulations/BarrosBugFix/outputs_blowoutOK_double/"
snap_path  = DATA_PATH + SNAPSHOT_PREFIX

RESULTS_PATH = DATA_PATH + "shockfind_results/"
rescale=0

# # if snapshot is a DIRECTORY (RAMSES) THEN WE PUT IT INTO THE SNAPSHOT ITSELF.
# if os.path.isdir(snap_path):
#     CACHE_PATH = snap_path + "/__shockfind_cache/"
# else:
# else, we create a cache directory in the same folder as the snapshot file (ENZO, FLASH, etc.)
CACHE_PATH = DATA_PATH + "__shockfind_cache/"

load = False

if __name__=="__main__":
    names=[
            "velocity_x",
            "velocity_y",
            "velocity_z",
            "magnetic_field_x",
            "magnetic_field_y",
            "magnetic_field_z",
            "pressure",
            "density",
             "dx"]
    
   
    ### collecting data from simulations
    def loadytData():  
        def fromScratch():
            os.makedirs(CACHE_PATH, exist_ok=True)
            if _rank == 0:
                logger.info("no cache found — loading simulation data from scratch")
            else: 
                logger.error("no cache found, but this is a worker rank — this should not happen, check your setup")
                raise RuntimeError("abort.")
            
            ds = an.analyse(outpath=DATA_PATH, outnumb=outnumb)
            dx   = (ds.width/2**level)[0].in_cgs().d
            dens, cube = ds.get_cube(level=level, field="density", ghost=1)
            
            
            if level > 8:
                directory2 = f"{CACHE_PATH}/cached_heavy_lvl_{level}_output_{outnumb:05d}/"
                
                os.makedirs(directory2, exist_ok=True)
                 
                for name in names:
                    with open(directory2+"%s.pkl"%name, 'wb') as handle:
                        if name=="dx":
                            c=pickle.dump(dx, handle)      
                        else:
                            data,cube=ds.get_cube(level=level, field=name, ghost=1)
                            c=pickle.dump(data.in_cgs().d, handle)
                            del data,cube,c
                            gc.collect()                       
            else:
                vx  =  cube["gas",  "velocity_x"].in_cgs().d
                vy  =  cube["gas",  "velocity_y"].in_cgs().d
                vz  =  cube["gas",  "velocity_z"].in_cgs().d
                Bx  =  cube["gas",  "magnetic_field_x"].in_cgs().d
                By  =  cube["gas",  "magnetic_field_y"].in_cgs().d
                Bz  =  cube["gas",  "magnetic_field_z"].in_cgs().d
                P   =  cube["gas",  "pressure"        ].in_cgs().d
                rho =  cube["gas",  "density"].in_cgs().d
                datas = [vx,vy,vz,Bx,By,Bz,P,rho,dx]
                #os.mkdir("pickled_data_from_SNR")
                with open(CACHE_PATH+f"/cached_lvl_{level}_output_{outnumb:05d}.pkl", 'wb') as handle:
                    pickle.dump(datas, handle)
                return datas
        try:
            if level > 8:
                a=[]        
                for name in names:
                    directory2 = CACHE_PATH+f"/cached_heavy_lvl_{level}_output_{outnumb:05d}/"
                    with open(directory2+"%s.pkl"%name, 'rb') as handle:
                            a.append(pickle.load( handle))
            else:
                with open(CACHE_PATH+f"/cached_lvl_{level}_output_{outnumb:05d}.pkl", 'rb') as handle:
                    a=pickle.load( handle)
            
            vx, vy, vz, Bx, By, Bz, P, rho, dx = a
            del a
            if _rank == 0:
                logger.info("data loaded from cache")
            else: 
                logger.error("no cache found, but this is a worker rank — this should not happen, check your setup")
                raise RuntimeError("abort.")
            return [vx, vy, vz, Bx, By, Bz, P, rho, dx]
        except Exception as e:
            return fromScratch()    
        
    
#########################################################################################################
    
    shocksfinder=shock_finder(name=analysis_name)
    
    if load:
        shocksfinder.load_results(path=RESULTS_PATH, name=analysis_name)
  
        
    else:
        dx = None  # workers never use dx; rank 0 overwrites below

        if _rank == 0:
            vx, vy, vz, Bx, By, Bz, P, rho, dx = loadytData()
            shocksfinder.load_data(rho,       # density
                            [vx,vy,vz], # velocity field
                            [Bx,By,Bz], # magnetic field
                            P         , # pressure
                            dx = dx     # spatial resolution - only for uniform grid
                            )
            shocksfinder.set_thresholds(dx         = dx,
                                vshock_min = 5e5,
                                rhomean    = 3.e-24
                                )

        # set code parameters (all ranks need self.extra for the C++ binding)
        shocksfinder.extra_params(periodic     = [ False ] * 3,
                            gamma        = 1.666667 )
        # find candidates — worker ranks return immediately; only rank 0 searches
        shocksfinder.find_candidates(use_gradTRho = True,
                               dx           = dx,
                               mach_min     = 1.5,    # raise from default 1.0
                               #vmag_min    = 5e5,    # optional: absolute velocity floor (cm/s)
                               )
        # run analysis of the shock candidates
        shocksfinder.analyse_candidates(quiet=False,
                                        rescale=rescale,
                                        use_cpp=True)
        # collect resulting data, only shock position and firectional information  
        shocksfinder.shocks_data()
        
        # save the results to file for later inspection 
        shocksfinder.save_results(path=RESULTS_PATH, name=analysis_name)
    #finally:
    # make a 3D plot with all the shocks found
    #shocksfinder.plot3D(types="sf", alpha=0.1, ss=1)
    # make histograms of the various shock quantities
   
    names = {i:name for i,name in enumerate(shocksfinder.header[1])}
    shared = dict(bins=15 , log=True , alpha=0.8, histtype="step", lw=2, density=True)
    ax, fig = shocksfinder.histograms(9, **shared )
    shocksfinder.histograms(10, ax = ax, fig=fig, linestyle="--", **shared )
    shocksfinder_results= shocksfinder.results#shocksfinder.load_results(path=directory, name=analysis_name)

    shocks = shocksfinder_results[0]#.shocks
    # cond   = np.logical_and(
    #                         np.logical_and(
    #                                         shocks[6] > 0,
    #                                         shocks[15]==0
    #                                       ),
    #                         shocks[14]==1
    #                        )
    # header=shocksfinder_results[1][1]
    # with open(f"{RESULTS_PATH}/{analysis_name}_shocks+header_{level}.pkl", 'wb') as handle:
    #               pickle.dump([shocks,header], handle)
    shocksfinder.plot3D(types = "fs")
    # plt.show(a)
    quit()
