import numpy as np
import pickle as pk
import pyvista as pv
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from utils.logger import setup_logger
_headless = not bool(os.environ.get("DISPLAY", ""))
logger = setup_logger("MakePlot", level="INFO")
# initialize actor list 

class SetVisibilityCallback:
    """Helper callback to keep a reference to the actor being modified."""

    def __init__(self, actor, plotter,title, scalarbar):
        
        self.actor   = actor
        self.plotter = plotter 
        self.name    = title
        self.scalarbar=scalarbar
    def __call__(self, state):
        self.actor.SetVisibility(state)
        
        if self.scalarbar:
            if state: 
            
                self.plotter.add_scalar_bar(title  = self.name,
                                            color  = "white",
                                            mapper = self.actor.mapper)
            else:
                self.plotter.remove_scalar_bar(title=self.name)
                
            
def set_button(plotter,
               actor,
               title,
               scalarbar,
               color    = "white",
               inistate = True,
               posy     = 5.0,
               posx     = 5.0):
    callback = SetVisibilityCallback(actor,plotter,title,scalarbar)
    #if not inistate: 
    actor.SetVisibility(inistate)
    plotter.add_checkbox_button_widget(
        callback,
        value            =  inistate,
        position         =  (posx, posy),
        size             =  size,
        border_size      =  1,
        color_on         =  color,
        color_off        =  'grey',
        background_color =  'grey',
    )
    return

def add_vectors(plotter, sel_cond, func = np.log10, scale = 2, posy = 0.5, tol = 0.005):
        new_cond = np.logical_and(cond, sel_cond)
        nx,ny,nz = shocks[0+3][new_cond],shocks[1+3][new_cond], shocks[2+3][new_cond]
        vshocks  = shocks[7][new_cond]
        x1,y1,z1 = shocks[0][new_cond], shocks[1][new_cond], shocks[2][new_cond]
        gridcut  = pv.StructuredGrid(x1, y1, z1)
        vectors  = np.empty((gridcut.n_points, 3))
        # normalise vectors data to have values close to 1
        
        vectors[:, 0] = nx.flatten()#/vnorm
        vectors[:, 1] = ny.flatten()#/vnorm
        vectors[:, 2] = nz.flatten()#/vnorm
        
        title="V_sh [ufunc(cm/s)]"#r"$v_{\rm sh}$ ufunc [$\rm{cm\,s^{-1}}$] "
        #logger.info("Vectors done.")
        gridcut["vectors"]  = vectors
        gridcut[title]      = func(vshocks.flatten())/scale
        gridcut.set_active_vectors("vectors")
        arrows=gridcut.glyph(orient   = True,
                            tolerance = tol,
                            scale     = title)
        actor = plotter.add_mesh(arrows, opacity = 0.5,
                        scalar_bar_args          = {"color":"white"},
                        show_scalar_bar          = False)       
        #actor.GetProperty().getScaleFactor(1)
        set_button(plotter,actor,title,scalarbar = True, 
                                        color    = "green",
                                        posx     = 20 + size // 2,
                                        posy     = posy,
                                        inistate = False
                                        )

    
def add_a_conditioned_grid(sel_cond         = True,
                            qidx            = 6,
                            title           = None,
                            color           = "green",
                            posy            = 0.5,
                            unicolor        = None,
                            inistate        = True,
                            add_vector_field= True,
                            ufunc           = lambda a:a,
                            cmap            = "jet",
                            scale           = 1,
                            tol             = 0.005
                            ):
    if title is None: title = "No name"
    new_cond           = np.logical_and(cond, sel_cond)
    x1,y1,z1           = shocks[0][new_cond],shocks[1][new_cond],shocks[2][new_cond]
    family             = shocks[qidx][new_cond]
    points             = np.column_stack((x1, y1, z1))       
    point_cloud        = pv.PolyData(points)
    point_cloud[title] = ufunc(family)/scale
    # Create a PyVista plotter

    logger.info("Building plot for index %s... "%title)
    
    
    if unicolor is not None: 
        scalar_name = None
        cmap        = None
    else: 
        scalar_name = title
    actor     = plotter.add_points(
                    points          = point_cloud,
                    scalars         = scalar_name,
                    point_size      = pointsize,#0.8,
                    color           = unicolor,
                    cmap            = cmap,
                    style           = "points_gaussian",
                    emissive        = True,
                    log_scale       = True,
                    diffuse         = diffusivity,
                    nan_opacity     = 0.,
                    show_scalar_bar = False
                    )
    
    if unicolor is None and inistate==True: plotter.add_scalar_bar(title  = title,
                                                                   color  = "white",
                                                                   mapper = actor.mapper)
    
    set_button(plotter,actor,title, scalarbar = unicolor is None, inistate = inistate, color = color, posy = posy)
    
    plotter.add_text(
                title,
                position  = (40 + size//2 ,posy),
                color     = 'white',
                shadow    = False,
                font_size = 8,
            )
    if add_vector_field: add_vectors(plotter,
                                     new_cond,
                                     func  = np.log10,
                                     scale = 2,
                                     posy  = posy,
                                     tol   = tol)
    
    actor_list.append(actor)
    logger.info("Done.")
    return

class DiffuseSlider:
    def __init__(self, actor,plotter):#,**keywargs):
        self.output  = actor  # Expected PyVista mesh type
        self.plotter = plotter
    
    def __call__(self, value):
        self.output.GetProperty().SetDiffuse(value)
        
        return

    def update(self):
        # This is where you call your simulation
        self.plotter.render()
        return  
          
    
class apply_allD:
    def __init__(self, actors,plotter):#,**keywargs):
        self.output  = actors  # Expected PyVista mesh type
        self.routines= []
        for actor in actors:
            self.routines.append(DiffuseSlider(actor = actor, plotter = plotter))
        self.plotter = plotter
    def __call__(self, value):
        for run in self.routines:
            run(value)
            self.update()
        return    
    def update(self):
        # This is where you call your simulation
        self.plotter.render()
        return


class apply_allS:
    """Not working 
    """
    def __init__(self, actors,plotter):#,**keywargs):
        self.output  = actors  # Expected PyVista mesh type
        self.routines= []
        for actor in actors:
            self.routines.append(PointSizeSlider(actor = actor, plotter = plotter))
        self.plotter = plotter
    def __call__(self, value):
        for run in self.routines:
            run(value)
            self.update()
        return    
    def update(self):
        # This is where you call your simulation
        self.plotter.render()
        return    
class PointSizeSlider:
    """not working. SetPointSize apply only if render_points_as_spheres is True
    """
    def __init__(self, actor,plotter):#,**keywargs):
        self.output  = actor  # Expected PyVista mesh type
        self.plotter = plotter
    
    def __call__(self, value):
        #logger.info("hhh")
        self.output.GetProperty().SetPointSize(value)
        #self.update()
        return

    def update(self):
        # This is where you call your simulation
        self.plotter.render()
        return   
if __name__ == "__main__":
    #global actor_list  
    actor_list= []
    size        = 20  # button size
    # pointsize   = 1.8 # base size of the points in the plot. The are then "splattered" in a gaussian profile
    pointsize   = 0.8 # base size of the points in the plot. The are then "splattered" in a gaussian profile
      # list of actors used to apply general changes to the scene
                      # updated at the creation of each actor
    diffusivity = 0.1 # set how "luminous the points are. "
#    "./SB_plots/shocks+header.pk"
    with open(sys.argv[1], 'rb') as handle:
                shocksfinder_results=  pk.load(handle)
    # logger.info(shocksfinder_results[0])
    logger.info(shocksfinder_results[1][1])
    # exit()
    shocks = shocksfinder_results[0]
    header = shocksfinder_results[1][1]        
                                           # both types   # no error flags     # only conv. peaked at the center
    msg = "The data contains:\n"
    msg +="0  - x coordinate shock location\n"
    msg +="1  - y coordinate shock location\n"
    msg +="2  - z coordinate shock location\n"
    msg +="3  - x component of propagation vector\n"
    msg +="4  - y component of propagation vector\n"
    msg +="5  - z component of propagation vector\n"
    msg +="6  - shock family  12 for fast, 34 for slow\n"
    msg +="7  - vs: shock speed in km/s\n"
    msg +="8  - va: preshock Alfven velocity in km/s\n" 
    msg +="9  - MachAlf: Alfvenic mach number\n"
    msg +="10 - r: compression ratio\n"
    msg +="11 - rho0: preshock density in g/cm^3\n" 
    msg +="12 - B0: preshock magnetic field strength in microG\n" 
    msg +="13 - pmag_ratio: ratio of postshock to preshock magnetic pressure\n"
    
    
    logger.info(msg)
    cond   =True # np.logical_and(np.logical_and( shocks[6] > 0,    shocks[15]==0),    shocks[14]==1)
    cond2 = True # np.logical_and( shocks[1] >120, shocks[1] <128 )
    cond = np.logical_and(cond, cond2)

    pv.OFF_SCREEN = _headless  # must be set before creating the plotter
    logger.info("Creating plotter...")
    plotter = pv.Plotter()
    add_a_conditioned_grid(sel_cond = shocks[6]==12, title = "FS", color = "blue",posy = size*2+size//10, unicolor = "blue" ,inistate = True, add_vector_field=True)
    add_a_conditioned_grid(sel_cond = shocks[6]==34, title = "SS", color = "red", posy = size*3+size//10, unicolor = "red"  ,inistate = True, add_vector_field=True)
    pos = 4
    for i in range(7,14):
        logger.info(f"Adding plot for index {i} - {header[i]}...")
        add_a_conditioned_grid(sel_cond = shocks[7] > 1e5, # only shocks moving at at least a km/s
                            qidx     = i,
                            title    = header[i],
                            ufunc    = lambda a: a,
                            color    = "yellow",
                            posy     = size*pos + size//10,
                            inistate = False ,
                            add_vector_field = False)
        pos+=1

    plotter.set_background("black")
    logger.info("Plotter created.")
    plotter.show_bounds(    color     = "white",
                            bold      = False,
                            location  = 'outer',
                            ticks     = 'both',
                            n_xlabels = 4,
                            n_ylabels = 4,
                            n_zlabels = 4,
                            xtitle    = "x",
                            ytitle    = "y",
                            ztitle    = "z",
                            font_size = 20,
                            )

    if not _headless:
        engineD = apply_allD(actor_list, plotter)
        plotter.add_slider_widget(
            callback     = lambda value: engineD(10**value),
            rng          = [-2.5, 0],
            value        = -1.0,
            title        = "Log10(Diffuse)",
            pointa       = (0.8, 0.8),
            pointb       = (1.0, 0.8),
            title_height = 0.03,
            title_color  = "white",
            color        = "white",
            slider_width = 0.01,
            tube_width   = 0.001,
        )

    out_png = sys.argv[1].replace("_result.pk", "_3Dplot.png")
    logger.info("Rendering plot...")
    plotter.show(screenshot    = out_png,
                 window_size   = [2048, 2048],
                 auto_close    = _headless,
                 interactive   = not _headless)
    logger.info(f"Screenshot saved: {out_png}")