#Copyright 2016 Andrew Lehmann
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import traceback
import time
import numpy as np
import matplotlib.pyplot as plt
from numpy import (
    average, 
    std, 
    empty, 
    where, 
    dot,
    log10,
    unique, 
    arange,
    array,
    intersect1d,
    sqrt as npsqrt
    )
from math import log10, pi, sqrt, fsum







class core:
    def __init__(self, hydro_only = False, use_pressure = False) -> None:
        self.hydro_only   = hydro_only
        self.use_pressure = use_pressure
        pass
    #######################################################################
    def masked_search(self, searchcube, mask):
        '''
        Give the coordinates of the maximum value of a masked cube.
        
        Parameters
        ----------
        searchcube : 3d array
            cube of values used to find shocks, e.g. div or grad 
        mask : 3d array
            mask of locations already searched
        
        Returns
        ----------
        location : ndarray
            the indices (location) of the shock        
        '''
        mx=np.ma.masked_array(searchcube,mask=mask)
        location = where(mx==mx.max())
        del mx
        if len(location[0]) > 1:
            location = np.array([location[0][0]]),np.array([location[1][0]]),np.array([location[2][0]])
            return location
        else:
            return location
    #######################################################################

























    #######################################################################
    def shock_normal(self, cube, method='point_gradient'):
        '''
        Function that defines the unit vector perpendicular to the shock
        front pointing in the direction of shock propagation.

        Parameters
        ----------
        cube : dict
            dictionary with content depending on method
        method : str
            determines definition method, choices are:
            
                'point_gradient'
                    normal is negative the grad(rho) vector. cube must
                    then contain the following entries:
                    
                    'idx' : 3-tuple
                        location of shock front
                    'gradx' : 3d array
                        x-cpt of grad(rho)
                    'grady' : 3d array
                        y-cpt of grad(rho)
                    'gradz' : 3d array
                        z-cpt of grad(rho)
                    
                'average_gradient'
                    normal is the average density gradient vector 
                    weighted by some chosen weight. cube must then contain 
                    the following entries:
                    
                    'idx' : 3-tuple
                        location of shock front
                    'gradx' : 3d array
                        x-cpt of grad(rho)
                    'grady' : 3d array
                        y-cpt of grad(rho)
                    'gradz' : 3d array
                        z-cpt of grad(rho)
                    'weight' : 3d array
                        weighting array
                    'radius' : float
                        averaging radius
                    'xmax' : int
                        x limit
                    'ymax' : int
                        y limit
                    'zmax' : int
                        z limit
                    'periodic_x' : bool
                        periodic at x-boundaries?
                    'periodic_y' : bool
                        periodic at y-boundaries?
                    'periodic_z' : bool
                        periodic at z-boundaries?
        
        Returns
        -------
        ns : 3-tuple 
            shock normal vector
        '''
        
        if method == 'point_gradient':
            if ('idx' and 'gradx' and 'grady' and 'gradz') in cube:
                x,y,z = cube['idx'][0], cube['idx'][1], cube['idx'][2]
                grad_mag = npsqrt(cube['gradx'][x,y,z]**2 + cube['grady'][x,y,z]**2 + cube['gradz'][x,y,z]**2)

                if grad_mag == 0:
                    print(' There is no gradient in the density here.')
                    return 'error'
                else:
                    nx = -cube['gradx'][x,y,z]/grad_mag
                    ny = -cube['grady'][x,y,z]/grad_mag
                    nz = -cube['gradz'][x,y,z]/grad_mag
                    return (nx,ny,nz)
            else:
                print('************')
                print('ERROR: incorrect data for the shock normal method.')
                print('Check documentation for function \'shock_normal\'.')
                print('************')
                quit()
        elif method == 'average_gradient':
            if ('idx' and 'gradx' and 'grady' and 'gradz' and 'radius' and 'weight' and 'xmax' and 'ymax' and 'zmax' and 'periodic_x' and 'periodic_y' and 'periodic_z') in cube:
                loc = cube['idx']
                R = cube['radius']

                #x,y,z = [],[],[]
                #w = []
                
                # Deal with crap near the boundary. If the box is periodic,
                # then we wrap around. If not, we restrict our averaging
                # sphere to not go to forbidden indices.
                if cube['periodic_x']:
                    x1, x2 = loc[0]-R, (loc[0]+R+1)%cube['xmax']
                else:
                    x1, x2 = max(0,loc[0]-R), min(loc[0]+R+1,cube['xmax'])

                if cube['periodic_y']:
                    y1, y2 = loc[1]-R, (loc[1]+R+1)%cube['ymax']
                else:
                    y1, y2 = max(0,loc[1]-R), min(loc[1]+R+1,cube['ymax'])

                if cube['periodic_z']:
                    z1, z2 = loc[2]-R, (loc[2]+R+1)%cube['zmax']
                else:
                    z1, z2 = max(0,loc[2]-R), min(loc[2]+R+1,cube['zmax'])

                
                x,y,z,w=0,0,0,0
                for a in range(x1,x2):
                    for b in range(y1,y2):
                        for c in range(z1,z2):
                            if (((a-loc[0])**2 + (b-loc[1])**2 + (c-loc[2])**2)**0.5 <= R):
                                x+=cube['weight'][a,b,c]*cube['gradx'][a,b,c]
                                y+=cube['weight'][a,b,c]*cube['grady'][a,b,c]
                                z+=cube['weight'][a,b,c]*cube['gradz'][a,b,c]
                                w+=cube['weight'][a,b,c]
                                #x.append(cube['weight'][a,b,c]*cube['gradx'][a,b,c])
                                #y.append(cube['weight'][a,b,c]*cube['grady'][a,b,c])
                                #z.append(cube['weight'][a,b,c]*cube['gradz'][a,b,c])
                                #w.append(cube['weight'][a,b,c])

                #ax, ay, az = sum(x)/sum(w), sum(y)/sum(w), sum(z)/sum(w)
                ax, ay, az = x/w, y/w, z/w
                a_mag = (ax**2 + ay**2 + az**2)**0.5
                
                nx = -ax/a_mag
                ny = -ay/a_mag
                nz = -az/a_mag

                return (nx,ny,nz)
            else:
                print('************')
                print('ERROR: incorrect data for the shock normal method.')
                print('Check documentation for function \'shock_normal\'.')
                print('************')
                quit()
        else:
            print('************')
            print('ERROR: Shock normal method \'' + method + '\' doesn\'t exist.')
            print('Check config file and documentation for function \'shock_normal\'.')
            print('************')
            quit()
    #######################################################################














    #######################################################################
    def transerve_direction(self, data, method='point_field'):
        '''
        Function that defines the transverse direction in the magnetic
        field - velocity plane. This plane should not change across
        a static shock.

        Parameters
        ----------
        data : dict
            dictionary with content depending on method
        method : str
            determines definition method, choices are
            
                'point_field'
                    transverse is the magnetic field vector at some 
                    reference point in the shock line. data must contain
                    the following entries:
                    
                        'ref': int
                            shock line index for reference point
                        'bx': 1d array
                            x-cpt of b on the shock line
                        'by': 1d array
                            y-cpt of b on the shock line
                        'bz': 1d array
                            z-cpt of b on the shock line
                        'ns': 3-tuple
                            shock normal vector
                    
                'average_field'
                    transverse is the average magnetic field vector in 
                    some region of the shock line. data must contain 
                    the following entries:
                
                        'region': 1d array
                            shock line indices of region to average the 
                            field over
                        'bx': 3d array
                            x-cpt of b field
                        'by': 3d array
                            y-cpt of b field
                        'bz': 3d array
                            z-cpt of b field
                        'ns': 3-tuple
                            shock normal vector
        
        Returns
        -------
            nt1 : 3-tuple
                unit vector perpendicular to shock normal in v-B plane
            nt2 : 3-tuple
                unit vector perpendicular to shock normal and nt1
        '''

        if method == 'hydro':
            # No magnetic field: build arbitrary frame perpendicular to ns
            nsx, nsy, nsz = data['ns']
            ref = (1.0, 0.0, 0.0) if abs(nsx) < 0.9 else (0.0, 1.0, 0.0)
            nt2x = nsy*ref[2] - nsz*ref[1]
            nt2y = nsz*ref[0] - nsx*ref[2]
            nt2z = nsx*ref[1] - nsy*ref[0]
            nt2mag = npsqrt(nt2x**2 + nt2y**2 + nt2z**2)
            nt2x, nt2y, nt2z = nt2x/nt2mag, nt2y/nt2mag, nt2z/nt2mag
            nt1x = nt2y*nsz - nt2z*nsy
            nt1y = nt2z*nsx - nt2x*nsz
            nt1z = nt2x*nsy - nt2y*nsx
            return [(nt1x, nt1y, nt1z), (nt2x, nt2y, nt2z)]

        if method == 'point_field':
            if ('ref' and 'bx' and 'by' and 'bz' and 'ns') in data:
                # Use the magnetic field at some reference place, use it as
                # reference vector, b.
                # nt2 = ns cross b
                # nt1 = nt2 cross ns

                nsx, nsy, nsz = data['ns']

                bmag = npsqrt(data['bx'][data['ref']]**2. + data['by'][data['ref']]**2. + data['bz'][data['ref']]**2.)
                try:
                    bx = data['bx'][data['ref']][0]/bmag
                    by = data['by'][data['ref']][0]/bmag
                    bz = data['bz'][data['ref']][0]/bmag
                except:
                    bx = data['bx'][data['ref']]/bmag
                    by = data['by'][data['ref']]/bmag
                    bz = data['bz'][data['ref']]/bmag
                    

                nt2x, nt2y, nt2z = nsy*bz-nsz*by, nsz*bx-nsx*bz, nsx*by-nsy*bx
                nt2mag = npsqrt(nt2x**2. + nt2y**2. + nt2z**2.)
                nt2x, nt2y, nt2z = nt2x/nt2mag, nt2y/nt2mag, nt2z/nt2mag # normalise

                nt1x, nt1y, nt1z = nt2y*nsz-nt2z*nsy, nt2z*nsx-nt2x*nsz, nt2x*nsy-nt2y*nsx
                #print nt1x
                #print nt1x[0]
                #return [(nt1x[0],nt1y[0],nt1z[0]),(nt2x[0],nt2y[0],nt2z[0])]
                try:
                    return [(nt1x[0],nt1y[0],nt1z[0]),(nt2x[0],nt2y[0],nt2z[0])]
                except:
                    return [(nt1x,nt1y,nt1z),(nt2x,nt2y,nt2z)]
            else:
                print('************')
                print('ERROR: incorrect data for the shock transverse method.')
                print('Check documentation for function \'transerve_direction\'.')
                print('************')
                quit()
        if method == 'average_field':
            if ('region' and 'bx' and 'by' and 'bz' and 'ns') in data:
                # Form the average magnetic field direction, use it as 
                # reference vector, b.
                # Then let nt2 = ns cross b, (ns is shock normal)
                # Then let nt1 = nt2 cross ns
                
                avg_bx = average(data['bx'][data['region']])
                avg_by = average(data['by'][data['region']])
                avg_bz = average(data['bz'][data['region']])
                
                mag = npsqrt(avg_bx**2. + avg_by**2. + avg_bz**2.)
                bx, by, bz = avg_bx/mag, avg_by/mag, avg_bz/mag
                nsx, nsy, nsz = data['ns']

                nt2x, nt2y, nt2z = nsy*bz-nsz*by, nsz*bx-nsx*bz, nsx*by-nsy*bx
                nt2mag = npsqrt(nt2x**2. + nt2y**2. + nt2z**2.)
                nt2x, nt2y, nt2z = nt2x/nt2mag, nt2y/nt2mag, nt2z/nt2mag # normalise

                nt1x, nt1y, nt1z = nt2y*nsz-nt2z*nsy, nt2z*nsx-nt2x*nsz, nt2x*nsy-nt2y*nsx 
                return [(nt1x,nt1y,nt1z),(nt2x,nt2y,nt2z)]
            else:
                print('************')
                print('ERROR: incorrect data for the shock transverse method.')
                print('Check documentation for function \'transerve_direction\'.')
                print('************')
                quit()
        else:
            print('************')
            print('ERROR: Shock normal method \'' + method + '\' doesn\'t exist.')
            print('Check config file and documentation for function \'shock_normal\'.')
            print('************')
            quit()
    #######################################################################



















    #######################################################################
    def cylinder(self, Rcyl, line_coords, n_shock, nt1, nt2, rho, pres, vx, vy, vz, div, bx=None, by=None, bz=None):
        '''
        Averaging over a cylinder
        
        Parameters
        ----------
        Rcyl : int
            Radius of cylinder
        line_coords : list of 3-tuples
            line coordinates
        nt1, nt2 : 3-tuple
            transverse normal vectors
        rho : 3d array
            mass density
        pres : 3d array
            pressure
        vx, vy, vz : 3d arrays
            x,y and z velocity cubes
        bx, by, bz : 3d arrays
            x,y and z magnetic field cubes
        div: 3d array
            divergence cube

        Returns
        -------
        rho_line : 1d array
            cylinder averaged (mass) density
        p_line : 1d array
            cylinder averaged pressure
        up_line : 1d array
            cylinder averaged parallel-velocity
        ut1_line : 1d array
            cylinder averaged transverse-velocity 1
        ut2_line : 1d array
            cylinder averaged transverse-velocity 2
        bp_line : 1d array
            cylinder averaged parallel-magnetic field
        bt1_line : 1d array
            cylinder averaged transverse-magnetic field 1
        bt2_line : 1d array
            cylinder averaged transverse-magnetic field 2
        conv_line : 1d array
            cylinder averaged convergence
        '''

        nt1x = nt1[0]
        nt1y = nt1[1]
        nt1z = nt1[2]
        nt2x = nt2[0]
        nt2y = nt2[1]
        nt2z = nt2[2]

        rho_line  =  empty(len(line_coords[0]), dtype='float')
        p_line    =  empty(len(line_coords[0]), dtype='float')
        up_line   =  empty(len(line_coords[0]), dtype='float')
        ut1_line  =  empty(len(line_coords[0]), dtype='float')
        ut2_line  =  empty(len(line_coords[0]), dtype='float')
        bp_line   =  empty(len(line_coords[0]), dtype='float')
        bt1_line  =  empty(len(line_coords[0]), dtype='float')
        bt2_line  =  empty(len(line_coords[0]), dtype='float')
        conv_line =  empty(len(line_coords[0]), dtype='float')
        for num,coord in enumerate(zip(line_coords[0],line_coords[1],line_coords[2])):
            a=coord[0]
            b=coord[1]
            c=coord[2]
            # For dynamic averaging, initial average is zero
            N_avg = 0 # this is the index for the dynamic averaging
            logrho_avg = 0
            logp_avg   = 0
            up_avg     = 0
            ut1_avg    = 0
            ut2_avg    = 0
            bp_avg     = 0
            bt1_avg    = 0
            bt2_avg    = 0
            conv_avg   = 0
            
            for mux in range(-Rcyl,Rcyl+1):
                for muy in range(-Rcyl,Rcyl+1):                        
                    if (((mux*nt2x + muy*nt1x)**2 + (mux*nt2y + muy*nt1y)**2 + (mux*nt2z + muy*nt1z)**2)**0.5 <= Rcyl):
                        cpx = int(round(a + mux*nt2x + muy*nt1x,0))
                        cpy = int(round(b + mux*nt2y + muy*nt1y,0))
                        cpz = int(round(c + mux*nt2z + muy*nt1z,0))

                        try:
                            vel = (
                                vx[cpx,cpy,cpz],
                                vy[cpx,cpy,cpz],
                                vz[cpx,cpy,cpz]
                            )
                        except:
                            continue

                        bfield = (
                            bx[cpx,cpy,cpz] if bx is not None else 0.0,
                            by[cpx,cpy,cpz] if by is not None else 0.0,
                            bz[cpx,cpy,cpz] if bz is not None else 0.0,
                        )
                        
                        logrho_avg = (N_avg*logrho_avg + log10(rho[cpx,cpy,cpz]))/(N_avg + 1)
                        logp_avg = (N_avg*logp_avg + log10(pres[cpx,cpy,cpz]))/(N_avg + 1)
                        
                        up_avg  = (N_avg*up_avg  + dot(n_shock,vel))/(N_avg + 1)
                        ut1_avg = (N_avg*ut1_avg + dot(nt1,vel))/(N_avg + 1)
                        ut2_avg = (N_avg*ut2_avg + dot(nt2,vel))/(N_avg + 1)

                        bp_avg  = (N_avg*bp_avg  + dot(n_shock,bfield))/(N_avg + 1)
                        bt1_avg = (N_avg*bt1_avg + dot(nt1,bfield))/(N_avg + 1)
                        bt2_avg = (N_avg*bt2_avg + dot(nt2,bfield))/(N_avg + 1)

                        conv_avg = (N_avg*conv_avg - div[cpx,cpy,cpz])/(N_avg + 1)
                        
                        N_avg +=1

            
            # Density averaged over the circle
            rho_line[num] = 10.**logrho_avg
            
            # Pressure averaged over the circle
            p_line[num] = 10.**logp_avg
            
            # Velocity vectors averaged over the circle
            up_line[num]  = up_avg
            ut1_line[num] = ut1_avg
            ut2_line[num] = ut2_avg
            
            # Magnetic field vectors averaged over the circle
            bp_line[num]  = bp_avg
            bt1_line[num] = bt1_avg
            bt2_line[num] = bt2_avg

            conv_line[num] = conv_avg
    
        return rho_line, p_line, up_line, ut1_line, ut2_line, bp_line, bt1_line, bt2_line, conv_line
    #######################################################################
                    












    #######################################################################
    def find_shocks(self, conv, grad, conv_threshold, threshold_grad, Ncells, block=0, quiet=False, ID=''):
        '''
        Function to flag shock candidate cells. Outputs a list of
        coordinates to be used by extract function.
        
        Parameters
        ----------
            conv : 3d array
                array of convergences to be searched
            grad : 3d array
                array of gradient magnitudes to be
                searched. shape(grad) must equal shape(conv).
            conv_threshold : float
                find_shocks searches for cells with
                convergence values above this threshold
            threshold_grad : float
                find_shocks searches for cells with
                gradient magnitudes above this threshold
            Ncells : int
                max number of cells to flag, find_shocks stops
                searching beyond this number
            block : int
                blocking length around shock candidate
        
        Returns
        -------
            shocklist : list
                list of locations of shock candidates [list of 3-tuples]
        '''
        

        Ncells=int(Ncells)
        if not quiet:
            print('*{0:s} Searching down to convergence of {1:.3f}. Then for gradients down to {2:.3f}.'.format(ID, conv_threshold,threshold_grad))
            print('*{0:s} Search limit set to {1:d} cells.'.format(ID, Ncells))
            
        search_loop = 'conv'
        search_iterator = conv.copy()
        threshold=conv_threshold
        mask = np.zeros_like(search_iterator, dtype=np.int64)



        xmax = np.shape(conv)[0] # x-limit
        ymax = np.shape(conv)[1] # y-limit
        zmax = np.shape(conv)[2] # z-limit
        
        i=0
        j=0
        
        x_loc, y_loc, z_loc = np.array([],dtype=np.int64),np.array([],dtype=np.int64),np.array([],dtype=np.int64)
        cell_conv, cell_grad = np.array([]),np.array([])
        
        # Search through convergences until threshold or Ncells is reached
        while search_loop == 'conv' or search_loop =='grad':
            t0 = time.time()
            j+=1
            idx_tmp = self.masked_search(search_iterator, mask)
            #idx_abs = (idx_tmp[0]+loadx1,idx_tmp[1]+y_low,idx_tmp[2]+z_low) # used to be absolute coordinates, but now this function won't know about it
            
            mask[idx_tmp]=1 # block out this candidate in the mask
            
            
            # Break the loop when the search limit is reached
            if j == Ncells+2:
                print('*{0:s} Reached shock search limit: {1:d}'.format(ID, Ncells))

                if search_loop == 'conv':
                    output_conv = [
                        x_loc,
                        y_loc,
                        z_loc,
                        cell_conv,
                        cell_grad
                    ]
                    
                    output_grad = [
                        np.array([]),
                        np.array([]),
                        np.array([]),
                        np.array([]),
                        np.array([])
                    ]
                elif search_loop == 'grad':
                    output_grad = [
                        x_loc,
                        y_loc,
                        z_loc,
                        cell_conv,
                        cell_grad
                    ]
                break
                
            # Break the loop when the search threshold is reached
            elif search_iterator[idx_tmp][0] < threshold:
                #print(search_iterator[idx_tmp][0])
                if search_loop=='grad':
                    if not quiet:
                        print('*{0:s} grad threshold reached.'.format(ID))

                    output_grad = [
                        x_loc,
                        y_loc,
                        z_loc,
                        cell_conv,
                        cell_grad
                    ]
                    break
                    
                else:
                    print('*{0:s} conv threshold reached, searching through gradients now.'.format(ID)
    )
                    output_conv = [
                        x_loc,
                        y_loc,
                        z_loc,
                        cell_conv,
                        cell_grad
                    ]

                    search_iterator=grad.copy()
                    threshold=threshold_grad
                    search_loop = 'grad'
                    x_loc = np.array([],dtype=np.int64)
                    y_loc = np.array([],dtype=np.int64)
                    z_loc = np.array([],dtype=np.int64)
                    cell_conv=[]
                    cell_grad=[]
                    continue

            # Ignore candidates that are diverging
            if conv[idx_tmp][0] <= 0:
                continue

            
            
            
            print( '{0:s} #{1:d} ({2:d},{3:d},{4:d}), {5:s}: {6:.3f}, threshold: {7:.3f}, t: {8:.3f} s'.format(ID, i+1, idx_tmp[0][0], idx_tmp[1][0], idx_tmp[2][0], search_loop, search_iterator[idx_tmp][0], threshold, time.time()- t0))
                            
                            
            # Save shock location, gradient, convergence
            #iii, jjj, kkk = [idx[0] for idx in idx_tmp]
            #print(iii,jjj,kkk, grad)
            x_loc=np.append(x_loc, idx_tmp[0][0])
            y_loc=np.append(y_loc, idx_tmp[1][0])
            z_loc=np.append(z_loc, idx_tmp[2][0])
            cell_conv=np.append(cell_conv, conv[idx_tmp][0])
            #cell_conv=np.append(cell_conv, conv[iii,jjj,kkk])
            cell_grad=np.append(cell_grad, grad[idx_tmp][0])
            #cell_grad=np.append(cell_grad, grad[iii,jjj,kkk])
            
            i+=1


            if block > 0:
                # Block out the surrounding region in the search iterator cube
                x1 = int(max(0., idx_tmp[0][0]-block))
                y1 = int(max(0., idx_tmp[1][0]-block))
                z1 = int(max(0., idx_tmp[2][0]-block))

                x2 = int(min(xmax, idx_tmp[0][0]+block+1))
                y2 = int(min(ymax, idx_tmp[1][0]+block+1))
                z2 = int(min(zmax, idx_tmp[2][0]+block+1))
                #print(x1,x2, y1,y2, z1,z2)
                mask[x1:x2, y1:y2, z1:z2] = 1 # block around the candidate in the mask

        return output_conv, output_grad
    ##############################################################









    #######################################################################
    def find_shocks_simple(self, conv, grad, conv_threshold, grad_threshold,
                        quiet=False, gtgrho=None, mach=None, mach_min=1.0,
                        vmag=None, vmag_min=None, ID=''):
        '''
        Function to flag shock candidate cells. Outputs a list of
        coordinates to be used by extract function. This differs
        from the non-simple version in that it just uses the numpy
        where function, and therefore can't block cells around 
        candidates. It's much faster than find_shocks. 
        
        Parameters
        ----------
        conv : 3d array
            array of convergences to be searched
        grad : 3d array
            array of gradient magnitudes to be
            searched. shape(grad) must equal shape(conv).
        conv_threshold : float
            find_shocks searches for cells with
            convergence values above this threshold
        threshold_grad : float
            find_shocks searches for cells with
            gradient magnitudes above this threshold
        Ncells : int
            max number of cells to flag, find_shocks stops
            searching beyond this number
        block : int
            blocking length around shock candidate
        
        Returns
        -------
        shocklist : list
            list of locations of shock candidates [list of 3-tuples]
        '''

        if not quiet:
            print ('*{0:s} Searching down to convergence of {1:.3f}. Then for gradients down to {2:.3f}.'.format(ID, conv_threshold,grad_threshold))
        if gtgrho is not None:
            mask = (gtgrho > 0.0) & (conv > conv_threshold) & (mach > mach_min)
            if vmag is not None and vmag_min is not None:
                mask &= (vmag > vmag_min)
            output_gtgrho = np.where(mask)

        output_conv = np.where(conv > conv_threshold)
        output_grad = np.where((grad > grad_threshold) & (conv > 0) & (conv <= conv_threshold))
        if gtgrho is not None:
            return [output_gtgrho]
        else:
            return [output_conv]
    ##############################################################








    ##############################################################
    def flux_capacitor(self, lines, gam, shock_strength=1.2, shock_size=4.):
        '''
        Function that takes the fluid variables through a shock front and
        searches for MHD shocks in it.
        
        Parameters
        ----------
        lines : dict
            dictionary of fluid variables with keys:
            
                'line' : 1d array
                    line indices centred on convergence peak
                'rho' : 1d array
                    density (g/cm^3)
                'pressure' : 1d array
                    pressure (g cm/s^2)
                'bp' : 1d array
                    magnetic field in direction of shock 
                    propagation (gauss)
                'bt1' : 1d array
                    magnetic field tranverse to the shock 
                    propagation (gauss)
                'bt2' : 1d array
                    other magnetic field tranverse to the shock 
                    propagation (gauss)
                'vp' : 1d array
                    velocity in direction of shock propagation (cm/s)
                'vt' : 1d array
                    velocity transverse to the shock propagation 
                    direction (cm/s)
                'conv' : 1d array
                    convergence (units don't matter)
                
        gam : float
            adiabatic index of fluid
        shock_strength : float
            sets the density contrast threshold across a shock. i.e. if
            rho2 is not shock_strength larger than rho1, it won't be 
            checked
        shock_size : float
            roughly how many pixels the shock is numerically spread out
            by. this makes sure we don't count pairs of states too 
            close to each other. it also defines how close to the 
            convergence peak we search for shocks
        
        Returns
        -------
        result : dict
            dictionary containing the following entries:
            
                'family' : str
                    12, 34, or if no shock is found 0
                'vs' : float
                    shock velocity (km/s)
                'vA' : float
                    preshock Alfven velocity
                'MachAlf' : float
                    preshock Alfvenic Mach number
                'rho0' : float
                    preshock mass density (g/cm^3)
                'r' : float
                    compression ratio (rho_pst/rho_pre)
                'B0' : float
                    preshock magnetic field strength
                'pmag_ratio' : float
                    ratio of postshock to preshock magnetic pressure
                'centre' : float
                    tells you if the profile is centred on the 
                    convergence peak:
        
                        0: line is not centred on the convergence peak, or the
                        centre is the first or last entry
                        
                        1: line is centred on the convergence peak
                        
                'flag' : float
                    extra information:
                    
                        0: nothing is wrong
                        
                        1: the density threshold is not met
                        
                        2: convergence peak is too close to the edge
                        
                        3: inconsistent field and mach number criteria

                        4: no positive convergence found. Why is this even here? 
        '''
        
        line = lines['line']
        up   = lines['vp']
        rho  = lines['rho']
        p    = lines['pressure']
        bp   = lines['bp']
        bt1  = lines['bt1']
        bt2  = lines['bt2']
        ut   = lines['vt']
        conv = lines['conv']
    
        result = {} # This will be the dictionary returned
        
        p_mag = (bp**2. + bt1**2. + bt2**2.)/8./pi
        B_mag = npsqrt(bp**2. + bt1**2. + bt2**2.)
        
        lw = 0.5 # linewidth for plots
        
        conv_pos = conv[where(conv > 0)]
        if len(conv_pos)==0:
            result['family']     = 0.
            result['vs']         = 0.
            result['r']          = 0.
            result['MachAlf']    = 0.
            result['Mach']       = 0.
            result['pmag_ratio'] = 0.
            result['vA']         = 0.
            result['rho0']       = 0. 
            result['B0']         = 0.
            result['flag']       = 4
            result['centre']     = 0
            return result
            #print("conv_pos has no >0",conv)
            #quit()
        conv_pos_avg = fsum(conv_pos.flatten())/len(conv_pos.flatten())
        idx_conv_pos = where(conv > conv_pos_avg)[0]
        peak_size    = int(shock_size)
        idx_search   = []
        for idx in idx_conv_pos:
            for i in range(peak_size+1):
                idx_search.append(idx-i)
                idx_search.append(idx+i)

        idx_search = unique(idx_search)
        
        centre_init = where(line == 0)[0][0]
        #print(centre_init, conv, line)
        
        if (centre_init == 0) or (centre_init == len(up)-1):
            result['family']     = 0.
            result['vs']         = 0.
            result['r']          = 0.
            result['MachAlf']    = 0.
            result['Mach']       = 0.
            result['pmag_ratio'] = 0.
            result['vA']         = 0.
            result['rho0']       = 0. 
            result['B0']         = 0.
            result['flag']       = 2
            result['centre']     = 0
            return result
        

        # Find the convergence peak within a shock_size of indices away. 
        # We do it by local walking, so as not to go down a valley to
        # another shock. HAVE TO PROTECT AGAINST INDICES GOING OUT OF BOUNDS
        #try:
        centre_tmp = centre_init
        if (centre_init==0 or centre_init==len(conv)):
            for i in range(1,int(shock_size)+1):
                
                if conv[centre_tmp-1] > conv[centre_tmp+1]:
                    step = -1
                elif conv[centre_tmp-1] < conv[centre_tmp+1]:
                    step = 1
                else:
                    step = 0
                if (centre_init+step <= 0) or (centre_init+step >= len(conv)):
                    shock_peak = centre_tmp
                    break
                if conv[centre_init+step] > conv[centre_init]:
                    centre_tmp = centre_tmp+step
        
        shock_peak = centre_tmp

        if shock_peak == centre_init:        
            result['centre'] = 1
        else:
            result['centre'] = 0
        
        
        state1 = arange(max(shock_peak-int(shock_size),0),shock_peak)
        state2 = arange(min(shock_peak+1,len(up)),min(shock_peak+int(shock_size)+1,len(up)))

        rho1 = average(rho[state1])
        rho2 = average(rho[state2])

        if rho1 > shock_strength * rho2:
            state_pre, state_pst = state2, state1
            rho_pre, rho_pst     = rho2, rho1
        elif shock_strength * rho1 < rho2:
            state_pre, state_pst = state1, state2
            rho_pre, rho_pst     = rho1, rho2
        else:
            result['family']     = 0.
            result['vs']         = 0.
            result['r']          = rho1/rho2
            result['MachAlf']    = 0.
            result['Mach']       = 0.
            result['pmag_ratio'] = average(p_mag[state2])/average(p_mag[state1])
            result['flag']       = 1
            result['vA']         = average(npsqrt(bp[state2]**2. + bt1[state2]**2. + bt2[state2]**2.))/sqrt(4*pi*rho2)
            result['rho0']       = rho2
            result['B0']         = average(B_mag[state1])
            return result
        
        
        result['r']    = rho_pst / rho_pre
        result['rho0'] = rho_pre

        u_pre = average(up[state_pre])
        u_pst = average(up[state_pst])
        vs = (u_pre - u_pst)/(1. - rho_pre/rho_pst)
        result['vs'] = abs(vs)

        csound = sqrt(gam * average(p[state_pre])/rho_pre)
        result['Mach'] = abs(vs) / csound

        if self.hydro_only:
            result['family']     = 1    # sonic shock (no B classification)
            result['vA']         = 0.
            result['MachAlf']    = 0.
            result['B0']         = 0.
            result['pmag_ratio'] = 0.
            result['flag']       = 0
        else:
            result['B0'] = average(B_mag[state_pre])

            p_mag_pre = average(p_mag[state_pre])
            p_mag_pst = average(p_mag[state_pst])

            if p_mag_pst / p_mag_pre > 1:
                result['family'] = 12
            elif p_mag_pst / p_mag_pre < 1:
                result['family'] = 34

            result['pmag_ratio'] = p_mag_pst / p_mag_pre

            b_pre  = average(npsqrt(bp[state_pre]**2. + bt1[state_pre]**2. + bt2[state_pre]**2.))
            vA_pre = b_pre / sqrt(4 * pi * rho_pre)
            result['vA']      = vA_pre
            result['MachAlf'] = abs(vs) / vA_pre if vA_pre > 0 else 0.

            if (result['family'] == 12 and result['MachAlf'] <= 1):
                result['flag'] = 3
            elif (result['family'] == 34 and result['MachAlf'] > 1):
                result['flag'] = 3
            else:
                result['flag'] = 0
        
        #if (result['family']==12 and result['MachAlf'] <= 1):
        #    result['flag'] = 3
        #elif (result['family']==34 and result['MachAlf'] > 1):
        #    result['flag'] = 3
        #else:
        #    result['flag'] = 0

        return result
    ##############################################################












    ###################################################################
    def characterise_shocks_para(self, shock_idx, rho, pres, field, vel, div,
                                grad, extra, quiet=True, ID='', ncpus = 8,
                                full = False):
        from multiprocessing import Pool
        from itertools import repeat
        #from utils.utils import set_axes_equal
        def chunks(l, n):
            """Yield n number of striped chunks from l."""
            for i in range(0, n):
                yield l[i::n]
        def chunks_swq(l, n):
            """Yield n number of sequential chunks from l."""
            d, r = divmod(len(l), n)
            for i in range(n):
                si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
                yield l[si:si+(d+1 if i < r else d)]
        NperDim=len(rho)
        buffer = max(extra["Rcylinder"], extra["Rgrad"],extra["line_range"])
        
        shock_idxes = list(chunks_swq(shock_idx, ncpus))
        if not full:
            starting_indexes = []
            rho_l, pres_l, field_l, vel_l, div_l, grad_l = [],[],[],[],[],[]
            field_x,field_y,field_z = [], [], []
            vel_x,  vel_y,  vel_z   = [], [], []
            grad_x, grad_y, grad_z  = [], [], []
            
            
            # fig0=plt.figure(16,figsize=(8,8))
            # ax0=fig0.add_subplot(1, 1, 1, projection = '3d')
            
            # fig=plt.figure(1)#figsize=(8,8))
            vmin=(rho.flatten()).min()
            vmax=(rho.flatten()).max()
            
            j=1
            new_chunks = []
            for chunck in shock_idxes:#chunks_swq(shock_idx, ncpus):        
                # ax=fig.add_subplot(2,ncpus,j)
                # axx=fig.add_subplot(2,ncpus,j+ncpus)

                xs,ys,zs=[],[],[]
                tmp=[]
                for s in chunck:
                    xs.append(s[0])
                    ys.append(s[1])
                    zs.append(s[2])
                    #tmp+=[s[0],s[1],s[2]]
                xm,xM = min(xs),max(xs)
                ym,yM = min(ys),max(ys)
                zm,zM = min(zs),max(zs)
                
                xm=max(xm-buffer,0)
                ym=max(ym-buffer,0)
                zm=max(zm-buffer,0)
                xM=min(xM+buffer,NperDim )
                yM=min(yM+buffer,NperDim )
                zM=min(zM+buffer,NperDim )
                rho_l.append(rho  [xm:xM,ym:yM,zm:zM])
                pres_l.append(pres[xm:xM,ym:yM,zm:zM])
                
                field_x.append((field[0]) [xm:xM,ym:yM,zm:zM])
                field_y.append((field[1]) [xm:xM,ym:yM,zm:zM])
                field_z.append((field[2]) [xm:xM,ym:yM,zm:zM])
                vel_x.  append(  (vel[0]) [xm:xM,ym:yM,zm:zM])
                vel_y.  append(  (vel[1]) [xm:xM,ym:yM,zm:zM])
                vel_z.  append(  (vel[2]) [xm:xM,ym:yM,zm:zM])
                div_l.  append(  div      [xm:xM,ym:yM,zm:zM])
                grad_x. append(  (grad[0])[xm:xM,ym:yM,zm:zM])
                grad_y. append(  (grad[1])[xm:xM,ym:yM,zm:zM])
                grad_z. append(  (grad[2])[xm:xM,ym:yM,zm:zM])
                
                
                for s in chunck:
                    tmp+=[(s[0]-xm,s[1]-ym,s[2]-zm)]
                new_chunks.append(tmp)
                starting_indexes.append([xm,ym,zm])
                
                # rho_cp = rho.copy()

                # extents=[xm - 0.5,xM + 0.5, ym - 0.5,yM + 0.5]
                #print(extents)
                
                # ax.imshow(rho_l[j-1][:,:, abs(zm-zM)//2].T, 
                #         origin='upper',
                #         extent=extents,
                #         vmin=vmin, vmax=vmax,
                #         #aspect="auto"
                #         )#, alpha=1.0/j)
                # extents=[xm-0.5,xM+0.5, zm-0.5,zM+0.5]
                # axx.imshow(rho_l[j-1][:,abs(ym-yM)//2,:].T, 
                #         origin='upper',
                #         extent=extents,
                #         vmin=vmin, vmax=vmax,
                #         #aspect="auto"
                #         )#, alpha=1.0/j)
                # if j==1:
                #     ax.set_ylabel(r"$y-{\rm axis}$")
                #     axx.set_ylabel(r"$z-{\rm axis}$")
                #     ax.set_xlabel(r"$x-{\rm axis}$")
                #     axx.set_xlabel(r"$x-{\rm axis}$")
                # ax.set_title("cpu #%d"%j)
                # ax.axes.get_xaxis().set_ticks([])
                # ax.axes.get_yaxis().set_ticks([])
                # axx.axes.get_xaxis().set_ticks([])
                # axx.axes.get_yaxis().set_ticks([])
                # ax0.scatter(xs,ys,zs, alpha=0.5, s=5, label=r"cpu #%d"%j+r" $n_{\rm cells} = %i$"%(len(chunck)))  
                j+=1    
            
            # ax0.legend(loc = 'lower right')#,bbox_to_anchor=(1.5,0.0))
            # fig0.suptitle( "Shocks in cpus")
            # fig0.savefig(  "shocks_cpus")
            # fig.suptitle(  "Domain decomposition")
            # fig.savefig(   "Domain_dec")#canvas.draw()
            for cpu in range(0,ncpus):
                field_l.append([field_x[cpu], field_y[cpu],field_z[cpu]])
                vel_l.append(  [vel_x  [cpu], vel_y  [cpu],vel_z  [cpu]])
                grad_l.append( [grad_x [cpu], grad_y [cpu],grad_z [cpu]])
        
        with Pool(processes=ncpus) as p:#, callback=out) as p:
                results=p.starmap_async(
                            self.characterise_shocks,
                            zip( shock_idxes  if full else new_chunks,
                                repeat(rho)   if full else rho_l,
                                repeat(pres)  if full else pres_l,
                                repeat(field) if full else field_l,
                                repeat(vel)   if full else vel_l,
                                repeat(div)   if full else div_l,
                                repeat(grad)  if full else grad_l,
                                repeat(extra),
                                repeat(quiet),
                                repeat(ID)
                                )
                                            )
                result=results.get()###NEEDED TO OBTAIN THE RESULTS IN ASYNC
                p.close()

        fullres=[]    
        
        for res_idx in range(0,len(result[0][0])): 
            tmp=[]
            for cpu_id, cpures in enumerate(result):
                #print(cpu_id)
                data, header = cpures
                if full:
                    to_add = data[res_idx]
                else:
                    to_add = np.array(data[res_idx])
                    
                
                    if res_idx==0:
                        to_add+=starting_indexes[cpu_id][0]
                    if res_idx==1:
                        to_add+=starting_indexes[cpu_id][1]
                    if res_idx==2:
                        to_add+=starting_indexes[cpu_id][2]
                tmp.extend(list(to_add))
            fullres.append(np.array(tmp))

        return   fullres, result[0]
        
    ###################################################################





    ##############################################################
    def characterise_shocks(self, shock_idx, rho, pres, field, vel, div, grad, extra, quiet=False, ID=''):
        '''
        Characterise the shocks at a search list of coordinates.
        
        Parameters
        ----------
        shock_idx : list
            Shock search list of three-tuples (coordinates).
        rho : (N, N, N) array_like
            3d array of mass density (g/cm^3).
        pres : (N, N, N) array_like
            3d array of pressure.
        field : (3, N, N, N) array_like
            three 3d arrays of magnetic field (G), x, y, and z components in order.
        vel : (3, N, N, N) array_like
            three 3d arrays of velocity (cm/s), x, y, and z components in order.
        div : (N, N, N) array_like
            3d array of divergence (dimensionless).
        grad : (N, N, N) array_like
            3d array of divergence (dimensionless).
        extra : dict
            dictionary of extra parameters. Requires:
            * method_norm : method for shock direction
            * periodic : list of 3-bools for periodic directions
            * Rgrad : size of averaging sphere
            * Rcylinder : size of averaging cylinder
            * gamma : adiabatic index
            * line_range : half the size of shock line
            * method_plane : method for transverse direction
            * field_ref : pixel location for transverse direction calculation
            * shock_ratio : lower limit of density ratio
            * offset : offset correction for x-direction of datacubes
        quiet : bool
            Whether or not to print commentary. The default value is False.
        ID : str
            Part of the printout in the commentary.

        Returns
        -------
        characterise_list : list
            16xN list of various shock variables.
        headers : list
            length 16 list of headers.

        '''
        
        #print(len(grad[2]), len(rho),len(grad[2]) )
        #quit()
        method_norm = extra['method_norm']
        periodic    = extra['periodic']
        Rgrad       = extra['Rgrad']
        Rcylinder   = extra['Rcylinder']
        gam         = extra['gamma']
        line_range  = extra['line_range']
        field_ref   = extra['field_ref']
        method_plane= extra['method_plane']
        shock_ratio = extra['shock_ratio']
        offset      = extra['offset']
        
        xmax        = np.shape(grad[0])[0]
        ymax        = np.shape(grad[0])[1]
        zmax        = np.shape(grad[0])[2]

        line        = np.arange(-line_range,line_range+1,1.) # array of indices along the shock line

        n_idx = 0
        if len(shock_idx)==0:
            print('* ' + ID + ' no cells to search, len(shock_idx):', len(shock_idx))
            raise Exception("probkem!!!!")
            quit()
        ##########################################################

        shock_id        =np.empty(len(shock_idx), dtype='int')
        loc_x           =np.empty(len(shock_idx), dtype='int')
        loc_y           =np.empty(len(shock_idx), dtype='int')
        loc_z           =np.empty(len(shock_idx), dtype='int')
        dir_x           =np.empty(len(shock_idx), dtype='float')
        dir_y           =np.empty(len(shock_idx), dtype='float')
        dir_z           =np.empty(len(shock_idx), dtype='float')
        shock_families  =np.empty(len(shock_idx), dtype='int')
        shock_speeds    =np.empty(len(shock_idx), dtype='float')
        vA_pre          =np.empty(len(shock_idx), dtype='float')
        Mach            =np.empty(len(shock_idx), dtype='float')
        MachAlf         =np.empty(len(shock_idx), dtype='float')
        density_contrast=np.empty(len(shock_idx), dtype='float')
        rho0            =np.empty(len(shock_idx), dtype='float')
        B0              =np.empty(len(shock_idx), dtype='float')
        pmag_ratio      =np.empty(len(shock_idx), dtype='float')
        peak_flag       =np.empty(len(shock_idx), dtype='int')
        flag            =np.empty(len(shock_idx), dtype='int')

    
        
        if method_norm == 'point_gradient':
            cube_normal = {
                'gradx': grad[0],
                'grady': grad[1],
                'gradz': grad[2]
            }
        elif method_norm == 'average_gradient':
            cube_normal = {
                'gradx': grad[0],
                'grady': grad[1],
                'gradz': grad[2],
                'radius': Rgrad,
                'weight': npsqrt(grad[0]**2. + grad[1]**2. + grad[2]**2.),
                'xmax': np.shape(grad[0])[0],
                'ymax': np.shape(grad[0])[1],
                'zmax': np.shape(grad[0])[2],
                'periodic_x': periodic[0],
                'periodic_y': periodic[1],
                'periodic_z': periodic[2]
            }
        else:
            raise Exception('Norm definition \'' + method_norm + '\' doesn\'t exist. Check config file and documentation for function \'search\'.')

        
        

        for loopnum,index in enumerate(shock_idx):
            idx = (index[0]-offset[0],index[1]-offset[1],index[2]-offset[2])
            #shockID = index[3]
            waypoint='start' # Start the waypoint system
            
            t1_local = time.time()
            n_idx+=1

            shock_name = '_%.3d' %int(idx[0]) +'_%.3d' %int(idx[1]) +'_%.3d' %int(idx[2])


            
            #try:
            ###############################
            # Compute the shock normal
            waypoint='norm'
            cube_normal['idx'] = idx            
                    
            ### Shock normal vector: n_shock = (nx,ny,nz)
            n_shock = self.shock_normal(cube_normal,method=method_norm)

            if n_shock == 'error':
                raise Exception('n_shock == error')
                #continue
            else:
                nx, ny, nz = n_shock[0], n_shock[1], n_shock[2]
                
            ###############################

            
            
            
            
            
            
            ###############################
            # Define the indices along the shock line
            waypoint='line'
            
            sx,sy,sz = [],[],[]
            
            plotline = np.array([])
            for l in line:
                if (round(idx[0] + l*nx,0) >= 0) and (round(idx[1] + l*ny,0) >= 0) and (round(idx[2] + l*nz,0) >= 0) and (round(idx[0] + l*nx,0) < xmax) and (round(idx[1] + l*ny,0) < ymax) and (round(idx[2] + l*nz,0) < zmax):
                    sx.append(int(round(idx[0] + l*nx,0)))
                    sy.append(int(round(idx[1] + l*ny,0)))
                    sz.append(int(round(idx[2] + l*nz,0)))
                    plotline = np.append(plotline,l)

            #print(plotline, "jjjjj")
            sx,sy,sz = np.array(sx),np.array(sy),np.array(sz)
            ###############################
            
            
            
            


            
            ###############################
            # Define the transverse vectors
            waypoint='ref_field'
            idx_ref = (
                int(sx[int(len(sx)/2.)]),
                int(sy[int(len(sx)/2.)]),
                int(sz[int(len(sx)/2.)])
            )
            idx_ref2 = np.where(np.array(plotline)==0)[0] - field_ref
            if idx_ref2 < 0:
                idx_ref2 = np.where(np.array(plotline)==0)[0] + field_ref
            if idx_ref2 > len(plotline)-1:
                raise Exception('Reference location out of bounds. Try a smaller number for \'ref\' in the config file.')
                
            waypoint='trans1'

            if self.hydro_only:
                nt = self.transerve_direction(data={'ns': n_shock}, method='hydro')
            elif method_plane == 'point_field':
                bxline = np.array([field[0][x,y,z] for x,y,z in zip(sx,sy,sz)])
                byline = np.array([field[1][x,y,z] for x,y,z in zip(sx,sy,sz)])
                bzline = np.array([field[2][x,y,z] for x,y,z in zip(sx,sy,sz)])

                data = {
                    'ref': idx_ref2,
                    'bx': bxline,
                    'by': byline,
                    'bz': bzline,
                    'ns': n_shock
                }
                nt = self.transerve_direction(data=data, method=method_plane)
            elif method_plane == 'average_field':
                bxline = np.array([field[0][x,y,z] for x,y,z in zip(sx,sy,sz)])
                byline = np.array([field[1][x,y,z] for x,y,z in zip(sx,sy,sz)])
                bzline = np.array([field[2][x,y,z] for x,y,z in zip(sx,sy,sz)])

                region = np.arange(idx_ref2[0],np.where(np.array(plotline)==0)[0])

                data = {
                    'region': region,
                    'bx': bxline,
                    'by': byline,
                    'bz': bzline,
                    'ns': n_shock
                }
                nt = self.transerve_direction(data=data, method=method_plane)
            else:
                raise Exception('Transverse definition \'' + method_plane + '\' doesn\'t exist. Check config file and documentation for function \'transerve_direction\'.')

            waypoint='trans2'

            nt1x, nt1y, nt1z = nt[0]
            nt2x, nt2y, nt2z = nt[1]
            ###############################          
            
            
            

            ###############################
            # Compute profiles by averaging through cylinder
            waypoint='cylinder'
            #print(sx)
            _bx = None if self.hydro_only else field[0]
            _by = None if self.hydro_only else field[1]
            _bz = None if self.hydro_only else field[2]
            rho_line, p_line, up_line, ut1_line, ut2_line, bp_line, bt1_line, bt2_line, conv_line = self.cylinder(Rcylinder, (sx, sy, sz), n_shock, nt[0], nt[1], rho, pres, vel[0], vel[1], vel[2], div, _bx, _by, _bz)
            ###############################
            #res=[rho_line, p_line, up_line, ut1_line, ut2_line, bp_line, bt1_line, bt2_line, conv_line]
            #if any([resu is [] for resu in res] ): exit("error. empty array.")
            #print("after cylinder",rho_line, p_line, up_line, ut1_line, ut2_line, bp_line, bt1_line, bt2_line, conv_line)

            #quit()




            ###############################
            # Characterise the line profiles, e.g. detect shock (or not)
            waypoint='detect'

            lines = {
                'line': plotline,
                'rho': rho_line,
                'pressure': p_line,
                'bp': bp_line,
                'bt1': bt1_line,
                'bt2': bt2_line,
                'vp': up_line,
                'vt': ut1_line,
                'conv': conv_line
            }

            result = self.flux_capacitor(lines, gam=gam, shock_strength=shock_ratio, shock_size=3.)
            ###############################







            ###############################
            # Save shock stats to table
            waypoint='table'

            # Add results to columns
            #shock_id[loopnum]=shockID
            loc_x[loopnum]              =idx[0]+offset[0]
            loc_y[loopnum]              =idx[1]+offset[1]
            loc_z[loopnum]              =idx[2]+offset[2]
            dir_x[loopnum]              =round(nx,3)
            dir_y[loopnum]              =round(ny,3)
            dir_z[loopnum]              =round(nz,3)
            shock_families[loopnum]     =result['family']
            shock_speeds[loopnum]       =result['vs']
            vA_pre[loopnum]             =result['vA']
            MachAlf[loopnum]            =result['MachAlf']
            Mach[loopnum]               =result['Mach']
            density_contrast[loopnum]   =result['r']
            rho0[loopnum]               =result['rho0']
            B0[loopnum]                 =result['B0']
            pmag_ratio[loopnum]         =result['pmag_ratio']
            peak_flag[loopnum]          =result['centre']
            flag[loopnum]               =result['flag']
            
            ###############################



            t2_local = time.time()
            
            if result['family'] == 12:
                shocktype = 'FAST-shock'
            elif result['family'] == 34:
                shocktype = 'SLOW-shock'
            elif result['family'] == 0:
                shocktype = '???-shock'
            
            if not quiet:
                print('{0:s} ({1:d}/{2:d}) {3:s} at ({4:03d}, {5:03d}, {6:03d}) in {7:.2f} s'.format(ID, n_idx, len(shock_idx), shocktype, int(idx[0]+offset[0]), int(idx[1]+offset[1]), int(idx[2]+offset[2]), t2_local - t1_local))
            
                
                
        #    except Exception as e: 
        #    
        #        #shock_id[loopnum]=shockID
        #        loc_x[loopnum]=idx[0]+offset[0]
        #        loc_y[loopnum]=idx[1]+offset[1]
        #        loc_z[loopnum]=idx[2]+offset[2]
        #        dir_x[loopnum]=np.nan
        #        dir_y[loopnum]=np.nan
        #        dir_z[loopnum]=np.nan
        #        shock_families[loopnum]=-1
        #        shock_speeds[loopnum]=np.nan
        #        vA_pre[loopnum]=np.nan
        #        density_contrast[loopnum]=np.nan
        #        rho0[loopnum]=np.nan
        #        B0[loopnum]=np.nan
        #        pmag_ratio[loopnum]=np.nan
        #        peak_flag[loopnum]=-1
        #        flag[loopnum]=-1
        #        
        #        t2_local = time.time()
        #        
        #        if not quiet:
        #            print('*{0:s} ({1:d}/{2:d}) FAILURE at (x={3:03d}, y={4:03d}, z={5:03d}) in {6:.2f} s. Waypoint: {7:s}; Error: {8:s}'.format(ID, n_idx, len(shock_idx), int(idx[0]+offset[0]), int(idx[1]+offset[1]), int(idx[2]+offset[2]), t2_local - t1_local, waypoint, str(e)))
        #        
        #        

        output = [
            loc_x,
            loc_y,
            loc_z,
            dir_x,
            dir_y,
            dir_z,
            shock_families,
            shock_speeds,
            vA_pre,
            MachAlf,
            Mach,
            density_contrast,
            rho0,
            B0,
            pmag_ratio,
            peak_flag,
            flag
        ]
        
        headers = [
            'x',
            'y',
            'z',
            'nx',
            'ny',
            'nz',
            'Family',
            'vs',
            'vA',
            'MachAlf',
            'Mach'
            'r',
            'rho0',
            'B0',
            'pmag_ratio',
            'peak',
            'FLAG'
        ]


        
        return output, headers
        ##########################################################    
