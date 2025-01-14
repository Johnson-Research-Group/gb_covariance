import openKimInterface, models
import numpy as np
import pandas as pd
import datetime
import os
import multiprocess
import wield
from time import time

start = time()

def folder_create(timestamp, master_directory):
    """create folder with timestamp
    """
    new_directory = os.path.join(master_directory,timestamp)
    os.mkdir(new_directory)
    print("directory created: ",new_directory)
    return(new_directory)  


def create_timestamp():
    """generate timestamp
    """
    date = datetime.datetime.now()
    timestamp = (str(date.year)[-2:]+ str(date.month).rjust(2,'0')+  
                 str(date.day).rjust(2,'0') 
                 + '-' + str(date.hour).rjust(2,'0') + 
                 str(date.minute).rjust(2,'0'))
    return timestamp

def make_data(df_merge, 
              use_model_props):
    """create DF with possible species, tilt axis, crystal, and angle combinations
    """
    df_analytical = df_merge.copy()
    if use_model_props == False:
        df_analytical = df_analytical[['species', 'tilt_axis', 'crystal_type', 'angle']].drop_duplicates(subset=['species','tilt_axis', 'crystal_type'])
    data = df_analytical.explode('angle').values.tolist()
    return df_analytical, data

#####################################################################
# inputs
file_loc = './data/df_analytical_new.csv'
fixed_species = []              # list of species strings to limit by, e.g. ['Cu','Fe']
fixed_axis = ""                 # fix tilt axis to string e.g. "[1, 1, 0]" or "[0, 0, 1]"
crystal_type_setting = 'fcc'    # only setup for FCC right now
use_model_props = False         # deprecated option to use individual IP model properties w/ LM code
perform_relaxation = True       # wield
tolerance = 1e-8                # 1e-8 was original, 1e-16 in some scripts
order = 8                       # 8 was original, 32 is in some test scripts


#####################################################################
df_merge = openKimInterface.get_df_grain()
df_merge = df_merge[df_merge['crystal_type'] == crystal_type_setting].reset_index(drop=True)

# filter based on species and axis list
if len(fixed_species) > 0:
    df_merge = df_merge[df_merge['species'].isin(fixed_species)] 
if len(fixed_axis) > 0:
    df_merge = df_merge[df_merge['tilt_axis'] == fixed_axis] #filter temporarily for debug

df_analytical, data = make_data(df_merge, 
                                use_model_props)

def generate(d):
    species, tilt_axis, crystal_type, angle = d
    Rtheta1 = wield.createMatrixFromXAngle(angle/2)
    Rtheta2 = wield.createMatrixFromXAngle(-angle/2)
    X1,Z1 = models.get_analytical_XZ(tilt_axis)
    X2,Z2 = models.get_analytical_XZ(tilt_axis)

    labels = {'species':species,
                'tilt_axis':tilt_axis,
                'crystal_type':crystal_type,
                'angle':angle}

    C1,C2,ground,eps = models.get_analytical_ground(labels, 
                                            tolerance = tolerance, 
                                            order = order, 
                                            use_model_props = use_model_props)

    R1 = wield.createMatrixFromZX(Z1,X1)
    R2 = wield.createMatrixFromZX(Z2,X2)

    e_unrelaxed  = 1.0 - wield.SurfaceGD(C1,Rtheta1 @ R1,C2,Rtheta2 @ R2,eps,tolerance)/ground

    if perform_relaxation == True:
        def energy(r, theta):
            x = r*np.cos(np.radians(theta))
            y = r*np.sin(np.radians(theta))
            n = [x,y,np.sqrt(abs(1 - x**2 - y**2))]
            Rnormal = wield.createMatrixFromNormalVector(n).transpose()
            energy = 1.0 - wield.SurfaceGD(C1, Rnormal@Rtheta1@R1, C2, Rnormal@Rtheta2@R2, eps, tolerance)/ground
            return energy
        venergy = np.vectorize(energy)

        # replaced 201 with 203 to run: Ag (110), Au (110), Pb (112), Pt (110), Rh (110), Rh (112)
        rs, thetas = np.meshgrid(np.linspace(0,1,65), np.linspace(0,360,201))
        ws = venergy(rs,thetas)

    elif perform_relaxation == False:
        rs = []
        ws = []
        thetas = []

    return [species, tilt_axis, crystal_type, angle, rs, thetas, e_unrelaxed, ws]

pool = multiprocess.Pool()
data = pool.map(generate,data)
pool.close()


def generate(d):
    species, tilt_axis, crystal_type, angle, rs, thetas, e_unrelaxed, ws = d
    if perform_relaxation == True:
        xs = rs * np.cos(np.radians(thetas))
        ys = rs * np.sin(np.radians(thetas))
        zs = np.sqrt(abs(1 - xs**2 - ys**2))

        e, [a,b,c] = wield.Convexify2D(xs,ys,zs,rs,thetas,ws,True) #True = threeFacet, False = secondFacet
    elif perform_relaxation == False:
        e = []
    
    return [species, tilt_axis, crystal_type, angle, e_unrelaxed, e]

pool = multiprocess.Pool()
out = pool.map(generate,data)
pool.close()


df_analytical = pd.DataFrame(out, columns = ['species',
                                             'tilt_axis',
                                             'crystal_type',
                                             'angle',
                                             'grain_energy_analytical_unrelaxed',
                                             'grain_energy_analytical'])
df_analytical = df_analytical.groupby(['species',
                                        'tilt_axis',
                                        'crystal_type']).agg(list).reset_index()

if perform_relaxation == False:
    df_analytical = df_analytical.drop("grain_energy_analytical",axis=1)

#save
df_analytical.to_csv(file_loc,
                     mode = 'a')

end = time()
print(f"time [sec]:", end-start)