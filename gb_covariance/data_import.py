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
    date = datetime.datetime.now()
    timestamp = (str(date.year)[-2:]+ str(date.month).rjust(2,'0')+  
                 str(date.day).rjust(2,'0') 
                 + '-' + str(date.hour).rjust(2,'0') + 
                 str(date.minute).rjust(2,'0'))
    return timestamp

def data_stats(df_full, df, crystal_type, stats):
    """add additional data statistics
    """
    all_models = df_full.model.drop_duplicates()
    used_models = df.model.drop_duplicates()
    n_models_dropped_total = len(all_models) - len(used_models)
    n_model_species_original = len(df_full[['model','species']].drop_duplicates())
    n_model_species = len(df[['model','species']].drop_duplicates())
    n_gb_data_used = len(df)


    stats_new = [["Crystal type selected", crystal_type],
             ["Number of unique model + species combos", n_model_species],
             ["Number of GB data points used", n_gb_data_used],
             ["Number of unique models total", len(all_models)],
             ["Number of models used", len(used_models)],
             ["Number of models dropped", n_models_dropped_total]
             ]
    
    stats.extend(stats_new)
    df_stats = pd.DataFrame(stats)

    return df_stats

def get_analytical():
    """import lattice matching curves, replace inf with 0
    """
    df_analytical = pd.read_csv('./data/df_analytical.csv',index_col=0)
    df_analytical['grain_energy_analytical'] = [i.replace("-inf","0") for i in df_analytical['grain_energy_analytical']]
    df_analytical['grain_energy_analytical'] = [i.replace("inf","0") for i in df_analytical['grain_energy_analytical']]
    return df_analytical

def main():
    #list of properties to get (gather all, can remove from saved df if needed)
    openkim_props = ['lattice_const', 'bulk_modulus', 'elastic_const', 
                        'cohesive_energy', 'thermal_expansion_coeff', 'surface_energy',
                        'extr_stack_fault_energy','intr_stack_fault_energy',
                        'unstable_stack_energy','unstable_twinning_energy',
                        'monovacancy_relaxed_formation_potential_energy',
                        'monovacancy_unrelaxed_formation_potential_energy',
                        'monovacancy_vacancy_migration_energy',
                        'monovacancy_relaxation_volume'] 
    crystal_type_setting = 'fcc'
    use_model_props = False # option to use individual model props (not complete)
    tolerance = 1e-8    #1e-8 was original, 1e-16 in some scripts
    order = 8           #8 was original, 32 is in some test scripts

    stats = []      # for data statistics

    df_merge_raw = openKimInterface.get_merged_df(openkim_props)
    df_merge_all = df_merge_raw.copy()
    stats.append(['Total number of GB data points', len(df_merge_all)])

    # only use FCC
    n_pre = len(df_merge_all)
    df_merge_all = df_merge_all[df_merge_all['crystal_type'] == crystal_type_setting].reset_index(drop=True)
    stats.append(['Number filtered by crystal type', n_pre - len(df_merge_all)])

    # remove outliers
    n_pre = len(df_merge_all)
    df_merge_all = openKimInterface.remove_gb_outliers(df_merge_all, [-0.1, 100])
    stats.append(['Number filtered by outlier limits', n_pre - len(df_merge_all)])

    # remove pair potentials
    n_pre = len(df_merge_all)
    df_merge_all = df_merge_all[~df_merge_all.model.str.contains("Morse_Shifted")]
    df_merge_all = df_merge_all[~df_merge_all.model.str.contains("MJ_Morris")]
    stats.append(['Number of pair potential points removed', n_pre - len(df_merge_all)])

    # add analytical
    df_analytical = get_analytical()
    df_analytical = df_analytical.drop('angle',axis=1)
    df_merge_all = df_merge_all.reset_index().merge(df_analytical,how='outer',on=['species','tilt_axis','crystal_type']).set_index('index')


    df_merge_all = models.get_analytical_coeff(df_merge_all)

    #get rid of diamond for now
    df_merge_all = df_merge_all.drop([i for i in df_merge_all.columns if 'diamond' in i],axis=1)

    #add prediction using average md properties
    df_md_avg = openKimInterface.get_md_avg(df_merge_all)

    if 'grain_energy_analytical_unrelaxed' in df_analytical.columns:
        if use_model_props == False:
            if 'grain_energy_analytical' in df_md_avg.columns:
                df_md_avg = df_md_avg.drop(['grain_energy_analytical','grain_energy_analytical_unrelaxed'],axis=1)
            df_md_avg = pd.merge(df_md_avg, 
                                df_merge_all[['species','tilt_axis','crystal_type','grain_energy_analytical','grain_energy_analytical_unrelaxed']].drop_duplicates(subset=['species','tilt_axis','crystal_type']),
                                how = 'left', 
                                on = ['species','tilt_axis','crystal_type'])
        elif use_model_props == True:
            raise Exception("not setup to run relaxed analytical on md_avg properties")    
    else:
        df_md_avg = models.add_analytical(df_md_avg, 
                                        tolerance = tolerance,
                                        use_model_props = use_model_props) 



    df_merge = df_merge_all.copy()

    # remove tilt_axis specific items
    n_pre = len(df_merge)
    df_merge = df_merge.drop(['basis_atom_coords','grain_energy','angle','tilt_axis','units','testnames',
                            'grain_energy_analytical','grain_energy_analytical_unrelaxed'],axis=1)
    df_merge = df_merge.drop_duplicates().reset_index(drop=True)
    stats.append(['Number of duplicate tilt axes removed', n_pre - len(df_merge)])

    df_stats = data_stats(df_merge_raw, df_merge, crystal_type_setting, stats)

    #save
    df_merge.to_csv('./data/df_merge.csv')
    df_merge_all.to_csv('./data/df_merge_all.csv')
    df_merge_raw.to_csv('./data/df_merge_raw.csv')
    df_md_avg.to_csv('./data/df_md_avg.csv')
    df_stats.to_csv('./data/stats.csv')

    end = time()
    print(f"time [sec]:", end-start)

if __name__ == "__main__":
    main()