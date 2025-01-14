from ast import literal_eval
from textwrap import wrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import container
from matplotlib.backends.backend_pdf import PdfPages
# import plotting
import seaborn as sns
# from itertools import combinations
from uncertainty_quantification_nested_cv import cv_lm
from data_exploration import import_label_dict
# from sklearn.impute import KNNImputer
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.compose import TransformedTargetRegressor
# from sklearn.pipeline import Pipeline
# from sklearn import linear_model, svm
# from sklearn.utils import resample
# from sklearn.metrics import r2_score
# from scipy import stats
from sklearn.metrics import mean_squared_error
from models import linear_model_create, get_analytical_coeff
from data_import import get_analytical
from model_selection import basic_outlier_removal
# import statsmodels.api as sm


def se_unit_convert(df):
    """convert J/m^2 to eV/angstrom^2
    """
    return df*6.241509e+18*1.0E-20

def get_df_dft(path = "data/dft.csv",
               add_c11_c12 = False):

    df_dft = pd.read_csv(path)
    se_list = ['surface_energy_111_fcc', 
               'surface_energy_121_fcc',
               'surface_energy_100_fcc', 
               'unstable_stack_energy_fcc',
               'intr_stack_fault_energy_fcc']
    df_dft[se_list] = se_unit_convert(df_dft[se_list])
    if add_c11_c12 == True:
        df_dft = add_elastic_consts(df_dft)
    return df_dft

def add_elastic_consts(df):
    """adds c11 and c12 elastic constants
    """
    df['c12_fcc'] = df['bulk_modulus_fcc'] - df['C11-C12']/3
    df['c11_fcc'] = df['c12_fcc'] + df['C11-C12']
    return df


def get_dft_regr_coeff(df_dft,
                       pipe,
                       model_properties,
                       rmse):
    """computes the regression coefficient, given DFT data
    """
    X_dft = df_dft[model_properties]     

    y_pred_dft = pipe.predict(X_dft)
    lower = [1.96*rmse for i in range(len(df_dft))]
    upper = lower
    df_dft_regr_coeff = pd.DataFrame({'species':df_dft.species,
                                      'regr_coeff':y_pred_dft,                                
                                      'regr_coeff_lower':lower,
                                      'regr_coeff_upper':upper})
    df_dft = df_dft.merge(df_dft_regr_coeff,on='species')
    return (df_dft)


def get_boxplot(df_clean, 
                df_dft, 
                df_dft_gb,
                readme,
                save_fig = True,
                save_loc = 'images/main',
                file_name = 'dft_w_pi',
                plot_errorbar = True,
                order_list = ["Ag","Al","Au","Cu","Ni","Pd","Pt"]):
    """plot boxplot of dft GB results
    """
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(4,3))
    sns.boxplot(data = df_clean, 
                x="species", 
                y="coeff", 
                order=order_list, 
                color = "0.8", 
                linewidth=1.0,
                fliersize=5.0,
                whis=0,
                flierprops={"marker":"."},
                zorder=1)
    ax.set_ylabel("GB scaling coefficient")

    # add dft Xs
    ax.scatter(df_dft['species'],
               df_dft['regr_coeff'], 
               marker='x', 
               s=100., 
               alpha=1.0, 
               color="r",
               label='\n'.join(wrap(r"$E_0$ regression prediction using DFT indicator properties",20)),
               zorder=3)

    df_dft_gb_plot = df_dft_gb[['species','dft_exact_coeff']].drop_duplicates()
    df_dft_gb_plot = df_dft_gb_plot[df_dft_gb_plot['species'].isin(order_list)]
    ax.scatter(df_dft_gb_plot['species'],
               df_dft_gb_plot['dft_exact_coeff'], 
               marker='<', 
               s=50., 
               alpha=0.9, 
               color="r",
               label='\n'.join(wrap(r"$E_0$ fit directly to DFT GB results",20)),
               zorder=2)

    # add errorbars if desired
    if plot_errorbar == True:
        ax.errorbar(df_dft['species'],
                    df_dft['regr_coeff'], 
                    yerr = (df_dft['regr_coeff_lower'],
                            df_dft['regr_coeff_upper']), 
                            fmt='.', 
                            markersize=0.0001, 
                            alpha=0.5, 
                            color="r",
                            #label='\n'.join(wrap("Predicted strength using DFT indicator properties",20)),
                            elinewidth=2.0,
                            capsize = 4)

    fig.legend(bbox_to_anchor = (0.05,0.9,0.85,.15),#(0.,1.02,1.,.102),
                    loc='lower left',
                    mode="expand",
                    #bbox_transform = fig.transFigure)
                    ncol = 2,
                    fontsize= 8)
    # ax.legend(new_handles, labels, fontsize=8)

    if save_fig == True:
        fig.savefig(f"{save_loc}/{file_name}.pdf", bbox_inches = 'tight')
        with open(f"{save_loc}/dft_readme.txt", "w") as out:
            out.write(readme)
    return 


def get_prop_boxplot(df_clean, 
                     prop, 
                     prop_label, 
                     dft_value, 
                     save_fig = True,
                     save_loc = "gb_covariance/data_ays/dft_props"):
    """boxplots of individual properties, IP vs. DFT.
    """
    order_list = ["Ag","Al","Au","Cu","Ni","Pd","Pt"]
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(4,3))
    g = sns.boxplot(data = df_clean, 
                x="species", 
                y=prop, 
                order=order_list, 
                color = "0.8", 
                linewidth=1.0,
                fliersize=5.0,
                whis=0,
                flierprops={"marker":"."})
    g.set_ylabel(prop_label)
    g.scatter(data = dft_value,
              x = "species",
              y = prop,
              marker = 'x',
              s=50.,
              color = "r")

    if save_fig == True:
        fig.savefig(f"{save_loc}/dft_{prop}.pdf", bbox_inches = 'tight')
    return fig


def dft_gb_energy_plots(species, 
                        tilt_axis, 
                        df_ip, 
                        df_analytical,
                        df_dft_gb,
                        df_dft_pred,
                        legend = False,
                        file_loc = 'images/dft/gb_vs_angle'):

    n_subplots = len(tilt_axis)
    n_species = len(species)
    fig_list = []
    for j in range(n_species):
        fig, ax = plt.subplots(1,
                               n_subplots, 
                               tight_layout = True, 
                               figsize = (12,4), 
                               dpi = 300)#, figsize=(4,8))#,layout='constrained')
        s = species[j]
        for i,t in enumerate(tilt_axis):
        
            # plot all DFT gb energy results
            print(s, i, t)
            if len(df_dft_gb) > 0:
                dft = df_dft_gb.copy()
                dft = dft[(dft['species'] == s) & (dft['tilt_axis'] == t)]
                if len(dft) > 0:
                    ax[i].scatter(dft['angle'].iloc[0], 
                                  dft['grain_energy'].iloc[0],
                                  marker='X',
                                  label='DFT')
        
            # plot all IM results from OpenKIM
            df1 = df_ip.copy()
            df1 = df1[df1['species'] == s]
            df1 = df1[df1['tilt_axis'] == t]
            crystal_type = df1.crystal_type.drop_duplicates().values
            print('df shape:',df1.shape)
            for a,b in zip(df1['angle'], df1['grain_energy']):
                ax[i].plot(literal_eval(a), 
                           literal_eval(b), linewidth=2, alpha = 0.10)

            #plot average coefficient times analytical using average lattice constant
            ay = df_analytical[(df_analytical['species'] == s) 
                               & (df_analytical['tilt_axis'] == t)]
            ay_angle = np.array(ay['angle'].to_list()).reshape(-1)
            gb_an = np.array(ay['grain_energy_analytical'].to_list()).reshape(-1)
            c_dft_fit = df_dft_gb[df_dft_gb['species'] == s]['dft_exact_coeff'].iloc[0]
            ax[i].plot(ay_angle, 
                       c_dft_fit*gb_an, 
                       alpha = 1, 
                       label='analytical using exact coeff.', 
                       linestyle = 'dashed',
                       linewidth=2,
                       color = 'green')
            # crystal_type = df_md_avg['crystal_type'].drop_duplicates().values

            c_dft_regr = np.array(df_dft_pred[df_dft_pred['species'] == s]['regr_coeff'].to_list())

            ax[i].plot(ay_angle, 
                       c_dft_regr*gb_an, 
                       alpha = 1, 
                       label='analytical using dft regr coeff.', 
                       linestyle = 'dashed',
                       linewidth=2,
                       color = 'orange')
        

            tilt_axis_dict = {'[0, 0, 1]':'[001]',
                              '[1, 1, 1]':'[111]',
                              '[1, 1, 0]':'[110]',
                              '[1, 1, 2]':'[112]'}    
            ax[i].set_xlabel(f"{tilt_axis_dict[t]} angle")#,fontsize = 18)
            ax[i].set_ylabel('GB energy')#,fontsize = 18)
        
        if legend == True:
            handles, labels = ax[0].get_legend_handles_labels()
            fig.legend(handles,
                       labels,
                       #bbox_to_anchor = (0.25,0.9,0.5,.15),#(0.,1.02,1.,.102),
                       loc='upper center',
                       #mode="expand",
                       ncol = len(labels),
                       fontsize= 12
                       )

        fig.savefig(f"{file_loc}/{species[j]}.pdf", 
                    bbox_inches = 'tight')

        plt.close()
        fig_list.append(fig)

    
    path_name = "images/dft/gb_vs_angle/all_data_plots.pdf"
    make_combined_pdf(fig_list,
                      path_name,
                      species = species)

    return


def make_combined_pdf(fig_list, 
                      path_name,
                      species = None):
    with PdfPages(path_name) as pdf:
        for i,fig in enumerate(fig_list):
            plt.figure(fig, bbox_inches = 'tight')
            if species != None:
                plt.title(f'{species[i]}')
            pdf.savefig(bbox_inches = 'tight')
            plt.close()

    return 


def error_plots(y, 
                y_pred,
                save_loc = "images/dft"):

    if isinstance(y,list):
        y = np.array(y)
    if isinstance(y_pred,list):
        y_pred = np.array(y_pred)
    error = y - y_pred
    rel_error = (y - y_pred)/y_pred
    
    fig,ax = plt.subplots(figsize=(3,3))
    ax.hist(error)
    ax.set_ylabel(r'$y-y_{pred}$')
    fig.savefig(f"{save_loc}/error.pdf", bbox_inches = "tight")
    
    fig,ax = plt.subplots(figsize=(3,3))
    ax.hist(rel_error)
    ax.set_ylabel(r'$(y-y_{pred})/y_{pred}$')
    fig.savefig(f"{save_loc}/rel_error.pdf", bbox_inches = "tight")

    fig,ax = plt.subplots(figsize=(3,3))
    ax.scatter(y,y_pred)
    ax.set_xlabel('actual')
    ax.set_ylabel('predicted by regr')
    fig.savefig(f"{save_loc}/pred_vs_actual.pdf", bbox_inches = "tight")

    return 

def add_analytical_to_dft(df_analytical, df_dft_gb):
    # add the analytical details to df_dft_gb
    df_dft_gb = df_dft_gb.explode(['angle','grain_energy']).reset_index(drop=True)
    df_dft_gb.angle = [round(i,2) for i in df_dft_gb.angle.to_list()]
    df_analytical = df_analytical.explode(['angle','grain_energy_analytical']).reset_index(drop=True)
    df_analytical = df_analytical.drop('grain_energy_analytical_unrelaxed', axis=1)
    df_analytical.angle = [round(i,2) for i in df_analytical.angle.to_list()]
    df_out = pd.merge(df_dft_gb, df_analytical,how = 'left',on=['species','tilt_axis','angle','crystal_type'])
    df_out = df_out.dropna()
    df_out = df_out.groupby(['species','tilt_axis','crystal_type','model'])[['angle','grain_energy','grain_energy_analytical']].agg(list).reset_index()
    return df_out

def combine_gb_dft():
    # originally from 240424_combine_gb_dft.ipynb
    df = pd.read_csv("./data/gb_dft.csv")
    df = df[df['confirmed'] == 'yes']
    df = df.rename({"rotation axis":"tilt_axis",
                    "rotation angle":"angle",
                    "Gamma, J/m^2":"grain_energy"},
                    axis=1)
    df = df.drop(["gb plane","sigma","Gamma, eV/A^2"],axis=1)
    df = df.sort_values(["species","tilt_axis"]).reset_index(drop=True)
    df.tilt_axis = df.tilt_axis.replace({"[100]":"[0, 0, 1]",
                                        "[110]":"[1, 1, 0]",
                                        "[111]":"[1, 1, 1]"},
    )
    df_gb_energy = df.groupby(by=["species","tilt_axis"])['grain_energy'].apply(list)
    df_angle = df.groupby(by=["species","tilt_axis"])['angle'].apply(list)

    df = pd.merge(df_gb_energy,df_angle,on=['species','tilt_axis'])
    df['model'] = ['dft' for i in range(len(df))]
    df['crystal_type'] = ['fcc' for i in range(len(df))]
    df.to_csv("./data/gb_dft_combined.csv")

    return


def error_table(df,
                model_properties,
                label_dict):
    df_out = pd.DataFrame()
    df_out['species'] = df['species']
    for model in model_properties:
        if model == "intr_stack_fault_energy_fcc":
            df_out[label_dict[model]] = (df[model].copy()*10**3).round(3)    
        elif model == "unstable_stack_energy_fcc":
            df_out[label_dict[model]] = (df[model].copy()*10**3).round(3)    
        else:
            df_out[label_dict[model]] = df[model].copy().round(3)
    df_out['Coefficient, exact fit using DFT GB energies'] = df['dft_exact_coeff'].round(3)
    df_out['Coefficient, regressed fit using DFT indicator properties'] = df['regr_coeff'].round(3)
    df_out['Percent Error'] = (100*(df['regr_coeff'] - df['dft_exact_coeff'])/df['dft_exact_coeff']).round(2)
    df_out.to_csv("./images/main/dft_coeff_results.csv")
    return


def dft_df_prep():
    df_ip = pd.read_csv("./data/df_merge_all.csv",index_col = 0)
    df_ip = basic_outlier_removal(df_ip)

    df_dft = get_df_dft()
    combine_gb_dft() # combine dft results, update if needed
    df_dft_gb = pd.read_csv("./data/gb_dft_combined.csv")
    df_dft_gb['grain_energy'] = [literal_eval(i) for i in df_dft_gb.grain_energy]
    df_dft_gb['angle'] = [literal_eval(i) for i in df_dft_gb.angle]

    df_analytical = get_analytical()
    df_analytical['grain_energy_analytical'] = [literal_eval(i) for i in df_analytical.grain_energy_analytical]
    df_analytical['angle'] = [literal_eval(i) for i in df_analytical.angle]

    # 110 is backwards
    # confirmed using Runnels 2016 Fig 5 (see annotations for boundary plane at top)
    # and http://crystalium.materialsvirtuallab.org/
    angle_new = []
    for i,row in df_dft_gb.iterrows():
        if row.tilt_axis == '[1, 1, 0]':
            new_angle = [180-j for j in row.angle]
            angle_new.append(new_angle)
        else:
            angle_new.append(row.angle)
    df_dft_gb.angle = angle_new

    # before merging, need to cut down angle/GB analytical to the angles used
    # make a separate function to do this.
    df_dft_gb = add_analytical_to_dft(df_analytical,
                                      df_dft_gb)

    
    df_dft_gb = get_analytical_coeff(df_dft_gb)
    df_dft_gb = df_dft_gb.rename({'coeff':'dft_exact_coeff'},axis=1)
    df_dft = df_dft.merge(df_dft_gb[['species','dft_exact_coeff']].drop_duplicates().reset_index(drop = True), 
                          on = 'species',
                          how = 'left',
                          )


    return df_ip, df_dft, df_dft_gb, df_analytical


def dft_prop_eval(df_dft,
                  prop_list,
                  label_dict):

    df = df_dft.copy()
    df = df.rename(label_dict, axis=0)
    df = df.rename(label_dict, axis=1)
    prop_list = [label_dict[i] for i in prop_list]
    fig,ax = plt.subplots(figsize=(3,3))
    plot_list = prop_list
    plot_list.extend(['dft_exact_coeff','species'])
    sns.pairplot(df[prop_list],hue='species',plot_kws = {"s":100})

    plt.savefig("dft_props.png",bbox_inches = 'tight')

    return


def main():
    readme = ""

    df_ip, df_dft, df_dft_gb, df_analytical = dft_df_prep()

    # list of possible DFT properties
    all_dft_properties = ['c44_fcc',
                          'surface_energy_111_fcc',
                          'unstable_stack_energy_fcc',
                          'intr_stack_fault_energy_fcc',
                          'lattice_constant_fcc',
                          ]

    # best model of the 5 available DFT properties
    model_properties = ['c44_fcc', 
                        'intr_stack_fault_energy_fcc', 
                        'unstable_stack_energy_fcc']

    #df, readme = data_import(clean=True)
    label_dict = import_label_dict()

    # predictor/response variables
    X_df = df_ip[model_properties]
    y = df_ip['coeff']
    X = X_df[model_properties]

    # first, get estimate for RMSE
    pipe = linear_model_create()
    y_pred, y_actual = cv_lm(df_ip,
                             model_properties,
                             pipe,
                             "dft_ip",
                             12345,
                             save = False)
    error_plots(y_actual, y_pred)
    rmse = mean_squared_error(y_actual, y_pred, squared = False)

    readme += "---------------dft results---------------\n"
    readme += f"factors used: {model_properties}\n\n"
    readme += f"rmse = {rmse}\n"
    readme += f"overall mean of strength: {np.mean(y)}\n"
    readme += f"averaged relative rmse = {rmse/np.mean(y)}\n"
    readme += f"2*rmse = {2*rmse}\n"  

    # now make regr prediction using all data
    pipe = linear_model_create()
    pipe.fit(X,y)
    df_dft = get_dft_regr_coeff(df_dft,
                                pipe,
                                model_properties,
                                rmse)

    print(df_dft)
    error_table(df_dft,
                model_properties,
                label_dict)
    species = ['Al','Cu','Ni','Ag','Au','Pd','Pt']

    # plotting
    get_boxplot(df_ip, 
                df_dft,
                df_dft_gb,
                readme,
                save_loc = "images/main",
                file_name='dft_w_pi_kfold',
                plot_errorbar=True)
        

    tilt_axis = ['[0, 0, 1]', '[1, 1, 1]', '[1, 1, 0]', '[1, 1, 2]']
    dft_gb_energy_plots(species,
                        tilt_axis,
                        df_ip,
                        df_analytical,
                        df_dft_gb,
                        df_dft,
                        legend=True)
           
    # generate property boxplots
    if True:
        fig_list = []
        for prop in all_dft_properties:
            fig_new = get_prop_boxplot(df_ip, 
                                       prop, 
                                       label_dict[prop], 
                                       df_dft,
                                       save_loc = "images/supplemental")
            fig_list.append(fig_new)
        
        path_name = 'images/dft/dft_props_combined.pdf'
        make_combined_pdf(fig_list,
                          path_name)
    return

if __name__=="__main__":
    main()