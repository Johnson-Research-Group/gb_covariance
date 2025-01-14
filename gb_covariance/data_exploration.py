import pandas as pd
import plotting
import models, model_selection
import seaborn as sns
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from ast import literal_eval

sns.set(font_scale=1.25)


def import_label_dict():
    df_labels = pd.read_csv("./gb_covariance/data_ays/label_dict.csv")
    label_dict = df_labels.to_dict(orient="records")[0]
    return label_dict


def corr_w_coeff(df):
    out = (df.corr(numeric_only = True)['coeff'].abs().sort_values(ascending=False))
    out.to_csv("./gb_covariance/data_ays/corr_w_coeff.csv")
    return 


def get_factor_list(df):
    factor_list = df.columns.tolist()
    factor_list = [i for i in factor_list if ("_fcc" in i) or ("_bcc" in i) or ("_sc" in i) or ("_diamond" in i) or ("_hcp" in i)]
    factor_list = [i for i in factor_list if df[i].count()>0]
    return factor_list


def pairplot(df, 
             params_list, 
             filename, 
             label_dict, 
             xlims = False, 
             height=0.75,
             file_loc = "./gb_covariance/data_ays"):
    params_list.extend(['species'])
    X = df[params_list]
    X.columns = [label_dict[x] for x in X.columns.to_list()]
    sns.set_theme(style="white", font_scale = .5)
    marker_list = ['o','^','v','<','>','s','D','p','X','*','.','P']
    g = sns.pairplot(X, hue='species', corner=True, markers = marker_list[0:len(df.species.drop_duplicates())],
                     plot_kws=dict(s=7, linewidth=0.1, rasterized = True), 
                     height=height,
                     diag_kind='hist')
    sns.move_legend(g, "upper right", bbox_to_anchor = (0.85,1))

    if xlims != False:
        for i in range(len(g.axes)):
            for j in range(len(g.axes[i])):
                g.axes[i][j].set_xlim(xlims)
                g.axes[i][j].set_ylim(xlims)

    g.savefig(f"{file_loc}/{filename}.pdf", bbox_inches = "tight", dpi=500)


def pairplots_manuscript(df, label_dict):
    """produce pairplots for manuscript
    """
    df['ao * c44'] = (df['c44_fcc']*10**9)*df['lattice_constant_fcc']*(10**(-10))
    df['Cohesive energy / ao^2'] = (df['cohesive_energy_fcc']*1.6022E-19)/((df['lattice_constant_fcc']*(10**(-10)))**2)
    pairplot(df, ['coeff',
                  'c44_fcc',
                  'ao * c44',
                  'cohesive_energy_fcc',
                  'Cohesive energy / ao^2'],
             'pairplot_specific_props', label_dict,
             file_loc = "./images/main")
    
    pairplot(df, 
             ['coeff',
              'vacancy_migration_energy_fcc',
              'relaxed_formation_potential_energy_fcc',
              'unrelaxed_formation_potential_energy_fcc',
              'unstable_stack_energy_fcc',
              'c44_fcc',
              'unstable_twinning_energy_fcc',
              ],
             'pairplot_top_factors', label_dict,
             file_loc = "./images/main")
    
    

def pairplots_supplemental(df, label_dict):
    """produce pairplots for supplemental material
    """
    pairplot(df, 
             ['coeff','c11_fcc','c12_fcc','bulk_modulus_fcc',],
             'pairplot_bulk_consts_fcc', 
             label_dict,
             height=0.7,
             file_loc = "./images/supplemental")
    
    pairplot(df, 
             ['coeff','bulk_modulus_sc','bulk_modulus_fcc','bulk_modulus_bcc'],
             'pairplot_bulk_mod', 
             label_dict,
             file_loc = "./images/supplemental")
    
    pairplot(df, ['coeff',
                  'surface_energy_100_fcc',
                  'surface_energy_110_fcc',
                  'surface_energy_111_fcc',
                  'surface_energy_121_fcc'],
             'pairplot_surf_energy_fcc', label_dict,
             file_loc = "./images/supplemental")
    
    pairplot(df, 
             ['coeff',
              'surface_energy_100_bcc',
              'surface_energy_110_bcc',
              'surface_energy_111_bcc',
              'surface_energy_121_bcc'],
             'pairplot_surf_energy_bcc', 
             label_dict,
             file_loc = "./images/supplemental")

    pairplot(df, ['coeff',
                  'relaxed_formation_potential_energy_fcc',
                  'unrelaxed_formation_potential_energy_fcc',
                  'vacancy_migration_energy_fcc',
                  'relaxation_volume_fcc',
                  'surface_energy_100_fcc',
                  'cohesive_energy_fcc'],
             'pairplot_vac_form', label_dict,
             file_loc = "./images/supplemental")
    
    df['ao * c44'] = df['c44_fcc']*df['lattice_constant_fcc']
    pairplot(df, ['coeff',
                  'c44_fcc',
                  'lattice_constant_fcc',
                  'ao * c44'],
                  'pairplot_elastic_C44_combos', label_dict,
                  file_loc = "./images/supplemental")



    pairplot(df, ['coeff',
                  'lattice_constant_sc',
                  'lattice_constant_fcc',
                  'lattice_constant_bcc'],
             'pairplot_lattice_consts', label_dict,
             file_loc = "./images/supplemental")
    
    pairplot(df, ['coeff',
                  'extr_stack_fault_energy_fcc',
                  'intr_stack_fault_energy_fcc',
                  'unstable_stack_energy_fcc',
                  'unstable_twinning_energy_fcc'],
             'pairplot_stacking_fault_full', label_dict,
             file_loc = "./images/supplemental")

 
def scaled_pairplot(df, params, label_dict):
    """extra pairplots w/ scaled values
    """
    params.append("coeff")
    y = df['species'].reset_index()
    X = df[params]
    X_scaled = StandardScaler().fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns = X.columns)
    x_min = X_scaled.min().min()
    x_max = X_scaled.max().max()
    df_scaled = pd.merge(y,X_scaled, left_index=True, right_index=True)
    pairplot(df_scaled,
             params,
             'scaled_pairplot', label_dict, (x_min*.95,x_max*1.05))
    return X_scaled


def fix_df_csv_import(df):
    """convert strings to values in df
    """
    cols_to_check = ['angle', 'grain_energy','grain_energy_analytical']
    for col in cols_to_check:
        if col in df.columns:
            if isinstance(df[col].iloc[0], str):
                df[col] = [literal_eval(i) for i in df[col]]
    return df

def plot_md_analytical_and_dft(file_loc):
    """plot MD, analytical and DFT results
    """
    df = pd.read_csv('./data/df_merge_all.csv', index_col= 0)
    #df = df[df.species == "Al"] #temp for debugging
    df = fix_df_csv_import(df)
    
    df_md_avg = pd.read_csv('data/df_md_avg.csv', index_col = 0)
    df_md_avg = fix_df_csv_import(df_md_avg)
    df_md_avg = models.get_avg_gb_energy(df, df_md_avg)
    species = list(df['species'].drop_duplicates())
    plotting.plot_gb_vs_angle(species,
                              ['[0, 0, 1]', '[1, 1, 1]', '[1, 1, 0]', '[1, 1, 2]'],
                              df,
                              df_md_avg,
                              file_loc = file_loc)
    

def correlation_df(df, label_dict):
    """create df with correlation values
    """
    df_corr = df.corr().round(2)
    columns = df_corr.columns.to_list()
    columns = [label_dict[x] for x in columns]
    df_corr.columns = columns
    
    df_index = df_corr.index.to_list()
    df_index = [label_dict[x] for x in df_index]
    df_corr.index = df_index

    order = df_corr['$E_0$'].sort_values(ascending=False).index.to_list()
    df_corr = df_corr[order].reindex(order)
    return df_corr


def coeff_corr_barchart(df,
                        label_list,
                        label_dict,
                        title,
                        figsize = (6,1)):
    df_corr = correlation_df(df[label_list], label_dict)
    df_corr = df_corr['$E_0$'].drop('$E_0$')
    fig,ax = plt.subplots(figsize = (figsize[0],figsize[1]))
    df_corr.plot(kind='bar')
    ax.tick_params(axis='both',labelsize=6)
    fig.savefig(f"./gb_covariance/data_ays/{title}.pdf", bbox_inches = "tight")
    return


def corr_plot(df, 
              label_list, 
              label_dict, 
              title, 
              annot = False, 
              save = True,
              figsize = (10,10),
              annotation_fontsize = 6,
              tick_fontsize = 6,
              include_coeff = True):
    
    df_corr = correlation_df(df[label_list], label_dict)
    df_corr.to_csv(f"./gb_covariance/data_ays/{title}.csv")
    if include_coeff == False:
        df_corr = df_corr.drop("$E_0$",axis=0)
        df_corr = df_corr.drop("$E_0$",axis=1)
    fig,ax = plt.subplots(figsize = (figsize[0],figsize[1]))
    colors = sns.color_palette("vlag", as_cmap=True)
    sns.heatmap(df_corr, vmin = -1, vmax = 1, cmap=colors,
                annot=annot,
                xticklabels=True,
                yticklabels=True,
                fmt = '.1f',
                annot_kws = {"fontsize":annotation_fontsize},
                cbar_kws = {"location":"top"}
                )
    ax.tick_params(labelsize=tick_fontsize)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=tick_fontsize)
    if save == True:
        fig.savefig(f"./gb_covariance/data_ays/{title}.pdf", bbox_inches = "tight")


def corr_plot_w_bar(df, 
              label_list, 
              label_dict, 
              title, 
              annot = False, 
              save = True,
              figsize = (10,10),
              annotation_fontsize = 6,
              tick_fontsize = 6,
              include_coeff = True,
              save_loc = './gb_covariance/data_ays'):
    df_corr = correlation_df(df[label_list], label_dict)
    df_corr.to_csv(f"./gb_covariance/data_ays/{title}.csv")
    corr_to_coeff = df_corr['$E_0$'].values.tolist()[1:]
    if include_coeff == False:
        df_corr = df_corr.drop("$E_0$",axis=0)
        df_corr = df_corr.drop("$E_0$",axis=1)
    fig,axs = plt.subplots(2, figsize = figsize, gridspec_kw={'height_ratios': [8, 1]}, sharex=True)
    colors = sns.color_palette("vlag", as_cmap=True)
    g = sns.heatmap(df_corr, vmin = -1, vmax = 1, cmap=colors,
                annot=annot,
                xticklabels=True,
                yticklabels=True,
                fmt = '.1f',
                annot_kws = {"fontsize":annotation_fontsize},
                cbar_kws = {"location":"top","shrink":0.5},
                ax = axs[0]
                )
    axs[0].set(facecolor = 'white')
    g.tick_params(labelsize=tick_fontsize)
    cbar = g.collections[0].colorbar
    cbar.ax.tick_params(labelsize=tick_fontsize)
    
    params = df_corr.index.tolist()
    x_loc = range(len(params))
    axs[1].bar(x=x_loc, height=corr_to_coeff, align='edge',width=1.0)
    axs[1].margins(x=0.0) #https://stackoverflow.com/questions/63719618/how-to-reduce-the-space-between-the-axes-and-the-first-and-last-bar
    axs[1].tick_params(axis='x', labelsize = tick_fontsize, rotation=90)
    axs[1].tick_params(axis='y', labelsize = tick_fontsize)
    axs[1].set_ylim([-1,1])

    plt.subplots_adjust(hspace=0.03) # spacing b/w subplots
    if save == True:
        fig.savefig(f"{save_loc}/{title}.pdf", bbox_inches = "tight")


def model_stats(df):
    """model statistics
    """
    stats = []
    # number of unique models
    stats.append(['model_count', df.model.drop_duplicates().count()])
    stats.append(['species_count', df.species.drop_duplicates().count()])
    df_stats = pd.DataFrame(stats)
    df_stats.to_csv("./gb_covariance/data_ays/data_stats.csv", index=False)
    return 


def save_pdf(df):    
    name = "./gb_covariance/data_ays/all_data_plots.pdf"
    with PdfPages(name) as pdf:
        for p in range(len(df)):
            fig = plotting.plot_analytical_model(df, [p], savefig = False)
            pdf.savefig()
            plt.close()
    return


def save_model_plot_pdf(df,figsize = (6,2)):
    name = "./gb_covariance/data_ays/all_data_model_plots.pdf"
    model_species = df[['model','species']].drop_duplicates().values.tolist()
    with PdfPages(name) as pdf:
        j = 0
        for model,species in model_species:
            j += 1
            print(f"{j} of {len(model_species)}: {model},{species}")
            df_current = df[(df['model'] == model) & (df['species'] == species)]
            fig,axs = plt.subplots(1, 
                                4, 
                                figsize = figsize,
                                sharey = True)
            for i in range(len(df_current)):
                row = df_current.iloc[i]
                tilt_axis = row.tilt_axis
                angle = literal_eval(row.angle)
                gb_energy = literal_eval(row.grain_energy)
                c = row.coeff
                gb_analytical = np.array(literal_eval(row.grain_energy_analytical))
                axs[i].scatter(angle,gb_energy, label='sim results', s=5)
                axs[i].plot(angle,c*gb_analytical,label='scaled analytical')
                axs[i].plot(angle,gb_analytical, label='base analytical')
            handles, labels = axs[-1].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center',
                       bbox_to_anchor = (0.5,1.05),
                       ncol = len(df_current),
                       fontsize = 8)
            fig.suptitle(f"{species}, {model}", fontsize = 8)
            pdf.savefig(bbox_inches = 'tight')
            plt.close()
    return


def create_IP_table(df):
    model_list = (df["model"].drop_duplicates())
    model_list = [i.split("__")[0] for i in model_list]
    model_list = [i.split("_")[:-3] for i in model_list]
    model_list = ["-".join(i) for i in model_list]
    df_models = pd.DataFrame(model_list)
    df_models = df_models.value_counts().reset_index()
    df_models.columns = ["model type","count"]
    df_models.to_csv("gb_covariance/data_ays/model_counts.csv")
    return


def main():
    # read in dataframes
    df_merge = pd.read_csv('./data/df_merge.csv', index_col= 0)
    df_merge_all = pd.read_csv('./data/df_merge_all.csv', index_col= 0)
    label_dict = import_label_dict()

    # basic outlier removal
    df_merge = model_selection.basic_outlier_removal(df_merge)

    # correlation coefficients and model stats
    corr_w_coeff(df_merge)
    model_stats(df_merge)

    params_list = ['lattice_constant', 
                   'bulk_modulus', 'c11', 'c12', 'c44',
                   'cohesive_energy_fcc', 'thermal_expansion_coeff_fcc',
                   'surface_energy_100_fcc',
                   'extr_stack_fault_energy', 
                   'intr_stack_fault_energy',
                   'unstable_stack_energy', 
                   'unstable_twinning_energy',
                   'relaxed_formation_potential_energy_fcc', #includes unrelaxed
                   'vacancy_migration_energy_fcc',
                   'relaxation_volume_fcc']

    params_list_heatmap = models.filter_param_list(df_merge, params_list, ['coeff'])

    params_list_full = get_factor_list(df_merge)
    params_list_full.append('coeff')   
    
    # individual species GB plots w/ average for manuscript
    if True: 
        plot_md_analytical_and_dft('images/supplemental/gb_vs_angle') 

    # correlation heatmap w/ barchart for manuscript and supplemental
    if True:
        corr_plot_w_bar(df_merge, 
                  params_list_heatmap, 
                  label_dict, 
                  "correlation_heatmap_manuscript", 
                  annot = True,
                  figsize =(6,6),
                  annotation_fontsize=5,
                  tick_fontsize = 8,
                  include_coeff = False,
                  save_loc = "images/main")

        corr_plot_w_bar(df_merge, 
                  params_list_full, 
                  label_dict, 
                  "correlation_heatmap_supplemental", 
                  annot = True,
                  figsize =(6,8),
                  annotation_fontsize=4,
                  tick_fontsize = 6,
                  include_coeff = False,
                  save_loc = "images/supplemental")

    # pairplots for manuscript
    if True:
        pairplots_manuscript(df_merge, label_dict)

    # pairplots for supplemental in manuscript
    if True:
        pairplots_supplemental(df_merge, label_dict)
    
    # extra plotting options below, not needed for manuscript
    if False:
        corr_plot(df_merge, 
                  params_list_full, 
                  label_dict, 
                  "correlation_heatmap_manuscript", 
                  annot = True,
                  figsize=(6,6),
                  annotation_fontsize=5,
                  tick_fontsize = 8,
                  include_coeff = False)
        coeff_corr_barchart(df_merge,
                            params_list_full,
                            label_dict,
                            'correlation_barchart_manuscript')

    if False:
        save_pdf(df_merge_all)
        plotting.plot_analytical_model(df_merge_all, [272])

    if False:
        save_model_plot_pdf(df_merge_all)

    if False:
        # SI table of IPs used
        create_IP_table(df_merge)

if __name__ == "__main__":
    main()
