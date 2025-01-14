import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import expon
import seaborn as sns
from ast import literal_eval


def plot_gb_vs_angle(species, 
                     tilt_axis, 
                     df, 
                     df_md_avg=[], 
                     df_dft=[], 
                     df_ay_w_dft = [], 
                     legend = False,
                     file_loc = 'gb_covariance/data_ays/gb_vs_angle'):
    n_subplots = len(tilt_axis)
    n_species = len(species)
    for j in range(n_species):
        fig, ax = plt.subplots(1,n_subplots, tight_layout = True, figsize = (12,4), dpi = 300)#, figsize=(4,8))#,layout='constrained')
        s = species[j]
        for i,t in enumerate(tilt_axis):
        #plot all DFT gb energy results
            print(s, i, t)
            if len(df_dft) > 0:
                dft = df_dft.copy()
                dft = dft[(dft['species'] == s) & (dft['tilt_axis'] == t)]
                ax[j,i].scatter(dft['angle'], dft['gb_energy'],marker='X',label='DFT')
        
            #plot all IM results
            df1 = df.copy()
            df1 = df1[df1['species'] == s]
            df1 = df1[df1['tilt_axis'] == t]
            crystal_type = df1.crystal_type.drop_duplicates().values
            print('df shape:',df1.shape)
            for a,b in zip(df1['angle'], df1['grain_energy']):
                ax[i].plot(a, b, linewidth=2, alpha = 0.10)
            
            if len(df_md_avg) > 0:
                #md average items
                regr_from_md_avg = df_md_avg[(df_md_avg['species'] == s) & (df_md_avg['tilt_axis'] == t)]            
                angle_md_avg = np.array(regr_from_md_avg['angle'].to_list()).reshape(-1)

                #plot average IM GB energies
                avg_grain_energy = np.array(regr_from_md_avg['grain_energy_mean'].to_list()).reshape(-1)
                ax[i].plot(angle_md_avg, 
                           avg_grain_energy, 
                           linewidth=2,
                           label = 'GB.avg from sim' if i == 0 else "", 
                           color = 'black')

                #plot average coefficient times analytical using average lattice constant
                anly_md_avg = np.array(regr_from_md_avg['grain_energy_analytical'].to_list()).reshape(-1)
                c_md_avg = float(regr_from_md_avg['coeff'])
                ax[i].plot(angle_md_avg, 
                           c_md_avg*anly_md_avg, 
                           alpha = 1, 
                           label='analytical using c.avg' if i == 0 else "", 
                           linestyle = 'dashed',
                           linewidth=2,
                           color = 'green')
                crystal_type = df_md_avg['crystal_type'].drop_duplicates().values

                #plot regressed coefficient times analytical using average lattice constant
                if 'coeff_regr' in regr_from_md_avg.columns:
                    c_regr_avg = float(regr_from_md_avg['coeff_regr'])
                    ax[i].plot(angle_md_avg, c_regr_avg*anly_md_avg, alpha = 1, label='c.regr', color = 'red')

                if 'coeff_svr_regr' in regr_from_md_avg.columns:
                    c_regr_svr_avg = float(regr_from_md_avg['coeff_svr_regr'])
                    ax[i].plot(angle_md_avg, c_regr_svr_avg*anly_md_avg, alpha = 1, label='c.regr_svr', color = 'orange')
    

            #plot regr prediction w/ dft canonical props
            if len(df_ay_w_dft) > 0:
                regr_from_dft = df_ay_w_dft[(df_ay_w_dft['species'] == s) & (df_ay_w_dft['tilt_axis'] == t)]
                angle_dft_avg = np.array(regr_from_dft['angle'].to_list()).reshape(-1)
                anly_dft_avg = np.array(regr_from_dft['grain_energy_analytical'].to_list()).reshape(-1)
                c_dft_avg = float(regr_from_dft['coeff'])
                crystal_type = regr_from_dft['crystal_type'].drop_duplicates().item()
                ax[i].plot(angle_dft_avg, c_dft_avg*anly_dft_avg, alpha = 1, label='Regr from DFT', color = 'blue')

            tilt_axis_dict = {'[0, 0, 1]':'[001]',
                              '[1, 1, 1]':'[111]',
                              '[1, 1, 0]':'[110]',
                              '[1, 1, 2]':'[112]'}    
            ax[i].set_xlabel(f"{tilt_axis_dict[t]} angle")#,fontsize = 18)
            ax[i].set_ylabel('GB energy')#,fontsize = 18)
            #ax[i].set_title(f"{s},{t},{crystal_type}")#,fontsize = 18)
            #ax[i].set(xlabel = 'angle', ylabel = 'grain boundary energy',
            #    title = f"{s}, {t}, {crystal_type}")
        
        if legend == True:
            fig.legend(bbox_to_anchor = (0.25,0.9,0.5,.15),#(0.,1.02,1.,.102),
                    loc='lower left',
                    mode="expand",
                    #bbox_transform = fig.transFigure)
                    ncol = 2,
                    fontsize= 12)

        fig.savefig(f"{file_loc}/{species[j]}.pdf")
        plt.close()
        
    
    #plt.show()

    return

def plot_pred_vs_actual_basic(title_name, y, y_pred):
    # be careful with rmse: no test/train split?
    rmse = mean_squared_error(y,y_pred)**.5
    r2 = r2_score(y, y_pred)
    plt.scatter(y_pred, y, label=f'N={len(y_pred)}')
    plt.xlabel('prediction')
    plt.ylabel('actual')
    plt.legend()
    plt.title(f'{title_name}\n'
              f'RMSE = {rmse:.3f}, R\N{SUPERSCRIPT TWO} = {r2:.3f}\n')
    plt.plot(np.linspace(0,max(y),50),np.linspace(0,max(y),50))
    plt.show()
    return

def plot_pred_vs_actual(title,
                        y_train,
                        y_pred_train,
                        y_test, 
                        y_pred_test,
                        filename,
                        gb_energy = True,
                        plot_hist = False):

    rmse = mean_squared_error(y_test,y_pred_test)**.5
    min_gb_energy_error = rmse/max(y_test)
    max_gb_energy_error = rmse/min(y_test)
    r2 = r2_score(y_test, y_pred_test)
    rel_resid_error = np.abs((y_test-y_pred_test)/y_test)
    _,mean = expon.fit(rel_resid_error)
    cutoff = expon.isf(.1, scale = mean)
    if plot_hist == True:
        fig,ax = plt.subplots()
        hist = ax.hist(rel_resid_error)
        ax.set_xlabel('Residual rel. error')
        ax.set_title(f'Residual relative error (test data) \n' #Exponential distribution
                f'90% cutoff = {cutoff:.3f}')
        ax.bar_label(hist[-1])
        #for p in ax.patches:
        #    if p.get_height() > 0:
        #        #ax.annotate(str(int(p.get_height())),(p.get_x()*1.005,p.get_height()*1.005), textcoords='data')
        #        ax.bar_label
        plt.show()

    #------------------
    plt.scatter(y_pred_train, y_train, label=f'Train, N={len(y_pred_train)}')
    plt.scatter(y_pred_test, y_test, label=f'Test, N={len(y_pred_test)}')
    plt.xlabel('prediction')
    plt.ylabel('actual')
    plt.legend()
    x_loc = max(y_pred_train)
    plt.text(x_loc,0.01,f'Test RMSE = {rmse:.3f}, R\N{SUPERSCRIPT TWO} = {r2:.3f}\n', 
             horizontalalignment='right',
             verticalalignment='center')
    # possible string: f'Additional GB energy relative error using predicted coeff (90% cutoff): {cutoff:.3f}')
    plt.title(title)
    plt.plot(np.linspace(0,max(y_test),50),np.linspace(0,max(y_test),50))
    plt.savefig(f"./gb_covariance/model_ays/{filename}.png")

    return


def plot_analytical_model(df, loc, savefig = True):
    """
    creates plot of MD and analytical (scaled/unscaled) points for given loc
    """
    angle = df.iloc[loc].angle.apply(literal_eval).iloc[0]
    ge_ay = df.iloc[loc].grain_energy_analytical.apply(literal_eval).iloc[0]
    ge_md = df.iloc[loc].grain_energy.apply(literal_eval).iloc[0]
    c = df.iloc[loc].coeff.iloc[0]
    
    title = f"{df.iloc[loc].species.iloc[0]}, tilt axis {df.iloc[loc].tilt_axis.iloc[0]}" 
    
    plt.plot(angle,ge_ay,label='analytical')
    plt.scatter(angle,ge_md,label='MD')
    plt.plot(angle,c*np.array(ge_ay),label='analytical w/ scale')
    plt.xlabel('angle [deg]')
    plt.ylabel('grain boundary energy [J/m^2]')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    #plt.show()
    if savefig == True:
        filename = f"{df.iloc[loc].species.iloc[0]}_{df.iloc[loc].tilt_axis.iloc[0]}_{loc}"
        plt.savefig(f"./gb_covariance/data_ays/{filename}.png", dpi=300)
        plt.savefig(f"./gb_covariance/data_ays/{filename}.tif", dpi=300)

    return plt