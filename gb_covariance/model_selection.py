import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from itertools import combinations
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score, LeaveOneOut, GridSearchCV
from sklearn.impute import KNNImputer
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn import linear_model, svm
import data_exploration, models, plotting
import seaborn as sns
from ast import literal_eval


def basic_outlier_removal(df):
    """remove extreme canonical property predictions

    :param df: dataframe with outliers
    :type df: pandas.DataFrame
    :return: dataframe without outliers
    :rtype: pandas.DataFrame
    """

    columns = df.columns.to_list()

    # get list of bulk_modulus
    bulk_list = [i for i in columns if "bulk" in i]
    # get list of elastic constants
    elastic_const_list = [i for i in columns if "c11" in i]
    elastic_const_list.extend([i for i in columns if "c12" in i])
    elastic_const_list.extend([i for i in columns if "c44" in i])
    # get list of stack_fault_energy
    stack_energy_list = [i for i in columns if "stack_fault_energy" in i]

    # set limits and remove extreme outliers
    df['vacancy_migration_energy_fcc'] = df['vacancy_migration_energy_fcc'][df['vacancy_migration_energy_fcc'] < 100]
    df['vacancy_migration_energy_fcc'] = df['vacancy_migration_energy_fcc'][df['vacancy_migration_energy_fcc'] > -100]

    df[bulk_list] = df[bulk_list][df[bulk_list] < 1e5]
    df[elastic_const_list] = df[elastic_const_list][df[elastic_const_list] < 1e5]
    df[stack_energy_list] = df[stack_energy_list][df[stack_energy_list] < 0.45]
    
    return df


def filter_param_list(df, base_labels, specific_items=""):
    """generate filtered list of specific parameters

    :param df: dataframe with columns to consider
    :type df: pd.DataFrame
    :param base_labels: base label strings to consider
    :type base_labels: list
    :param specific_items: specific strings to include
    :type specific_items: list
    :return: list of specific label strings of parameters
    :rtype: list
    """
    params_list_full = []
    params_list_full.extend(specific_items)

    for i in base_labels:
        current_list = [j for j in df.columns if i in j]
        params_list_full.extend(current_list)

    return params_list_full


def factor_select_cv(X, y, pipe, n_factor_max=2, cv=5, scoring='r2'):
    """return list of parameters w/ cv score
    """
    factor_list = X.columns.to_list()
    subsets = []
    for n in range(1, (n_factor_max+1)):
        for subset in combinations(factor_list, n):
            subsets.append(list(subset))

    cv_score_mean = []
    cv_score_std = []
    num_factors = []
    for i, subset in enumerate(subsets):
        print(f"{i} of {len(subsets)}")
        print('current subset: ', subset)
        score = cross_val_score(pipe, X[subset], y, cv=cv, scoring=scoring, n_jobs = -1)
        print('score mean: ', np.mean(score))
        cv_score_mean.append(np.mean(score))
        cv_score_std.append(np.std(score))
        num_factors.append(len(subset))

    df_results = pd.DataFrame({'factors': subsets,
                               'num_factors': num_factors,
                               'cv_score': cv_score_mean,
                               'cv_score_std': cv_score_std})
    df_results = df_results.sort_values('cv_score', ascending=False)

    return df_results


def coeff_and_r2_improvement(df, X_factor_list, y_factor):
    if y_factor in X_factor_list:
        X_factor_list.remove(y_factor)
    X = df[X_factor_list]
    y = df[y_factor]

    # Need linear model for factor comparison
    model = linear_model.LinearRegression()
    imput = KNNImputer(n_neighbors=2, weights="uniform")
    pipe = Pipeline(steps=[('scale', StandardScaler()),
                           ('imp', imput),
                           ('lr', model)])

    pipe_out = pipe

    factor = list(X.columns)
    r2_improvement_perc = models.estimator_factor_influence_base(
        pipe_out, X, y, plot=False)
    df = pd.DataFrame({'factor': factor,
                       "r2_improve_perc": r2_improvement_perc})
    pipe_out.fit(X, y)
    std_coeff = pipe_out.named_steps.lr.coef_
    label = pipe_out.feature_names_in_

    df2 = pd.DataFrame({'std_coeff': abs(std_coeff),
                        'factor': label})
    df = df.merge(df2, on='factor')
    df = df.set_index('factor')
    df = df.sort_values('r2_improve_perc', ascending=False)
    return df


def factor_percent_usage(df_results, N_lines, title):
    """calculate percentage of times a factor is used
    """
    factor_list_in = df_results['factors'].iloc[:N_lines]
    factor_list = [i for i in factor_list_in]
    if isinstance(factor_list[0], str):
        factor_list = [literal_eval(i) for i in factor_list_in] #only needed when importing a csv
    factor_list_combined = [j for i in factor_list for j in i]
    df_factors = pd.DataFrame(factor_list_combined)
    fig = plt.figure()
    ax = df_factors.value_counts().plot.bar()
    fig.add_axes(ax)
    fig.subplots_adjust(bottom=0.6)
    fig.savefig(f"./gb_covariance/model_ays/{title}.png", dpi=300)
    return df_factors.value_counts().rename_axis('factor').reset_index(name='count')


def create_factor_select_plot(df_merge, params, filename, label_dict):
    """different options for creating factor selection plot
    """
    df_corr = df_merge.corr(numeric_only = True).round(2)
    abs_coeff_corr = abs(df_corr['coeff']).sort_values(ascending=False)
    abs_coeff_corr.index.name = "factor"
    abs_coeff_corr = abs_coeff_corr.rename('corr_coeff')

    df_results_loocv = pd.read_csv("./gb_covariance/model_ays/kfold_models_svr.csv", index_col=0)
    boxplot = df_results_loocv.boxplot("cv_score","num_factors")
    plt.savefig(f"./gb_covariance/model_ays/{filename}_boxplot.png", dpi=300)
    df_factor_count = factor_percent_usage(df_results_loocv, 100, 'loocv_factor_usage')

    df = coeff_and_r2_improvement(
        df_merge, params, 'coeff')
    
    df = df.merge(abs_coeff_corr,how="left",on="factor")
   
    df = df.merge(df_factor_count, how="left", on='factor').set_index("factor").fillna(0)
    df = df.sort_values("count", ascending=False)
    df = df[['count','corr_coeff','std_coeff','r2_improve_perc']]
    df = df.rename(columns = {"corr_coeff":"Correlation Coefficient",
                              "count":"Usage, top 100 models",
                              "r2_improve_perc":"$R^2$ improvement (percentage)",
                              "std_coeff":"Standardized Regression Coefficient (absolute)"})
    
    df = (df - df.min())/(df.max() - df.min())

    
    factor_select_plotting(df, label_dict, filename)


def create_factor_select_plot_2(df_merge, params, filename, label_dict):
    """different options for creating factor selection plot
    """
    df_corr = df_merge.corr(numeric_only = True).round(2)
    abs_coeff_corr = abs(df_corr['coeff']).sort_values(ascending=False)
    abs_coeff_corr.index.name = "factor"
    abs_coeff_corr = abs_coeff_corr.rename('corr_coeff')

    df_results_loocv = pd.read_csv("./gb_covariance/model_ays/kfold_models_svr.csv", index_col=0)
    boxplot = df_results_loocv.boxplot("cv_score","num_factors")
    plt.savefig(f"./gb_covariance/model_ays/{filename}_boxplot.png", dpi=300)
    df_factor_count = factor_percent_usage(df_results_loocv, 100, 'loocv_factor_usage')

    df = coeff_and_r2_improvement(
        df_merge, params, 'coeff')
    
    df = df.merge(abs_coeff_corr,how="left",on="factor")
    df = df.merge(df_factor_count, how="left", on='factor').set_index("factor").fillna(0)
    df = df[['count','corr_coeff','std_coeff']]
    df = (df - df.min())/(df.max() - df.min())
    df = df.rename(columns = {"corr_coeff":"Correlation Coefficient",
                              "count":"Usage, top 100 models",
                              "std_coeff":"Standardized Regression Coefficient (absolute)"})
    
    df = df.sort_values("Correlation Coefficient", ascending=False)
    factor_select_plotting(df.iloc[:10,:], label_dict, filename+"_corr")

    df = df.sort_values("Usage, top 100 models", ascending=False)
    factor_select_plotting(df.iloc[:10,:], label_dict, filename+"_count")


def create_factor_select_plot_3(df_merge, params, filename, label_dict, figsize = (16,10)):
    """different options for creating factor selection plot
    """
    df_corr = df_merge.corr(numeric_only = True).round(2)
    abs_coeff_corr = abs(df_corr['coeff']).sort_values(ascending=False)
    abs_coeff_corr.index.name = "factor"
    abs_coeff_corr = abs_coeff_corr.rename('corr_coeff')

    df_results_loocv = pd.read_csv("./gb_covariance/model_ays/kfold_models_svr.csv", index_col=0)
    boxplot = df_results_loocv.boxplot("cv_score","num_factors")
    plt.savefig(f"./gb_covariance/model_ays/{filename}_boxplot.png", dpi=300)
    df_factor_count = factor_percent_usage(df_results_loocv, 100, 'loocv_factor_usage')
    
    df = pd.merge(df_factor_count, abs_coeff_corr, how="left", on='factor').set_index("factor").fillna(0)
    df = df[['count','corr_coeff']]
    df = df.rename(columns = {"corr_coeff":"Correlation Coefficient",
                              "count":"Usage, top 100 models"})
    
    df = df.sort_values("Correlation Coefficient", ascending=False)
    factor_select_plotting_2(df, label_dict, filename+"_corr", width = 0.325, figsize = figsize)

    df = df.sort_values("Usage, top 100 models", ascending=False)
    factor_select_plotting_2(df, label_dict, filename+"_count", width = 0.325, figsize = figsize)


def create_factor_select_plot_4(df_merge, params, filename, label_dict, figsize = (6,4)):
    """manuscript option for creating factor selection plot
    """
    df_corr = df_merge.corr(numeric_only = True).round(2)
    coeff_corr = df_corr['coeff'].sort_values(ascending=False)
    coeff_corr.index.name = "factor"
    coeff_corr = coeff_corr.rename('corr_coeff')

    df_results_loocv = pd.read_csv("./gb_covariance/model_ays/kfold_models_svr.csv", index_col=0)
    boxplot = df_results_loocv.boxplot("cv_score","num_factors")
    plt.savefig(f"./gb_covariance/model_ays/{filename}_boxplot.png", dpi=300)
    df_factor_count = factor_percent_usage(df_results_loocv, 100, 'loocv_factor_usage')
    
    df = pd.merge(df_factor_count, coeff_corr, how="left", on='factor').set_index("factor").fillna(0)
    df = df[['count','corr_coeff']]
    df = df.rename(columns = {"corr_coeff":"Correlation\nCoefficient",
                              "count":"Usage\nTop 100 models"})
    
    df = df.sort_values("Correlation\nCoefficient", ascending=False)
    factor_select_plotting_3(df, label_dict, filename+"_corr", figsize = figsize)


def factor_select_plotting(df, label_dict, filename, width = 0.125): 
    """helper function, factor selection plot
    """
    sns.set_style("whitegrid")
    cols = df.columns

    x = np.arange(len(df))    
    multiplier = 0


    plt.rc('xtick', labelsize=16) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=16) #fontsize of the y tick labels

    fig, ax = plt.subplots(figsize=(16,10))
    for attribute in cols:
        offset = width * multiplier
        scaled_value = df[attribute]
        #scaled_value = (df[attribute]-min(df[attribute]))/(max(df[attribute])-min(df[attribute]))
        #scaled_value[scaled_value==0.0] = 0.01 # set small value for plotting
        ax.bar(x+offset,scaled_value,width, label=attribute)
        #ax.set_ylabel("$R^2$ improvement (percentage)")

        #axs[1].set_ylabel('Standardized Regression Coefficients (absolute)')
        multiplier += 1
    #ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
    x_labels =  [label_dict[x] for x in df.index.to_list()]
    ax.set_ylabel("Normalized Value\n(Value - Min Value)/(Max Value - Min Value)",fontsize=16)
    tick_loc = (len(cols)*width/2-width/2)
    ax.set_xticks(x + tick_loc, x_labels, rotation = 90)
    ax.legend(bbox_to_anchor = (0,1,1,1), loc="lower center", mode="expand", ncol = 4)
    ax.xaxis.set_minor_locator(ticker.FixedLocator(0.5+x+tick_loc))
    ax.xaxis.grid(visible=True, which="minor")

    #plt.show()
    fig.tight_layout()
    fig.savefig(f"./gb_covariance/model_ays/{filename}.png", dpi=300)


    return


def factor_select_plotting_2(df, label_dict, filename, width = 0.125, figsize = (16,10)): 
    """helper function, factor selection plot
    """
    sns.set_style("white")
    cols = df.columns

    x = np.arange(len(df))    

    plt.rc('xtick', labelsize=8) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=8) #fontsize of the y tick labels

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, df[cols[0]], width, label=cols[0], color='b')#color='#13294B')
    x_labels =  [label_dict[x] for x in df.index.to_list()]
    ax.set_ylabel(cols[0],fontsize=8)
    ax2 = ax.twinx()
    ax2.bar(x + width, df[cols[1]], width, label=cols[1], color='orange')#color='#FF5F05')
    ax2.set_ylabel(cols[1], fontsize=8)
    tick_loc = (len(cols)*width/2-width/2)
    ax.set_xticks(x + tick_loc, x_labels, rotation = 90)
    ax.xaxis.set_minor_locator(ticker.FixedLocator(0.5+x+tick_loc))
    #ax.xaxis.grid(visible=True, which="minor")

    fig.savefig(f"./gb_covariance/model_ays/{filename}.pdf", bbox_inches = 'tight')

    return


def factor_select_plotting_3(df, label_dict, filename, figsize = (6,4), fontsize = 8): 
    """helper function, factor selection plot
    
    two separate bar charts, one for corr coeff and one for usage
    """
    sns.set_style("white")
    cols = df.columns.to_list()
    props = [label_dict[i] for i in df.index.to_list()]
    color_code = ['tab:orange','tab:blue']

    fig, ax = plt.subplots(2,1, figsize=figsize, sharex = True)
    for i in range(len(cols)):
        ax[i].bar(x = props, height = df[cols[i]].values.tolist(), label=cols[i], color=color_code[i])#color='#13294B')
        ax[i].tick_params(axis='x',rotation=90)
        ax[i].xaxis.grid(visible=True, which="major")
        ax[i].set_ylabel(cols[i],fontsize=fontsize)
        ax[i].tick_params(labelsize=fontsize)
    #ax[1].yaxis.set_major_locator(ticker.FixedLocator([-1,0,1]))
    ax[1].set_yticks([-0.8,0,.8])
    ax[1].set_ylim(-1,1)
    ax[0].set_yticks([20,40,60])
    plt.subplots_adjust(hspace=0)
    fig.align_ylabels()
    fig.savefig(f"./images/main/{filename}.pdf", bbox_inches = 'tight')

    return


def top5_table(label_dict, sort_by):
    df_results = pd.read_csv(f"./gb_covariance/model_ays/kfold_models_svr.csv", index_col=0)
    df_results = df_results.sort_values(sort_by, ascending=False)
    score = df_results[sort_by].values.tolist()[:5]
    factor_list = [literal_eval(i) for i in df_results['factors'].iloc[0:5]]
    df5 = pd.DataFrame(factor_list)
    df5 = df5.replace(label_dict)
    df5['score'] = score
    df5.to_csv(f"images/main/kfold_svr_top5.csv")
    return


def main():
    df_in = pd.read_csv("./data/df_merge.csv", index_col=0)
    label_dict = data_exploration.import_label_dict()

    df_in = df_in.drop([i for i in df_in.columns if 'diamond' in i], axis=1) # ignore diamond
    df_in = df_in.sample(frac=1)  # shuffle

    params_list = ['lattice_constant',
                   'bulk_modulus', 'c11', 'c12', 'c44',
                   'cohesive_energy_fcc', 'thermal_expansion_coeff_fcc',
                   'surface_energy_100_fcc', # highly correlated with other surf energies
                   'surface_energy_110_bcc', # highly correlated with other surf energies
                   'extr_stack_fault_energy',
                   'intr_stack_fault_energy', #highly correlated with extr_stack_fault_energ
                   'unstable_stack_energy', 
                   'unstable_twinning_energy', #highly correlated with unstable stack energy
                   'relaxed_formation_potential_energy_fcc', #includes unrelaxed
                   'vacancy_migration_energy_fcc',
                   'relaxation_volume_fcc'
                   ]

    params_list_full = filter_param_list(df_in, params_list)
    df_in = basic_outlier_removal(df_in)
    X = df_in[params_list_full]
    y = df_in['coeff']

    imput = KNNImputer(n_neighbors=2, weights="uniform",
                       keep_empty_features=True)
    if False:
        # SVR selection pipeline (obsolete, can likely remove code)
        model = svm.SVR(kernel='rbf')

        pipe = Pipeline(steps=[('scale', StandardScaler()),
                               ('imp', imput),
                               ('lr', model)])
        pipe = TransformedTargetRegressor(regressor=pipe,
                                        transformer=StandardScaler())
        filename = 'kfold_models_svr'
        
    if True:
        # Linear regression selection pipeline, used in manuscript
        imput = KNNImputer(n_neighbors=2, weights="uniform",
                       keep_empty_features=True)
        model = linear_model.LinearRegression()

        pipe = Pipeline(steps=[('scale', StandardScaler()),
                               ('imp', imput),
                               ('lr', model)])
        pipe = TransformedTargetRegressor(regressor=pipe,
                                        transformer=StandardScaler())
        filename = "kfold_models_lr"

    n_factor_max = 3

    if True: #repeated K-fold cv selection, used in manuscript
        cv = RepeatedKFold(n_splits=10, n_repeats=5)
        df_results = factor_select_cv(
            X, y, pipe, n_factor_max=n_factor_max, cv=cv, scoring='neg_root_mean_squared_error')
        df_results.to_csv("./gb_covariance/model_ays/"+filename+".csv")
        factor_percent_usage(df_results, 100, filename)

    if True: #model selection plot
        # create_factor_select_plot_2(df_in, params_list_full, "factor_importance_2", label_dict)
        # create_factor_select_plot_3(df_in, params_list_full, "factor_importance_3", label_dict, figsize = (6,2))
        create_factor_select_plot_4(df_in, params_list_full, "factor_importance_manuscript", label_dict, figsize = (6,3)) #manuscript version
        top5_table(label_dict, 'cv_score')

    if False: #train/test split eval, not used
        cv = RepeatedKFold(n_splits=10, n_repeats=3)
        X_train, X_test, y_train, y_test = train_test_split(X,y, shuffle = True)
        
        n_factor_max = 1 #temporary for testing
        df_results_loocv = factor_select_cv(
            X_train, y_train, pipe, n_factor_max=n_factor_max, cv=cv, scoring='neg_root_mean_squared_error')
        factors = df_results_loocv.iloc[0]['factors']
        X_train = X_train[factors]
        X_test = X_test[factors]
        pipe.fit(X_train,y_train)
        y_pred_train = pipe.predict(X_train)
        y_pred_test = pipe.predict(X_test)
        title = f"Model performance, Train/Test Split\nfactors = {factors}"
        plotting.plot_pred_vs_actual(title,
                        y_train,
                        y_pred_train,
                        y_test, 
                        y_pred_test,
                        "train_test_model_eval",
                        gb_energy = True)
    return


if __name__ == "__main__":
    main()
