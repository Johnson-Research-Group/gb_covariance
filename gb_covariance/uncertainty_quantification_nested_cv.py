import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import models
from model_selection import basic_outlier_removal
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn import linear_model, svm
from sklearn.utils import resample
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats


def nested_cv_w_hyperparam_opt(df, 
                               X,
                               y,
                               pipe_in,
                               readme,
                               random_state):
    """perform SVR nested cv w/ hyperparameter optimization    
    
    ref: https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/
    """
    cv_outer = KFold(n_splits=10, 
                     shuffle = True,
                     random_state = random_state)

    # define search space
    space = dict()
    space['regressor__lr__C'] = [1,10,20,100,1000,10000] # strength of regularization inversely proportional to C
    space['regressor__lr__epsilon'] = [0.01,0.05,0.1,0.5] # epsilon-tube for no penalty

    y_actual, y_pred, y_index = [], [], []
    for train_ix, test_ix in cv_outer.split(X):
        X_train, X_test = X.iloc[train_ix,:], X.iloc[test_ix,:]
        y_train, y_test = y.iloc[train_ix],y.iloc[test_ix]
        cv_inner = KFold(n_splits=10, 
                         shuffle=True,
                         random_state = random_state)

        pipe = pipe_in
        search = GridSearchCV(pipe, space, scoring='r2', n_jobs=-1, cv=cv_inner, refit=True)
        result = search.fit(X_train, y_train)
        best_model = result.best_estimator_
        print(f"best model: C = {best_model.regressor.named_steps.lr.C}, Epsilon = {best_model.regressor.named_steps.lr.epsilon}")
        y_pred_value = best_model.predict(X_test).tolist()
        y_pred.extend(y_pred_value)
        y_actual.extend(y_test.to_list())
        y_index.extend(test_ix)

    df_pred = pd.DataFrame({'coeff_actual':y_actual,
                            'coeff_pred':y_pred},
                            index=y_index)

    df = df.merge(df_pred, left_index=True, right_index=True)
    return df, readme


def cv_lm(df, 
          params_list_in, 
          pipe_in, 
          filename,
          random_state,
          save = True):
    """perform cross-validation for linear regression model

    ref: https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/
    """

    X = df[params_list_in]
    y = df['coeff']

    cv = KFold(n_splits=10, shuffle=True, random_state = random_state)

    y_actual, y_pred, y_index = [], [], []
    for train_ix, test_ix in cv.split(X):
        X_train, X_test = X.iloc[train_ix,:], X.iloc[test_ix,:]
        y_train, y_test = y.iloc[train_ix],y.iloc[test_ix]
        pipe = pipe_in
        pipe.fit(X_train, y_train)
        
        y_pred_value = pipe.predict(X_test).tolist()
        y_pred.extend(y_pred_value)
        y_actual.extend(y_test.to_list())
        y_index.extend(test_ix)

    df_pred = pd.DataFrame({'coeff_actual':y_actual,
                            'coeff_pred':y_pred},
                            index=y_index)

    df = df.merge(df_pred, left_index=True, right_index=True)

    if save == True:
        df.to_csv(f"./gb_covariance/model_ays/{filename}_data.csv")
    
    return y_pred, y_actual


def r2_adj_fun(r2,n,k):
    """calculate adjusted r^2 
    """
    return (1 - ((1-r2)*(n-1)/(n-k-1)))


def call_nested_cv_plot(df, 
                        params_list_in, 
                        filename,
                        save_loc = "./gb_covariance/model_ays"):
    """prepare data, call for nested cv plot
    """
    X = df[params_list_in]
    y = df['coeff']
    y_pred = df['coeff_pred']

    r2 = r2_score(y, y_pred)
    k = len(X.columns)
    n = len(y)
    r2_adj = r2_adj_fun(r2, n, k)

    rmse = rmse_assessment(df,y,y_pred)
    nested_cv_plot(df, r2_adj, filename, rmse, save_loc = save_loc)
    return


def nested_cv_plot(df, 
                   r2_adj, 
                   filename, 
                   rmse_dict, 
                   figsize = (3,3),
                   save_loc = "./gb_covariance/model_ays"):
    y_pred = df['coeff_pred']
    y = df['coeff']
    df['species (RMSE)'] = [f"{i} ({rmse_dict[i]:.3f})" for i in df.species]
    error_bar = [2*rmse_dict[i] for i in df.species]
    sns.set_style("whitegrid")
    fig,ax = plt.subplots(figsize = (figsize[0],figsize[1]))
    ax.errorbar(y,y_pred, yerr=error_bar, fmt='.',markersize=0.001, alpha=0.12)
    sns.scatterplot(data=df, x='coeff',y='coeff_pred',hue='species (RMSE)',style = 'species (RMSE)')
    ax.plot(np.linspace(min(y),max(y),50),
           np.linspace(min(y),max(y),50))
    ax.set_xlabel(r"actual coefficient [$J/m^2$]", fontsize=8)
    ax.set_ylabel(r"predicted coefficient [$J/m^2$]", fontsize=8)
    ax.tick_params(labelsize=8)
    #ax.set_title()
    ax.text(0.5, 0.99, f"Adjusted r\N{SUPERSCRIPT TWO} = {r2_adj:.3f}",#\nError bar = 2x RMSE",
           verticalalignment='top', horizontalalignment='center',
           transform=ax.transAxes, fontsize=6)
    #fig = plt.figure()
    #fig.add_axes(p)
    #fig = ax.get_figure()
    ax.legend(bbox_to_anchor = (-0.2,1.02,1.2,.102),
              mode="expand",
             #bbox_transform = fig.transFigure)
              ncol = len(df['species'].drop_duplicates())/3,
              fontsize= 6)
             
    plt.savefig(f"{save_loc}/{filename}.pdf", bbox_inches = "tight")
    plt.close()


def perform_nested_cv(df, 
                      params_list_in, 
                      pipe, 
                      filename,
                      random_state):
    """prepare data, perform nested cv w/ hyperparameter optimization
    """
    readme = []
    readme.append(f"factor list: {params_list_in}\n")
    X = df[params_list_in]
    y = df['coeff']

    print(f"running {filename}")
    df, readme = nested_cv_w_hyperparam_opt(df, 
                                            X, 
                                            y, 
                                            pipe, 
                                            readme,
                                            random_state)

    with open(f"./gb_covariance/model_ays/{filename}_readme.txt", "w") as text_file:
        for line in readme:
            text_file.write(line)
    
    df.to_csv(f"./gb_covariance/model_ays/{filename}_data.csv")


def rmse_assessment(df_in,y,y_pred):
    """calculate RMSE 
    """
    rmse = {}
    species = df_in.species.to_list()
    df = pd.DataFrame({'species':species,
                       'y':y,
                       'y_pred':y_pred})
    species_list = df.species.drop_duplicates().to_list()
    for i in species_list:
        rmse[i] = (mean_squared_error(df[df.species == i]['y'], df[df.species == i]['y_pred']))**0.5
    return rmse

def main():
    # prepare data
    df_in = pd.read_csv("./data/df_merge.csv")
    df_in = df_in.drop([i for i in df_in.columns if 'diamond' in i],axis=1) # drop diamond

    df_in = df_in.drop(['thermal_expansion_coeff_bcc',
                        'surface_energy_100_bcc',
                        'surface_energy_110_bcc',
                        'surface_energy_111_bcc',
                        'surface_energy_121_bcc'], axis=1) # drop sparse properties
    
    df_in = basic_outlier_removal(df_in)

    # set parameters
    params_list = ['lattice_constant','bulk_modulus','c44','c11','c12',
                  'cohesive_energy','thermal_expansion_coeff','surface_energy_100',
                  'extr_stack_fault_energy','intr_stack_fault_energy','unstable_stack_energy',
                  'unstable_twinning_energy','relaxed_formation_potential_energy_fcc',
                  'unrelaxed_formation_potential_energy_fcc','relaxation_volume_fcc']
    
    imput = KNNImputer(n_neighbors=2, weights="uniform", keep_empty_features=True)

    params_3_factor = ['unstable_stack_energy_fcc', 'relaxed_formation_potential_energy_fcc', 'vacancy_migration_energy_fcc']


    if True:
        #SVR pipelines
        # create pipeline
        model = svm.SVR(kernel='rbf')
        pipe = Pipeline(steps=[('scale',StandardScaler()),
                                ('imp',imput),
                                ('lr',model)])
        pipe = TransformedTargetRegressor(regressor = pipe,
                                                transformer = StandardScaler())

        # full model, all parameters
        if True:
            params_list_full = models.filter_param_list(df_in, params_list)
            perform_nested_cv(df_in, 
                              params_list_full, 
                              pipe, 
                              'nested_cv_svr_all_props',
                              random_state = 12345)

        # 3 parameter model, not used in manuscript
        if False:
            perform_nested_cv(df_in, 
                              params_3_factor, 
                              pipe, 
                              'nested_cv_svr_3props',
                              random_state = 12345)

    if True: # linear model plot
        model2 = linear_model.LinearRegression()
        # poly = PolynomialFeatures(2, interaction_only=True)
        pipe = Pipeline(steps=[('imp',imput),
                               #('interaction',poly),
                               ('lr',model2)])
        pipe = TransformedTargetRegressor(regressor = pipe,
                                                transformer = None)
        
        name = 'cv_linear_3props'
        cv_lm(df_in, 
              params_3_factor, 
              pipe, 
              name,
              random_state = 12345)
        
        X = df_in[params_3_factor]
        y = df_in['coeff']
        pipe.fit(X,y)

        readme = []
        readme.append(f"factors: {pipe.regressor_.named_steps.imp.get_feature_names_out()}\n")
        readme.append(f"coeff: {pipe.regressor_.named_steps.lr.coef_}\n")
        readme.append(f"intercept: {pipe.regressor_.named_steps.lr.intercept_}\n")
        # readme.append(f"coeff combos: {pipe.regressor_.named_steps.interaction.get_feature_names_out()}\n")

        with open(f"./gb_covariance/model_ays/{name}_model.txt", "w") as text_file:
            for line in readme:
                text_file.write(line)

    if True: # plotting
        df_svr_all = pd.read_csv("gb_covariance/model_ays/nested_cv_svr_all_props_data.csv")
        params_list_full = models.filter_param_list(df_in, params_list)
        call_nested_cv_plot(df_svr_all, 
                            params_list_full, 
                            "nested_cv_svr_all_props",
                            save_loc = "images/main")
        
        df_linear = pd.read_csv("gb_covariance/model_ays/cv_linear_3props_data.csv")
        call_nested_cv_plot(df_linear, 
                            params_3_factor, 
                            "cv_linear_3props",
                            save_loc = "images/main")

if __name__ == "__main__":
    main()