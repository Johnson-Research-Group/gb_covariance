import statsmodels.api as sm
import numpy as np
from sklearn  import linear_model, svm, metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler, scale
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import KNNImputer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import wield
import pickle
from itertools import combinations
from ast import literal_eval


plt.rcParams['figure.dpi'] = 400

def get_analytical_ground(labels, tolerance = 1e-8, order = 8, use_model_props = False):
    species = labels['species']
    crystal_type = labels['crystal_type']
    
    desc = species + '_' + crystal_type
    table = {'Ag_fcc':{'nondim_temp':0.175,'a':4.082,'e':0.5}, #assume nt based on past results, average openKIM value for a, default e 
             'Al_fcc':{'nondim_temp':0.175,'a':4.046,'e':0.5}, #assume nt based on past results, default e (was 0.175 b4 runnels meeting)
             'Au_fcc':{'nondim_temp':0.175,'a':4.065,'e':0.5}, 
             'Cu_fcc':{'nondim_temp':0.175,'a':3.615,'e':0.5},
             'Fe_bcc':{'nondim_temp':0.19,'a':2.856,'e':0.25}, 
             'Fe_fcc':{'nondim_temp':0.175,'a':2.856,'e':0.5}, #assume nt based on past results
             'Mo_bcc':{'nondim_temp':0.19,'a':3.15,'e':0.25}, #average of ref data in openKIM for a
             'Ni_fcc':{'nondim_temp':0.175,'a':3.499,'e':0.5}, #assume nt based on past results, default e
             'Pb_fcc':{'nondim_temp':0.175,'a':4.95,'e':0.5}, #assume nt based on past results, default e, openKIM a 
             'Pd_fcc':{'nondim_temp':0.175,'a':3.89,'e':0.5}, #assume nt based on past results, default e, openKIM a 
             'Pt_fcc':{'nondim_temp':0.175,'a':3.92,'e':0.5}, #assume nt based on past results, default e, openKIM a 
             'Rh_fcc':{'nondim_temp':0.175,'a':3.9,'e':0.5}, #assume nt based on past results, default e, openKIM a 
            }
    nt = table[desc]['nondim_temp']
    if use_model_props == True:
        a = labels['lattice_constant_' + crystal_type]
    else:
        a = table[desc]['a']
    eps = table[desc]['e'] #a "material parameter"

    sigma = a*nt
    
    if crystal_type == 'fcc': #for reason why, see Runnels thesis, pg 39, Prop 3.1, and Apendix A
        X = [0,   0,    0,    0,    0, a/2, -a/2,  a/2, -a/2, a/2,  a/2, -a/2, -a/2]
        Y = [0, a/2,  a/2, -a/2, -a/2,   0,    0,    0,    0, a/2, -a/2,  a/2, -a/2]
        Z = [0, a/2, -a/2,  a/2, -a/2, a/2,  a/2, -a/2, -a/2,   0,    0,    0,    0]

        Rground = wield.createMatrixFromZX([1,1,1],[-1,1,0])

    if crystal_type == 'bcc': 
        X = [a/2,  0,    0,    0,    0, -a/2]
        Y = [0,  a/2,  a/2, -a/2, -a/2,    0]
        Z = [0,  a/2, -a/2,  a/2, -a/2,    0]

        Rground = wield.createMatrixFromZX([1,1,0],[-1,1,0])

    C1 = wield.CrystalGD(order, a, a, a, sigma, X, Y, Z, 1, True)
    C2 = C1
    
    ground  = wield.SurfaceGD(C1,Rground,C1,Rground,eps,tolerance)

    return C1,C2,ground,eps

def get_analytical_XZ(tilt_axis):
    """adjust XZ based on tilt axis
    """
    if tilt_axis == "[0, 0, 1]":
        Z = [1,1,0]
        X = [0,0,1]

    elif tilt_axis == "[1, 1, 0]":
        Z = [0,0,1] 
        X = [1,1,0]

    elif tilt_axis == "[1, 1, 2]":
        Z = [1,-1,0]
        X = [1,1,2]

    elif tilt_axis == "[1, 1, 1]":
        Z = [1,1,-2]
        X = [1,1,1]

    else:
        raise Exception("tilt axis not defined")
    
    return X,Z


def analytical_model(labels, tolerance, use_model_props = False):
    """unrelaxed analytical model
    """
    
    tilt_axis = labels['tilt_axis']

    C1,C2,ground,eps = get_analytical_ground(labels,
                                             tolerance = tolerance,
                                             use_model_props = use_model_props)

    X,Z = get_analytical_XZ(tilt_axis)   
    
    R1 = wield.createMatrixFromZX(Z,X)
    R2 = wield.createMatrixFromZX(Z,X)

    energies_unrelaxed = []

    angles = labels['angle']
    for angle in angles:
        Rtheta1 = wield.createMatrixFromXAngle(angle/2)
        Rtheta2 = wield.createMatrixFromXAngle(-angle/2)
        energy_unrelaxed  = 1.0 - wield.SurfaceGD(C1,Rtheta1 @ R1,C2,Rtheta2 @ R2,eps,tolerance)/ground
        energies_unrelaxed.append(energy_unrelaxed)
    return tuple(energies_unrelaxed)


def coeff_determ(X,y):
    """takes in MD energies and analytical model est, determine c
       for a given set of data (one species, tilt axis, crystal_type)
    """
    #reg = linear_model.LinearRegression().fit(X,y)
    #reg = linear_model.Ridge(alpha = 0.25).fit(X,y)   
    c = X.T@y/(X.T@X)
    c = c.reshape(-1)[0]
    return c #reg.coef_


def add_analytical(df, tolerance, use_model_props = False):
    df_analytical = df.copy()

    if use_model_props == False:
        df_analytical = df_analytical[['species','tilt_axis', 'crystal_type','angle']].drop_duplicates(subset=['species','tilt_axis', 'crystal_type'])
    
    analytical_grain_energy = []
    
    for i in range(len(df_analytical)):
        print(f"adding analytical model: {i+1} of {len(df_analytical)}")
        labels = df_analytical.iloc[i].to_dict()
        analytical_grain_energy.append(analytical_model(labels,
                                                        tolerance = tolerance,
                                                        use_model_props = use_model_props))
        
    df_analytical['grain_energy_analytical'] = analytical_grain_energy

    if use_model_props == False:
        df_analytical = df_analytical.drop('angle',axis=1)
        df_analytical = pd.merge(df_analytical, df, how='outer',on=['species','tilt_axis', 'crystal_type'])

    return df_analytical


def get_analytical_coeff(df):
    """create new dataframe of species, model, coefficent
       merge with existing dataframe, return
    """

    df_coeff = df[['species','model']].drop_duplicates()
    coef = []
    for i in range(len(df_coeff)):
        #for each set of data, send list of grain energies to get coefficient
        current_combo = df_coeff.iloc[i,:]
        current_species = current_combo['species']
        current_model = current_combo['model']
        current_lines = df[(df['species'] == current_species) & (df['model'] == current_model)]
        
        # next, need to concatenate the X and y together before doing coeff_determ
        y,X = [],[]
        for j in range(len(current_lines)):
            y.extend(current_lines.iloc[j]['grain_energy'])
            if isinstance(current_lines.iloc[j]['grain_energy_analytical'],str):
                X.extend(literal_eval(current_lines.iloc[j]['grain_energy_analytical']))
            elif isinstance(current_lines.iloc[j]['grain_energy_analytical'],list):
                X.extend(current_lines.iloc[j]['grain_energy_analytical'])
        y = np.array(y)
        X = np.array(X).reshape(-1,1)
        
        coef.append(coeff_determ(X, y))
    df_coeff['coeff'] = coef
    df = df.merge(df_coeff,
                  how = 'left',
                  on = ['species','model'])
    return df


def analytical_w_coeff(df, tolerance, use_model_props = False):
    """
    adds the analytical model and (exact) coefficient to df
    """
    
    df_analytical = add_analytical(df, 
                                   tolerance = tolerance,
                                   use_model_props = use_model_props)
    df_analytical = get_analytical_coeff(df_analytical)
    
    return df_analytical


def filter_param_list(df, base_labels, specific_items=""):
    """generate filtered list of parameters
        
    :param df pd.DataFrame: dataframe with columns to consider
    :param base_labels list: base label strings to consider
    :param specific_items list: specific strings to include
    :return: list of specific label strings of parameters 
    """
    params_list_full = []
    params_list_full.extend(specific_items)

    for i in base_labels:
        current_list = [j for j in df.columns if i in j]
        params_list_full.extend(current_list)

    return params_list_full


def prep_model_data(df, params, y_label = 'coeff'):
    """split up into X and y for model
    """
    if y_label in df.columns:
        y = df[y_label]
    else:
        y = []
    X = df[[i for i in params if i != y_label]]
    return X,y


def get_avg_gb_energy(df, df_md_avg):
    """takes in full MD df; averages gb energy; adds to df_md_avg
    """
    avg_gb = []
    for i in range(len(df_md_avg)):
        species = df_md_avg.iloc[i].species
        tilt_axis = df_md_avg.iloc[i].tilt_axis
        crystal_type = df_md_avg.iloc[i].crystal_type

        df_filtered = df[(df.species == species) & 
                        (df.tilt_axis == tilt_axis) & 
                        (df.crystal_type == crystal_type)]
        df_current = pd.DataFrame(df_filtered.grain_energy.to_list())
        avg_gb.append(df_current.mean().to_list())
    df_md_avg['grain_energy_mean'] = avg_gb
    #uses species, tilt_axis, crystal_type
    return df_md_avg


def num_feature_plot(pipe, X, y, cv = 'default', num_features = 'default'):
    if cv == 'default':
        cv = RepeatedKFold(n_splits = 10, n_repeats = 3, random_state = 1)
    if num_features == 'default':    
        num_features = np.arange(2, X.shape[1]+1)
    results = []
    for k in num_features:
        pipe.regressor['sel'].k = k  #set k to generate plot
        scores = cross_val_score(pipe, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs = -1)
        results.append(scores)

    plt.boxplot(results, labels=num_features, showmeans=True)
    plt.ylabel('neg mean absolute error')
    plt.xlabel('num features')
    plt.show()


def num_feature_select(pipe, X, cv = 'default', grid = 'default'):
    """sets up an instance of GridSearchCV, to be used with search.fit
       find the optimal number of features using CV for each feature number
       grid search from 2 to num_params, with cross-validation, to get best model
    """
    
    if cv == 'default':
        cv = RepeatedKFold(n_splits = 10, n_repeats = 3, random_state = 1)
    if grid == 'default':
        grid = dict()
        grid['regressor__sel__k'] = [i for i in range(2, X.shape[1]+1)]
    if grid == 'default_w_knn':
        grid = dict()
        grid['regressor__sel__k'] = [i for i in range(2, X.shape[1]+1)]
        #add step here to determine n_neighbors for KNNImputer
    search = GridSearchCV(pipe, 
                          grid, 
                          scoring = 'neg_mean_squared_error', 
                          n_jobs = -1, 
                          cv = cv)
    
    return search


def estimator_factor_influence_base(pipe, X, y, plot = False):
    """early attempt to estimate factor influence. plots influence of each factor
    """

    #get list of factors
    factors = list(X.columns)
    r2_increase_perc = []
    for factor in factors:
        orig_factors = factors.copy()
        orig_factors.remove(factor)
        pipe.fit(X[orig_factors], y)
        y_orig = pipe.predict(X[orig_factors])
        r2_orig = metrics.r2_score(y_orig, y)
        pipe.fit(X, y)
        y_full = pipe.predict(X)
        r2_full = metrics.r2_score(y_full, y)
        r2_increase_perc.append(100*(r2_full-r2_orig))#/r2_full) #don't think I should normalize
    if plot == True:
        df = pd.DataFrame({'factors':factors,
                           'r2_increase_perc':r2_increase_perc})
        df = df.sort_values('r2_increase_perc', ascending=False)
        plt.bar('factors', 'r2_increase_perc', data=df)
        plt.xticks(rotation=90, ha = "right")
        plt.ylabel(f"Increase in R-squared %")
        plt.show()    

    return r2_increase_perc


def estimator_factor_influence(pipe, X_train, y_train, X_test, y_test, plot = False):
    """early attempt to estimate factor influence. plots influence of each factor
    """

    #get list of factors
    factors = list(X_train.columns)
    r2_increase_perc = []
    for factor in factors:
        orig_factors = factors.copy()
        orig_factors.remove(factor)
        pipe.fit(X_train[orig_factors], y_train)
        y_orig = pipe.predict(X_test[orig_factors])
        r2_orig = metrics.r2_score(y_orig, y_test)
        pipe.fit(X_train, y_train)
        y_full = pipe.predict(X_test)
        r2_full = metrics.r2_score(y_full, y_test)
        r2_increase_perc.append(100*(r2_full-r2_orig))#/r2_orig) #don't think I should normalize
    if plot == True:
        plt.bar(factors, r2_increase_perc)
        plt.xticks(rotation=20, ha = "right")
        plt.ylabel(f"Increase in R-squared %")
        plt.show()    

    return r2_increase_perc


def estimator_factor_influence_average(pipe, X, y, n_iter = 10, test_size_in = 0.2):
    perc_influence_list = []
    for i in range(n_iter):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size_in)
        pipe.fit(X_train, y_train)
        perc_influence = estimator_factor_influence(pipe, X_train, y_train, X_test, y_test)
        perc_influence_list.append(perc_influence)

    df_influence = pd.DataFrame(perc_influence_list, columns = X_train.columns)
    r2_increase_perc = df_influence.median(axis=0)
    plt.bar(list(X_train.columns), r2_increase_perc)
    plt.grid()
    plt.xticks(rotation=90, ha = "right")
    plt.ylabel(f"Increase in R-squared %")
    plt.show()  

    return r2_increase_perc


def extract_interaction_coeff_names(pipe):
    interaction_out = pipe.best_estimator_.regressor_.named_steps.interaction.get_feature_names_out()
    sel_map = {f"x{i}":f"{j}" for i,j in enumerate(interaction_out)}

    features_in = pipe.best_estimator_.regressor_.named_steps.scale.feature_names_in_
    #features_in = pipe_out.regressor_.named_steps.scale.feature_names_in_
    feature_map_base = {f"x{i}":j for i,j in enumerate(features_in)}
    feature_map_inter = {f"x{i} x{k}":f"{j} + {l}" for i,j in enumerate(features_in) for k,l in enumerate(features_in)}
    for i,j in enumerate(features_in):
        feature_map_inter[f"x{i}^2"] = f"({j})^2"
    feature_map = {**feature_map_base, **feature_map_inter}
    features_out = pipe.best_estimator_.regressor_.named_steps.sel.get_feature_names_out()
    #features_out = pipe_out.regressor_.named_steps.interaction.get_feature_names_out()
    features_out = [sel_map[i] for i in features_out]
    features_out = [feature_map[i] for i in features_out]
    coeff = pipe.best_estimator_.regressor_.named_steps.lr.coef_
    return features_out, coeff


def r2_adj(r2, n, k):
    return (1 - ((1-r2)*(n-1)/(n-k-1)))

 
def param_select_cv(X, y, pipe, n_factor_max=2, cv=5):
    """return list of parameters w/ cv score
    """
    
    factor_list = X.columns.to_list()
    subsets = []
    for n in range(1,(n_factor_max+1)):
        for subset in combinations(factor_list, n):
            subsets.append(list(subset))

    cv_score_mean = []
    cv_score_std = []
    for subset in subsets:
        print('current subset: ',subset)
        score = cross_val_score(pipe, X[subset], y, cv=cv)
        print('score mean: ',np.mean(score))
        cv_score_mean.append(np.mean(score))
        cv_score_std.append(np.std(score))
    
    df_results = pd.DataFrame({'factors':subsets,
                       'cv_score':cv_score_mean,
                       'cv_score_std':cv_score_std})
    df_results = df_results.sort_values('cv_score', ascending=False)

    return df_results


def linear_model_create(scale = True):
    imput = KNNImputer(n_neighbors=2, weights="uniform",
                       keep_empty_features=True)
    model = linear_model.LinearRegression()

    if scale == True:
        pipe = Pipeline(steps=[('scale', StandardScaler()),
                               ('imp', imput),
                               ('lr', model)])
    elif scale == False:
        pipe = Pipeline(steps=[('imp', imput),
                               ('lr', model)])

    return pipe