Fundamental Microscopic Properties as Predictors of Large-Scale Quantities of Interest: Validation through Grain Boundary Energy Trends
==============
***Supporting data and code for (https://arxiv.org/abs/2411.16770)***

**Authors**: *Benjamin A. Jasperson, Ilia Nikiforov, Amit Samanta,
        Brandon Runnels, Harley T. Johnson, Ellad B. Tadmor*

# Installation
1. Create a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) using the attached yml file.
2. Activate conda environment.
3. Install [wield](https://github.com/solidsgroup/wield) per the instructions. This research utilized the 51f96b4 commit, which should install in the conda environment provided.


# Descriptions
Dataset and script descriptions.

## Datasets
- **df_analytical.csv**: lattice-matching model GB curves.
- **df_md_avg.csv**: averages of IP-generated GB curves and property data from OpenKIM. 
- **df_merge_all.csv**: combined dataset of OpenKIM IP and analytical results, with outlier removal and filtering applied (see data_import.py for workflow). Includes scaling coefficient.
- **df_merge_raw.csv**: raw dataset from OpenKIM, prior to data_import workflow.
- **df_merge.csv**: subset of df_merge_all, filtered by tilt axis and duplicates removed.
- **dft.csv**: dft dataset.
- **gb_dft_combined.csv**: merge individual lines from gb_dft.csv so they are grouped by species and tilt axis.
- **gb_dft.csv**: original GB DFT results, from [Crystalium database](http://crystalium.materialsvirtuallab.org/).
- **prop_table.csv**: property descriptions for manuscript.
- **stats.csv**: database stats. 

## Scripts
- **bibfile_create.py**: used to generate bibfile/citations for all IPs used.
- **data_exploration.py**: plotting and analysis for manuscript/supplemental. Includes GB plots for manuscript/supplemental (Fig 1), correlation heatmap w/ barchart (Fig 2), pairplots (Figs 3 and 7).
- **data_import.py**: combine OpenKIM property data, analytical results, and grain boundary data. Save combined results for further analysis. 
- **dft_import.py**: import and analysis of DFT results and predictions (Fig 6) 
- **generate_analytical.py**: produce GB lattice matching interatomic energy model
- **model_selection.py**: model evaluation using k-fold CV. Factor importance figure for manuscript (Fig 4). 
- **models.py**: various helper functions
- **openKimInterface.py**: helper code to extract property data from OpenKIM database.
- **plotting.py**: plotting code
- **uncertainty_quantification_nested_cv**.py: k-fold cross-validation, linear and SVR models (Fig 5)



