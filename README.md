atom3d
==============================

predicting protein flexibility for better ligand docking

Steps
------------

1. src/data_analysis/dock_check.py - sets index for dataset
2. src/data/add_basic_files.py - adds protein structures and ground truth ligand structures
3. src/data/lig_extractor.py - adds glide ligand poses
4. src/data/translation_stats.py - determines mean and average for translation 
5. src/data/decoy_creator.py - creates translated and rotated structures
6. src/data/data_converter.py - converts protein mae files to pdb files and ligand mae files to sdf files
7. src/data/process_pdbbind.py - finds binding pockets
8. src/data/decoy_rmsd.py - 1st step to find rmsd of each ligand pose
9. src/data/get_labels.py - 2nd step to find rmsd of each ligand pose
10. src/data/mcss_similarity.py - finding mcss similarity for ligand pairs
11. src/models/pdbbind_dataloader.py - create graphs and data files
12. src/models/split_file.py - create train, val, and test splits
13. src/models/train_pdbbind.py - train model

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │                     predictions
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   |   └── visualize.py
    │   │  
    │   └── utils  <- Files with functions used in data creation and training
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
