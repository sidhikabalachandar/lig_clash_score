atom3d
==============================

predicting protein flexibility for better ligand docking

Steps
------------
Create decoys + get mcss:
1. src/data_analysis/decoy.py task='all'

Get physics score and RMSD
2. src/data/grid.py task='run' 
3. src/data/score_and_rmsd.py task='run'
4. src/data/get_csv.py task = 'all' - create csv with mcss, physics score, and rmsd data for each decoy

Convert protein mae files to pdb files and ligand mae files to sdf files 
5. src/data/data_converter.py task='run'

Create graphs
6. src/models/pdbbind_dataloader.py task='all' - create graphs
7. src/models/pdbbind_dataloader.py task='combine_all' - create data files

Train
8. src/models/split_file.py - create train, val, and test splits
9. src/models/train_pdbbind.py - train model

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
