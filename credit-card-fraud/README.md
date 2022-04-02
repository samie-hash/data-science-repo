credit-card-fraud
==============================

Context 
==============================
It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase. 

Brief overview
==============================
#### Data Collection
The dataset was collected from kaggle as a csv file and stored in `data/raw/`

#### Data Preparation
During the preparation stage, the columns with the highest correlation with the target variable `Class` were selected 
and used for the modeling. 

#### Model Building
This stage involves the actual model building. We considered an unsupervised learning approach due to the following reasons
- Imbalanced dataset
- Fraudster might change their pattern and this will require retraining the model to learn the new patterns

A quantile based anomaly detection algorithm uses a box plot which flags data points outside the upper and lower whiskers as outliers. This model is relatively fast but suffers performance wise. Some of the challenges of this model include

- Outliers in the data expand the quantiles
- Strictly one dimensional, we cannot use two variables to train the model as such the variable with the highest correlation with the target is used.
- Skewed data might requires different values of k to detect upper and lower outliers

#### Insights
The quantile based model is relatively simple and it is obvious that it flags values between -5 and 5 as normal transaction, every other data points are classified as fraudulent.

#### Future Work
- Create a distribution based model (Z-score) for classifying fraudulent transactions and normal transactions
- Create a DBSCAN clustering based model to segment fraudulent transactions and normal transactions
- Use autoencoders to detect fraudulent transactions and normal transactions

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
    ├── notebooks          <- Jupyter notebooks.
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
    │   ├── models         <- Scripts to train models
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

Tools Used
==============================
Python==3.8.3
scikit-learn==1.0.2
pandas==1.4.1
numpy==1.22.3
matplotlib==3.5.1
matplotlib-inline==0.1.3
jupyter-client==7.1.2
jupyter-core==4.9.2
gradio==2.9.1

How to use the application
==============================

Step 1: Clone the repository

Step 2: Open the project folder in vscode or navigate to the project respository from command line i.e cd Desktop/data-science-repo/credit-card-fraud

Step 3: Install the project dependencies by running the following command `pip install -r requirements.txt`

If you want to retrain the model follow this steps
    i. Navigate to `cd src/data/`
    ii. Run the following command `python make_dataset.py ../../data/raw/creditcard.csv ../../data/processed/processed.csv` which would create the processed file ready for training. NOTE: ensure you download the data from [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and copy the file to `data/raw/`
    iii. Navigate to by using `cd ..` and `cd src/models`
    iv. Run the following command `python train_model.py`
    v. Wait for the model to finish training

Step 4: navigate into app by using `cd app/`

Step 5: run `python main.py`

This should start up gradio.io; copy the link to a browser. Have Fun!!!

<p><small>Project based created by Samuel I. Oseh</small></p>
