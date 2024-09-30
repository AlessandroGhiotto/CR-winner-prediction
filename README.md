# Clash Royale winner prediction

The aim of the project is to predict the winner of a Clash Royale match, given the two decks of the players, with non-neural machine learning methods using the Scikit Learn library.

## Description

**DATASET :** Clash Royale matches.

The dataset that I have chosen from kaggle is [Clash Royale S18 Ladder Datasets (37.9M matches)](https://www.kaggle.com/datasets/bwandowando/clash-royale-season-18-dec-0320-dataset). In particular I have used the first csv file, which is the one containing $2\,823\,527$ Season $18$ ladder matches of January $2021$ (_"battlesStaging_01012021_WL_tagged.csv"_).

**TASK :** Predict who is the winner of the match - Binary Classification.

On the dataset I have already done a first rough preprocessing, because the dataset on kaggle is rich in columns that are not useful for a classification task. In addition the two players are represented as _"winner"_ and _"loser"_, instead I will call them _"p0"_ for the first player, _"p1"_ for the second player and add the column _"winner"_ which will indicate the winner of the match. This _"winner"_ column that contain binary values is fundamental for our task of binary classification. Also I have added the missing values with the given functions in the nootebook _"dataset_noiser"_.

All the operations that I have done on the dataset can be seen in the notebook _"dataset_first_preprocessing.ipynb"_. The final result are the two csv files _"./data/ClashRoyaleDataset_corrupted.csv"_ and _"./data/ClashRoyaleDataset.csv"_ (for the version without NaN values). By maintaining only the main game mode we remain with $1\,815\,549$ matches. Each card is encoded with an id, the corresponding cards to this ids are written in the file _"CardMasterListSeason18_12082020.csv"_, which was given at the same link in kaggle.

### Dataset analysis

1. analize the missing values
2. look at the distributions of our features

### Data transformation Pipeline

1. eliminate the rows with too many missing values
2. define a <code>ColumnTransformer</code> for handling all the preprocessing that is needed
3. split the dataset into train and test

Here we can see the correlation between trophies, elixir average and the sum of the level of the cards:

![hexbinplot](https://github.com/AlessandroGhiotto/CNN-image-classification/blob/main/images/hebinplot.png)

Now for the true clash royale gamers we can look at correlation matrix given by the co-occurences of the cards in the **2.8 deck** ( _Hog_Rider, Musketeer, Ice_Golem, Cannon, Fireball, The_Log, Ice_Spirit, Skeletons_ ):

![2dot8](https://github.com/AlessandroGhiotto/CNN-image-classification/blob/main/images/2dot8-correlation.png)

### Model selection

I will perform a nested cross validation with five outer and two inner folds **(** $5 \times 2$ **cross-validation)**, in Halving Random Search settings.

Dimensionality reduction configurations:

```python
dim_reduction_configs = [
    {
        'dim_reduction': [None]
    },
    {
        'dim_reduction': [PCA()],
        'dim_reduction__n_components': uniform(loc=0.1, scale=0.6) # Unif[0.1, 0.7]
    },
    {
        'dim_reduction': [LinearDiscriminantAnalysis()]
    },
    {
        'dim_reduction': [SequentialFeatureSelector(estimator=LogisticRegression)],
        'dim_reduction__k_features': randint(3, 16) # [3, 15]
    },
]
```

Classifier configurations:

```python
classifier_configs = [
    {
        'classifier': [Perceptron(max_iter=100)],
        'classifier__eta0' : loguniform(0.01, 1000),
        'classifier__class_weight' : [None, 'balanced'],

    },
    {
        'classifier': [LogisticRegression(max_iter=100, solver='saga')],
        'classifier__C' : loguniform(0.01, 1000),
        'classifier__penalty': ['l1','l2'],
        'classifier__class_weight' : [None, 'balanced']

    },
    {
        'classifier' : [RandomForestClassifier(n_estimators=100)],
        'classifier__max_depth' : [None, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    },
    {
        'classifier' : [XGBClassifier(n_estimators=100)],
        'classifier__max_depth' : [2, 3, 4]
    },
]
```

Then I have done a more focused Randomized Search CV around the solution found in the previous nestes cross validation. I have obtained the following model:

![model](https://github.com/AlessandroGhiotto/CNN-image-classification/blob/main/images/FinalModelPipeline.png)

## Installing

- Download _"battlesStaging_01012021_WL_tagged.csv"_ from [Clash Royale S18 Ladder Datasets (37.9M matches)](https://www.kaggle.com/datasets/bwandowando/clash-royale-season-18-dec-0320-dataset) and put it in the _"data/"_ folder.

- Unzip _"data/ClashRoyaleDataset_corrupted.zip"_ and move _"ClashRoyaleDataset_corrupted.csv"_ in the _"data/"_ folder.

## Acknowledgments

Thanks to BwandoWando for the dataset [Clash Royale S18 Ladder Datasets (37.9M matches)](https://www.kaggle.com/datasets/bwandowando/clash-royale-season-18-dec-0320-dataset)
