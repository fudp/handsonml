# Chapter 2 - End to end ML Project

## Roadmap

- Write small functions for repetitive tasks. Even when the don't appear immediately repetitive.
- Different types of attributes (numerical and `object`).
- Capped values: two possible approaches:
  - Collect uncapped data.
  - Remove the capped values.
- Immediately after seeing what the issue in the data may be (through visualization and numerical inspection, set aside a test set).
- Hashing the IDs to avoid train-test set leakage.

## Random split vs stratified split

- Random split contains a `stratify` option.
- `StratifiedShuffleSplit` class allows doing the same.

`train_test_split` calls `ShuffleSplit` when `stratify = False` and `Stratifiedshufflesplit` otherwise.

```py
split = StratifiedShufflesplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test = housing.loc[test_index]
```

## Feature generation

Correlation plots showing "stripes" of constant median house value. Recommendation is to remove these districts to prevent your algorithms from learning to reproduce these data quirks.

In a regression problem one can "see" the quality of new features by the correlation with the target.

## Imputation

The `SimpleImputer` class from `sklearn.impute` is an easy starting point. The `fancyimpute` module has more sophisticated imputation methods.

For categorical variables, we may want to impute using the `most_frequent` strategy of `SimpleImputer`, but this requires first converting to numbers, for example with `OrdinalEncoder`. One simple way to impute missing values in categorical variables with the most frequent entry is the following.

```py
s = pd.Series(['a', 'a', 'b', np.NaN, 'a'])
s.fillna(s.mode().iloc[0])
```

`s.mode()` returns a Series with the most frequent entry. To isolate the value, we need to use integer indexing.

## Categorical Predictors

The `OrdinalEncoder` class allows transforming categorical predictors into integers. `OrdinalEncoder` is better than `LabelEncoder` because it is designed for input features and plays well with pipelines.

## One-Hot Encoding

Now you can use `sklearn.preprocessing.OneHotEncoder` to create one-hot encoded sparse matrices. You can directly pass a column of class `object`.

## Custom Transformers

You need to create a class that implements these three methods:
- `fit()` which simply returns itself.
- `transform()` that applies the actual transformation.
- `fit_transform()`

The last one is automatically created if the class inherits from `TransformerMixin`. If it also inherits from `BaseEstimator` and does not use `**args` and `**kargs`, we also automatically get `get_params()` and `set_params()`.

Adding boolean conditions on whether to include a certain predictor or not turns out to be useful when you run grid or random search.

## Feature Scaling

`MinMaxScaler` vs. `StandardScaler`.

### `MinMaxScaler`

z = (x - min(x)) / (max(x) - min(x))

### `StandardScaler`

z = (x - x.mean()) / (x.std())

## Pipelines

Pipelines are list of two-tuples, like the one below:

```py
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])
```

Note that we pass the classes, not the instances. Here, `CombinedAttributesAdder` is a custom transformer. All but the last estimator must be transformers, i.e., they must have a `fit_transform` method.

**TODO** create a pipeline where you use various imputation mechanisms (for example from `fancyimpute`). The name of the method is a parameter in the transformer class, and can be passed to the cross validation machinery (see below). This gives you an unbiased idea of the relative merits of different imputation mechanisms.

## The `ColumnTransformer` class

You can build a pipeline for the numeric and the categorical features, and then combine everything into one single pipeline with the `ColumnTransformer` class. Say we have a numeric pipeline `num_pipeline` and a categorical one, `cat_pipeline`. If `num_attribs` is a list (possibly containing a single element) of numerical attributes and `cat_attribs` is the same for categorical attributes, then we can write the following:

```py
full_pipeline ` ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', cat_pipeline, cat_attribs),
])
```

and we can call `full_pipeline.fit_transform()`. We can also use the names `'drop'` to remove unwanted columns and `'passthrough'` to leave columns untouched.

**TODO** Create a simple dataset containing numeric and categorical variables. Introduce missing values in both. How can you build a transformer to deal with the categorical missing values? How can you put everything into a pipeline?

## Select and train a model

The idea is selecting a small number of models (2-5) that seem to work well without spending time on hyperparameter optimization. You want to see whether the models are seriously underfitting or overfitting. This is where we introduce kfold cross validation and the `cross_val_score`. Remember that the higher the score, the better, this is why we use `neg_mean_squared_error` as the metric.

You should save the model, the hyperparameters, the scores and the predictions. You can use `pickle` or `sklearn.externals.joblib.dump` for this.

```py
from sklearn.externals import joblib

# To save the model
joblib.dump(my_model, 'my_model.pkl')

# To reload the model
my_model = joblib.load('my_model.pkl')
```

## Grid Search and Random Search

### Grid Search

You can combine multiple grid searches into one, as shown in the example. Assuming you have a `GridSearchCV` instance called `grid_search`, you can get the following attributes:

- `grid_search.best_estimator_`. If `refit=True` the model is retrained on the whole training set.
- `grid_search.cv_results_`
- `grid_search.best_params_`

As stated above, it is possible to treat some of the data preparation steps as hyperparameters. This is much more effective than the trial and error approach you normally use.

### Randomized Search

Better option when the search space is large.

## Feature Importance in Random Forest

You can obtain this information as follows:

```py
feature_importances = grid_search.best_estimator_.feature_importances_
```

## Evaluating on the test set

After re-running the whole pipeline on the test set by using `full_pipeline.transform(X_test)`, you can get the predictions from the model. This returns a point estimate, but a confidence interval may be more useful. One can use `scipy.stats.t.interval()` to obtain a confidence interval. Another possibility may be (but it's not in the book) a boostrap estimate.
