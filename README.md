# Recipe Reviews Model

### By Heidi Tam and Kyla Park

---

### Framing the Problem

We will build a **regression model** to predict the popularity of each recipe.
In this project, we will define more popular recipes as those that contain a **greater
number of reviews**. We will use a regression model since our response variable,
the number of reviews, is a count variable, which is a type of discrete variable.
This means the number of reviews is not restricted by a finite number of categories.
Therefore, it is more appropriate to use a regression model over a classification model.

- **Response Variable**: `review_count` (the total number of reviews that
  users left for each recipe)
  - We chose `review_count` since this represents the number of users who tried
    a particular recipe and left a review, so we would expect it to be directly positively
    correlated with the recipe's popularity.
- **Evaluation Metric:** RMSE (root mean squared error)
  - RMSE is **sensitive to magnitude**, which means larger errors are penalized more heavily than small errors. This
    trait is usually desirable in regression models.
  - By squaring our errors before taking the mean, we ensure that errors in the positive or negative
    direction don't cancel each other out.
  - RMSE tells us whether the model is overfitting or well-generalized.
- **Evaluation Metric in Comparison to Other Metrics**:
  - Since we are assessing how well a model fits the dataset, we should RMSE since it has the same units as the target variable, in our case, `review_count`. This makes the performance of RMSE easy to interpret. On the other hand, MSE (mean squared error) is measured in squared units of the response variable.
- **Information Known**: At the time of prediction, we would already know `n_steps`,
  `n_ingredients`, and `minutes` since in order for people to try the recipe and leave a review, we need to have made the recipe first. This implies that `n_steps`, `n_ingredients`, and `minutes` already existed at the time the `review_counts` column was created.

### Baseline Model

For the baseline model, we used **linear regression.**

**Type of Features**

- Quantitative features: `minutes` , `n_steps`, `n_ingredients`, `rating`, `tag_count`
- Categorical (nominal) features: `date`

**Necessary Encodings**

- Ignoring Outliers: We used quantile to set the lower bound and upper bound for `review_count`. We got rid of `review_count` values if the values don't lie within the boundaries.

- Selecting Features for `train_test_split`: We set y to `review_count` column and X to datframe with all the columns other than `review_count` column. With X and y, we got the values for `X_train`, `X_test`, `y_train`, and `y_test` through `train_test_split`.

- Preprocessor: We used a single `ColumnTransformer` to perform different transformations on several different features.

  - `StandardScaler()` on quantitative features: Standardization scales the numerical features by dividing them with their standard deviation, ensuring that all features have the same scale. Since `minutes` feature has relatively large numbers compared to other features, we usd standardization to prevent it from dominating the statistical power of algorithm.
  - `QuantileTransformer` on `date` feature: Beforehand, we created `days_since_posted` column, where we converted `date` to numerical value in data cleaning part. Then, we used `QuantileTransformer` on `days_since_posted` column to uniformize data, aiming to make it easier for the model to analyze the patterns.

- Pipeline: We performed feature engineering with preprocessor and training/prediction with `LinearRegression()` within a single object, `pipeline`.

  - We fit the pipeline, predicted with `X_test`, and got root mean squared error for test sets.

**Performance of the Model**

4.670457208006444

We believe that our current model is not good. The goodness of The minimum value of `review_count` is 0 and the maximum value of `review_count` is 325. Based on the scale of predicting value `review_count`, RMSE of 4.670457208006444 is relatively large, which means that our current model should be improved.

### Final Model

### Fairness Analysis
