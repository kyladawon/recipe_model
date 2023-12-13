# Investigating Recipe Reviews

### By Heidi Tam and Kyla Park

---

### Framing the Problem
We will build a **regression model** to predict the popularity of each recipe. 
In this project, we will define more popular recipes as those that contain a **greater
number of reviews**. We will use a regression model since our response variable, 
the number of reviews, is a count variable, which is a type of discrete variable.
This means the number of reviews is not restricted by a finite number of categories.
Therefore, it is more appropriate to use a regression model over a classification model.
* **Response Variable**: ```review_count``` (the total number of reviews that 
users left for each recipe)
  * We chose ```review_count``` since this represents the number of users who tried
  a particular recipe and left a review, so we would expect it to be directly positively
  correlated with the recipe's popularity. 
* **Evaluation Metric:** RMSE (root mean squared error)
  * RMSE is **sensitive to magnitude**, which means larger errors are penalized more heavily than small errors. This 
  trait is usually desirable in regression models.
  * By squaring our errors before taking the mean, we ensure that errors in the positive or negative
  direction don't cancel each other out. 
  * RMSE tells us whether the model is overfitting or well-generalized.
* **Evaluation Metric in Comparison to Other Metrics**:
  * Since we are assessing how well a model fits the dataset, we should RMSE since it has the same
  units as the target variable, in our case, ```review_count```. This makes the performance
  of RMSE easy to interpret. On the other hand, MSE (mean squared error) is measured in squared units of
  the response variable. 
* **Information Known**: At the time of prediction, we would already know ```n_steps```,
```n_ingredients```, and ```minutes``` since in order for people to try the recipe and leave a review, we need to have 
made the recipe first. This implies that ```n_steps```, ```n_ingredients```, and ```minutes``` already existed at the 
time the ```review_counts``` column was created. 

### Baseline Model.
* **Model Description**:  

### Final Model

### Fairness Analysis
