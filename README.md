# The Great Recipe Rating Race: Using Machine Learning to Help You Cook Like a Pro

## Introduction

Welcome to our model building project, a project focused on predicting ratings of recipes. The goal of this project is to develop a machine learning model that can accurately predict the ratings of recipes based on various features such as ingredients, cooking time, and complexity. This is a regression problem, as the predicted variable (ratings) is a continuous numerical variable.

Our response variable is the recipe rating, which ranges from 1 to 5. We chose this variable because it is a crucial factor for determining the success and popularity of a recipe. Our model will assist users in finding recipes that are likely to be highly rated, leading to a more satisfying cooking experience.

The dataset used for this analysis is collected from <a href="food.com">food.com</a>. and contains information on a variety of recipes, including their ingredients, cooking times, and user ratings. By exploring this data, we hope to gain insights into the factors that contribute to a recipe’s success and popularity.

We hope that this project will help food enthusiasts and cooking enthusiasts discover new and highly-rated recipes with ease.

Check out our exploratory data analysis on the same dataset <a href="https://adityaarun2.github.io/exploratory-data-analysis-project/" target="_blank">here</a>!

<center><img src="cook.jpeg" alt="Picture of food getting cooked" height="300" width="600"></center>

---

## Framing the Problem

The prediction problem we are going to be exploring is predicting the average rating of a given recipe. Since this is a **regression** problem, we will be building a linear regression model using `sklearn` in order to solve it.

The response variable in our problem is the `avg_rating` column of the `recipes` DataFrame which describes the aggregate average rating of each recipe (merged from the `interactions` DataFrame). The reason we chose `avg_rating` is because we believe it is valuable to have a general idea of how well a certain recipe is going to perform based on a few features. Additionally, websites such as [food.com](https://www.food.com) and other organizations could utilize such a model in order to curate certain types of recipes which might perform better or simply to fill in missing values of recipes that haven't received any ratings yet.

In order to evaluate our regression model, we will be measuring the **coefficient of determination**, or $R^2$, because it is simple to interpret . While metrics such as Root Mean Squared Error (RMSE) are much more detailed since they tell us exactly how much our model's predictions deviate from the actual values on average, it varies by the scale and type of data. For example, the RMSE of a model predicting the revenue of a company might be in the billions, while the RMSE of a model predicting the price of a banana can vary by a few cents. In other words, it is difficult to gain a deep understanding of your model's performance without important context. On the other hand, $R^2$ is limited to a range of 0 to 1, which is convenient since we can easily understand how well our model fits with unseen data. 

Additionally, it is important to note what type of information will likely be provided at the time of prediction. In our case, since we are predicting the `avg_rating` of a recipe, we will probably have access to most of the features in the `recipes` DataFrame. A few features we plan on including as part of the model are: `complexity`, `n_steps` (number of steps), `calories`, and `saturated_fat (PDV)`. These features will be available to us at the time of prediction because we would have access to the recipe when predicting its rating.


| name                                 |     id |   minutes |   contributor_id | submitted   |   n_steps |   n_ingredients |   avg_rating |   calories |   total_fat (PDV) |   sugar (PDV) |   sodium (PDV) |   protein (PDV) |   saturated_fat (PDV) |   carbohydrates (PDV) | complexity   |
|:-------------------------------------|-------:|----------:|-----------------:|:------------|----------:|----------------:|-------------:|-----------:|------------------:|--------------:|---------------:|----------------:|----------------------:|----------------------:|:-------------|
| 1 brownies in the world    best ever | 333281 |        40 |           985201 | 2008-10-27  |        10 |               9 |            4 |      138.4 |                10 |            50 |              3 |               3 |                    19 |                     6 | simple       |
| 1 in canada chocolate chip cookies   | 453467 |        45 |          1848091 | 2011-04-11  |        12 |              11 |            5 |      595.1 |                46 |           211 |             22 |              13 |                    51 |                    26 | complex      |
| 412 broccoli casserole               | 306168 |        40 |            50969 | 2008-05-30  |         6 |               9 |            5 |      194.8 |                20 |             6 |             32 |              22 |                    36 |                     3 | simple       |
| millionaire pound cake               | 286009 |       120 |           461724 | 2008-02-12  |         7 |               7 |            5 |      878.3 |                63 |           326 |             13 |              20 |                   123 |                    39 | simple       |
| 2000 meatloaf                        | 475785 |        90 |          2202916 | 2012-03-06  |        17 |              13 |            5 |      267   |                30 |            12 |             12 |              29 |                    48 |                     2 | complex      |

---

## Baseline Model

### Create and Split the Data
The first step in building the baseline model is to create our test and training sets. In order to do this, we will utilize `sklearn`'s `train_test_split` function. Additionally, we will use the default split proportion of 0.25.

The first couple of features we will build our baseline model on are: `complexity` (a categorical column) and `n_steps` (a quantitative column).

Our baseline model is going to be built on a Pipeline which utilizes a ColumnTransformer() for preprocessing and transforming the data as well as a Linear Regression model/estimator. Let's take a deeper look at the type of transformers we are going to be applying to our features.

Since complexity is a categorical column, we will have to transform it. Here, we will use the OneHotEncoder() because there are only two types of complexities: simple or complex. Moreover, we decided to drop one of the columns in order to prevent multicollinearity. This means that the OneHotEncoder() has only one category called x0_simple, indicating that a value of 1 means a recipe is simple (less than 9 ingredients), while a value of 0 means a recipe is complex (greater than 9 ingredients). Furthermore, this makes sense intuitively since we would expect simpler recipes which cater to the general public to have higher ratings compared to recipes that are complex and difficult to make.

Also, we will leave `n_steps` as it is because it is a quantitative column.

### Model Performance
After fitting our pipeline creating a prediction based on the input testing set, we end up with an $R^2$ of our model to be `-0.00017975548942428254`. This means that our model barely, if not, didn't fit with the testing data at all. This current baseline model is **not good** based off this performance. In other words, there was a very poor linear fit. Hopefully by adding more features, we can improve the performance in the Final Model.

<center><img src="ml.jpg" alt="Picture of a robot thinking" height="300" width="450"></center>

---

## Final Model

The next couple of features that we will be adding on top of our baseline model are: `calories` and `protein (PDV)`. These features could improve the performace of our model because they are important aspects people consider when rating recipes. For example, someone on a diet would be looking for recipes with a low calorie count and high protein content. Therefore, recipes with high calories and less protein might be rated lower when these additional features are considered. On the other hand, some delectable, but unhealthy, recipes such as desserts (which have lots of calories but less protein) might be rated higher simply because they taste incredible. Let's begin exploring these relationships with our model.

Since the `calories` data has such a wide range and tends to be in the hundreds if not thousands, we will apply `StandardScaler()` in order to standardize the data. Moreover, we will apply `Binarizer()` to the `protein (PDV)` column because it will be helpful to determine how much protein can be considered a significant amount. For the threshold of our `Binarizer()`, we will determine the optimal value using `GridSearchCV`.

### Tuning the Model
In order to optimize our model, we will utilize `GridSearchCV` in order to find the best combinations of hyperparameters. Specifically, we will be searching for the optimal `threshold` parameter value for the `Binarizer()` transformer since we are unsure what a good cutoff is. This is a great way to tune our model and maximize performance in a concise manner.

Based on the `GridSearch`, the optimal threshold for our `Binarizer()` is 30. This means that any values above 30 will be set to 1 and the rest will be set to 0. Now that we have the optimal threshold for our `Binarizer()`, we can create our Pipeline once again and evaluate the performance with $R^2$.

### Model Performance
The model chosen is a regression model, which predicts recipe ratings based on features such as `calories` and `total_fat (PDV)`. The dataset was split into training and testing sets and the data was standardized using `StdScaler` for the calories column. `Binarizer` was applied to the saturated fat column with the threshold value determined by `GridSearchCV`.

The hyperparameter tuned was the threshold value for the Binarizer() transformer. The method used to select hyperparameters was `GridSearchCV`. This method exhaustively searches through a specified parameter grid, fitting the estimator for each combination of parameters and returns the best combination. The performance of the Final Model was evaluated using RMSE, and the best performing hyperparameters were selected based on the lowest RMSE value.

The $R^2$ of our final model is `0.0003677010479545828`. This is a clear improvement over our baseline model which had a negative $R^2$. Undoubtedly, the addition of the new features `calories` and `protein` were critical to the improvement in performance of our Linear Regression model. We believe this is because these are important factors people consider when rating recipes. They are important indicators of the nutritional value and overall healthiness of a recipe. For instance, a recipe with a low calorie count and high protein content is considered a healthy meal. These are aspects everyone takes into consideration when rating a recipe.

The final model's performance was an improvement over the Baseline Model's performance as it included additional features such as calories and total fat. The addition of these features helped the model better predict recipe ratings, leading to a more accurate and comprehensive model.

---

## Fairness Analysis

For our Fairness Analysis, we will be choosing groups based on how long a recipe takes to prepare, in minutes. We will split our data based on the `minutes` column with a threshold of 40 minutes. That is, a `long` recipe is anything greater than 40 minutes while a `short` recipe is anything below.

Group X: `short` recipes that take 40 minutes or less. \
Group Y: `long` recipes that take over 40 minutes.

- **Null Hypothesis:** Our model is fair. Its RMSE for short recipes and long recipes are roughly the same, and any differences are due to random chance.
- **Alternative Hypothesis:** Our model is unfair. Its RMSE for short recipes is lower than its RMSE for long recipes.

In order to evaluate our model, we will be using the **Root Mean Squared Error (RMSE)** as the metric. Specifically, we will be calculating the difference in RMSE between the two Groups X and Y. If the difference is negative, it means that the RMSE for short recipes is lower than the RMSE for long recipes. Additionally, we will be using a significance level of 𝛼 = 5%.

### Conclusion
After running the test for a 1000 repitions, the resulting p-value from our permutation test is 0.0, which is **less than** our significance level of 5% or 0.05. This means that the test was statistically significant, so we **reject the null**. In other words, our model *likely* performs worse for recipes in Group X compared to recipes in Group Y because the RMSE of the short recipe predictions were lower than the RMSE of the long recipe predictions.
