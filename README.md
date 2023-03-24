# The Great Recipe Rating Race: Using Machine Learning to Help You Cook Like a Pro!

Welcome to our model building project, a project focused on predicting ratings of recipes. The goal of this project is to develop a machine learning model that can accurately predict the ratings of recipes based on various features such as ingredients, cooking time, and difficulty level. This is a regression problem, as the predicted variable (ratings) is a continuous numerical variable.

Our response variable is the recipe rating, which ranges from 1 to 5. We chose this variable because it is a crucial factor for determining the success and popularity of a recipe. Our model will assist users in finding recipes that are likely to be highly rated, leading to a more satisfying cooking experience.

The dataset used for this analysis is collected from <a href="food.com">food.com</a>. and contains information on a variety of recipes, including their ingredients, cooking times, and user ratings. By exploring this data, we hope to gain insights into the factors that contribute to a recipe’s success and popularity.

To evaluate our model's performance, we will use the root mean squared error (RMSE) metric. This metric measures the average of the squared differences between the predicted and actual ratings. We chose RMSE because it provides a more comprehensive measure of our model's accuracy compared to other metrics such as mean absolute error.

We hope that this project will help food enthusiasts and cooking enthusiasts discover new and highly-rated recipes with ease.

<center><img src="cook.jpeg" alt="Picture of food getting cooked" height="300" width="600"></center>

---

## Problem Identification

The prediction problem we are going to be exploring is predicting the average rating of a given recipe. Since this is a **regression** problem, we will be building a linear regression model using `sklearn` in order to solve it.

The response variable in our problem is the `avg_rating` column of the `recipes` DataFrame which describes the aggregate average rating of each recipe (merged from the `interactions` DataFrame). The reason we chose `avg_rating` is because we believe it is valuable to have a general idea of how well a certain recipe is going to perform based on a few features. Additionally, websites such as [food.com](https://www.food.com) and other organizations could utilize such a model in order to curate certain types of recipes which might perform better or simply to fill in missing values of recipes that haven't received any ratings yet.

In order to evaluate our regression model, we will be measuring the **coefficient of determination**, or $R^2$, because it is simple to interpet . While metrics such as Root Mean Squared Error (RMSE) are much more detailed since they tell us exactly how much our model's predictions deviate from the actual values on average, it varies by the scale and type of data. For example, the RMSE of a model predicting the revenue of a company might be in the billions, while the RMSE of a model predicting the price of a banana can vary by a few cents. In other words, it is difficult to gain a deep understanding of your model's performance without important context. On the other hand, $R^2$ is limited to a range of 0 to 1, which is convenient since we can easily understand how well our model fits with unseen data. 

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

<center><img src="ml.jpg" alt="Picture of a robot thinking" height="300" width="450"></center>

### Create and split the data
The first step in building the baseline model is to create our test and training sets. In order to do this, we will utilize `sklearn`'s `train_test_split` function. Additionally, we will use the default split proportion of 0.25.

The first couple of features we will build our baseline model on are: `complexity` (a categorical column) and `n_steps` (a quantitative column).

Since `complexity` is a categorical column, we will have to transform it. Here, we will use the `OneHotEncoder()` because there are only two types of complexities: `simple` or `complex`. Moreover, we decided to drop one of the columns in order to prevent multicollinearity. As we will see later on, the `OneHotEncoder()` has only one category called `x0_simple` indicating that a value of 1 means a recipe is `simple` (less than 9 ingredients), while a value of `0` means a recipe is `complex` (greater than 9 ingredients).

We will leave `n_steps` as it is because it is a quantitative column.

### Training the Pipeline
Now that we have our transformers and `Pipeline` declared, we are ready to train our model. Below, we fit the Pipeline on the `X_train` and `y_train` data sets that we created earlier.

### Performance
After fitting our pipeline creating a prediction based on the input testing set, we end up with a Root Mean Squared Error of `0.6319575251766202`. This means that, on average, our model's predictions differed from the actual ratings by about 0.63. Considering that the `avg_ratings` are on a scale from 1 to 5, this is not the best performance. For example, if we decide to round the RMSE up, we will be nearly a whole rating (1) off from the actual ratings most of the time. 

---
