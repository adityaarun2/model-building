# Model Building

Welcome to our model building project, a project focused on predicting ratings of recipes. The goal of this project is to develop a machine learning model that can accurately predict the ratings of recipes based on various features such as ingredients, cooking time, and difficulty level. This is a regression problem, as the predicted variable (ratings) is a continuous numerical variable.

Our response variable is the recipe rating, which ranges from 1 to 5. We chose this variable because it is a crucial factor for determining the success and popularity of a recipe. Our model will assist users in finding recipes that are likely to be highly rated, leading to a more satisfying cooking experience.

The dataset used for this analysis is collected from <a href="food.com">food.com</a>. and contains information on a variety of recipes, including their ingredients, cooking times, and user ratings. By exploring this data, we hope to gain insights into the factors that contribute to a recipe’s success and popularity.

To evaluate our model's performance, we will use the root mean squared error (RMSE) metric. This metric measures the average of the squared differences between the predicted and actual ratings. We chose RMSE because it provides a more comprehensive measure of our model's accuracy compared to other metrics such as mean absolute error.

We hope that this project will help food enthusiasts and cooking enthusiasts discover new and highly-rated recipes with ease.

---

## Problem Identification

The prediction problem we are going to be exploring is predicting the average rating of a given recipe. Since this is a **regression** problem, we will be building a linear regression model using `sklearn` in order to solve it.

The response variable in our problem is the `avg_rating` column of the `recipes` DataFrame which describes the aggregate average rating of each recipe (merged from the `interactions` DataFrame). The reason we chose `avg_rating` is because we believe it is valuable to have a general idea of how well a certain recipe is going to perform based on a few features. Additionally, websites such as [food.com](https://www.food.com) and other organizations could utilize such a model in order to curate certain types of recipes which might perform better or simply to fill in missing values of recipes that haven't received any ratings yet.

In order to evaluate our regression model, we will be measuring the **Root Mean Squared Error (RMSE)** because it is simple to understand and explicitly tells us how much the predictions deviate from the actual values, on average. While metrics such as $R^2$ are easy to interpret since it is limited to a certain range (0 to 1), it is often abstract since it has no units and we can't calculate the error of our model. Specifically, the RMSE can tell be very helpful when interpreted in the context of our data. For example, if the RMSE is 1, we know that, on average, our model differs by an `avg_rating` of 1, which is not great. However, if it is 0.1, our model is actually pretty close to predicting the actual ratings of given recipes.  

Additionally, it is important to note what type of information will likely be provided at the time of prediction. In our case, since we are predicting the `avg_rating` of a recipe, we will probably have access to most of the features in the `recipes` DataFrame. A few features we plan on including as part of the model are: `complexity`, `n_steps` (number of steps), `calories`, and `carbohydrates (PDV)`. These features will be available to us at the time of prediction because we would have access to the recipe which comes with all of the information listed above.


| name                                 |     id |   minutes |   contributor_id | submitted   | nutrition                                                         |   n_steps |   n_ingredients |   avg_rating |   calories |   total_fat (PDV) |   sugar (PDV) |   sodium (PDV) |   protein (PDV) |   saturated_fat (PDV) |   carbohydrates (PDV) | complexity   |
|:-------------------------------------|-------:|----------:|-----------------:|:------------|:------------------------------------------------------------------|----------:|----------------:|-------------:|-----------:|------------------:|--------------:|---------------:|----------------:|----------------------:|----------------------:|:-------------|
| 1 brownies in the world    best ever | 333281 |        40 |           985201 | 2008-10-27  | ['138.4', ' 10.0', ' 50.0', ' 3.0', ' 3.0', ' 19.0', ' 6.0']      |        10 |               9 |            4 |      138.4 |                10 |            50 |              3 |               3 |                    19 |                     6 | simple       |
| 1 in canada chocolate chip cookies   | 453467 |        45 |          1848091 | 2011-04-11  | ['595.1', ' 46.0', ' 211.0', ' 22.0', ' 13.0', ' 51.0', ' 26.0']  |        12 |              11 |            5 |      595.1 |                46 |           211 |             22 |              13 |                    51 |                    26 | complex      |
| 412 broccoli casserole               | 306168 |        40 |            50969 | 2008-05-30  | ['194.8', ' 20.0', ' 6.0', ' 32.0', ' 22.0', ' 36.0', ' 3.0']     |         6 |               9 |            5 |      194.8 |                20 |             6 |             32 |              22 |                    36 |                     3 | simple       |
| millionaire pound cake               | 286009 |       120 |           461724 | 2008-02-12  | ['878.3', ' 63.0', ' 326.0', ' 13.0', ' 20.0', ' 123.0', ' 39.0'] |         7 |               7 |            5 |      878.3 |                63 |           326 |             13 |              20 |                   123 |                    39 | simple       |
| 2000 meatloaf                        | 475785 |        90 |          2202916 | 2012-03-06  | ['267.0', ' 30.0', ' 12.0', ' 12.0', ' 29.0', ' 48.0', ' 2.0']    |        17 |              13 |            5 |      267   |                30 |            12 |             12 |              29 |                    48 |                     2 | complex      |
