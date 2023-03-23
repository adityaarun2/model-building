# Model Building

Welcome to the exploratory data analysis project on the relationship between cooking ingredients and average rating of recipes! In this project, we will aim to investigate whether there is a correlation between the complexity of a recipe and how highly it is rated by users.

The dataset used for this analysis is collected from food.com. and contains information on a variety of recipes, including their ingredients, cooking times, and user ratings. By exploring this data, we hope to gain insights into the factors that contribute to a recipe’s success and popularity.

We initially started with two DataFrames: raw_recipes, which contains information about recipes and their details, and interactions, which contains information about user reviews and ratings for each recipe. We merged the two datasets on the recipe id and calculated the average rating for each recipe using aggregate statistics. Finally, we merged the Series containing the average rating for each recipe with the original raw_recipes DataFrame in the avg_rating column. This resulted in the recipes DataFrame which we will use for the rest of the project.
