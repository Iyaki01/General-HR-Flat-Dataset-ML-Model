
# <a name="_toc127032419"></a>**Introduction**
Machine learning is the use of algorithms on data to train a model and make predictions based on the training. In this report, we will be doing supervised learning. There are two types of problems that supervised learning models are used to solve, classification and regression.

Classification problems involve predicting a categorical label or class for a given input data. For example, predicting whether an employee will be promoted or not, based on their job performance, years of experience, education, and other factors. In this case, the target variable is categorical value, a binary, and can only be one of two values "promoted" or "not promoted.".

Regression problems, on the other hand, involve predicting a continuous value for a given input data. For example, predicting what a house listing will be priced at, based on the location, size, amenities, reviews, and other features. In this case, the target variable is continuous, which can be a decimal range, between tens of dollars to thousands of dollars.

`	`To predict a value most accurately, the appropriate models should be used to match the data and scenario. Some examples of models are, Logistic Regression, Decision Tree, SVM, and many more. Moreover, ensemble models, which is a type of model that is comprised of a multitude of models, such as Random Forest and Gradient Boosting can be used. 

`	`To create a model, the same general steps must be taken. In assignment 1, the cleaning and transformation of data have already been done. In this assignment, the following steps will be taken to complete the machine learning process.

- Sample the data, split into training, and test sets.
- Choose and build a suitable machine learning model.
- Train the model on the training data.
- Evaluate the model’s performance based on the testing data.
- Improve the model to obtain better results based on initial performance.
- Select a suitable model with sufficient performance for the problem statement.


# <a name="_toc127032420"></a>**HR Analytics**
# <a name="_toc127032421"></a>**Problem Understanding**
The dataset ‘hr\_data,csv’ contains 14 variables, with is\_promoted as its target. The data is assumed all be from a singular company. The data has been collected across many regions, each with a multitude of departments. This means that the promotion criteria, or whether a person is promoted may be highly dependent on these key categorizations. 

|Column|Dtype|
| :- | :- |
|department|Category|
|region|Category|
|education|Category|
|gender|Category|
|recruitment\_channel|Category|
|no\_of\_trainings|Category|
|age|int64|
|previous\_year\_rating|float64|
|length\_of\_service|int64|
|KPIs\_met >80%|Bool|
|awards\_won?|Bool|
|avg\_training\_score|Bool|
|is\_promoted|Bool|

With this knowledge and given that the prediction target is ‘is\_promoted’, with a binary value of 0 and 1, with 1 representing an employee that has been promoted. The following problem statement can be addressed: **“Given these factors, should I plan around this employee being promoted in the future?”** This allows a department like HR to make informed decisions about the career progression of employees and plan accordingly. By using a machine learning model to predict employee promotions based on relevant factors, HR can get a better understanding of which employees are likely to be promoted in the future and make informed decisions regarding resource allocation. This can help the organization to retain its top talent and ensure that it has a strong pipeline of employees for the future. 


# <a name="_toc127032422"></a>**Sample the Data**
## <a name="_toc127032423"></a>Stratified Sampling
1\.1.4 

`	`Sampling is done in classification problems to balance the dataset. This ensure that each class in the target variable is represented equally. This is because the dataset has a highly imbalanced target variable, with the target variable, “is\_promoted’ containing a ratio of 50 000:5 000 of 0:1 binary value. This means the model will be biased towards predicting the 0 value, false, where an employee is not promoted, instead of the 1, true value, where an employee is promoted. The ratio of the target variable in the dataset can be seen below:

`	`To fix this imbalance, stratified sampling will be done. By using stratified sampling, we can ensure that both classes in the target variable are represented equally in the sample. This is done by using the ‘sample’ method. Since we know that the 0 value is overrepresented, and the number of 1 values are smaller in comparison, we will want to sample the 0 value, to make sure that it is balanced with the 1 values.

`	`The above code samples the dataset. A subset of data that contains only 0 values in the target variable is sampled to the size of the number of 1 value. This ensures are balanced sampling of variables within the 0 value, and a balanced ratio of the target variable. The sampled data is then concatenated to the dataframe with 1 value, to create the sampled dataframe.

## <a name="_toc127032424"></a>Train Test Split
1\.2.1 

`	`The next step in the process is to split the sampled data into training and test sets using train/test split. The purpose of the train/test split is to evaluate the performance of the model on unseen data.

The training set is used to train the model, while the test set is used to evaluate the model's performance. A ratio of 70:30 train to test split will be used. Both the train and test split will have the target variable dropped, which can then be used to verify the accuracy of the model. The random sample seed will be fixed to ensure reproducibility but will be tested for edge cases.


# <a name="_toc127032425"></a>**Building the Classification Model**
1\.2

`	`Now that the data has been prepared and split, the next step is to build the classification model to predict employee promotions based on the HR dataframe. Three different models will be used to make predictions, logistic regression, XGBoost Classifier, and Support Vector Classifier (SVC). Logistic regression will be used as the base model, serving as the baseline comparison, while XGBoost Classifier and SVC will be tested, and the best will be chosen to suit the problem statement.

## <a name="_toc127032426"></a>Logistic Regression
1\.2.2

Logistic regression models the relationship between the dependent variable and one or more independent variables by fitting a logistic curve to the data. It uses a logistic function to model the probability of the binary response as a function of the predictor variables. It is easy to implement, fast, and interpretable, making it a good choice for a base model.


## <a name="_toc127032427"></a>XGBoost
1\.2.3

XGBoostClassifier is a gradient boosting algorithm uses a decision tree-based method, combining the predictions of many weak decision trees. XGBoost is designed to be both fast and accurate, and it is well-suited for large datasets with many features. 

## <a name="_toc127032428"></a>Support Vector Machine
1\.2.4

SVC is a linear classifier that finds the hyperplane that separates the data into two classes. It uses a cost function to determine the best hyperplane that separates the data. SVC is a powerful algorithm that can handle non-linear data, but it is more complex to implement and requires more computational resources than logistic regression or XGBoost.


# <a name="_toc127032429"></a>**Evaluation**
## <a name="_toc127032430"></a>Results

|Model|Train Score|Test Score|Score Difference|
| :- | :- | :- | :- |
|Logistic Regression|0\.728|0\.735|-0.007|
|XGBoost Classifier|0\.93|0\.8|0\.13|
|Support Vector Classifier|0\.72|0\.71|0\.01|

`	`The above table are the accuracies of the models when fitted with the parameters in the pervious section. It shows the Train and Test scores and difference between the scores. 

## <a name="_toc127032431"></a>Logistic Regression
For logistic regression, the train score is 0.728 and the test score is 0.735, and a very small score difference of -0.007. However, as the score difference is negative, meaning the test score is higher than the train score, it suggests underfitting. This is because the model is not learning enough from the train dataset, and thus can achieve higher scores when new data is shown. Moreover, though the score differential is small, the overall score is at ~0.73, which is not very high.
## <a name="_toc127032432"></a>XGBoost
XGBoost Classifier has a train score of 0.93 and a test score of 0.80, with a score difference of 0.13. This suggests that the model is overfitting to the training data as the train score is much higher than the test score, meaning it is not able to generalize well to the test data. The high accuracy of the model on the training data is part of an indication of good model performance, but a high test score combined with a small difference does. This model performs the best, at 0.80 score, but has the highest overfitting problem.
## <a name="_toc127032433"></a>Support Vector Machine
SVC has a train score of 0.72 and a test score of 0.71, with a score difference of 0.01. While the model does not have under or overfitting issues like the previous models, it also has a low overall score. This suggests that SVC may not be a suitable model for this particular scenario or dataset.


# <a name="_toc127032434"></a>**Improvement** 
## <a name="_toc127032435"></a>Removing Input Variables
Removing input variables based on P>|z| is a way to improve the performance of a model by reducing the number of input variables that are used in the model. The P value is used to determine if the relationship between the input variable and the target variable is significant. The |z| is the absolute value of the Z-score, which measures the number of standard deviations away from the mean a data point is. This means that a high P value indicates that the variable is insignificant. 

`	`Above is a summary of the input variables provided using the ‘summary’ method from the logistic regression model. The variable ‘region’ has a high P value of 0.821. This high P value of 0.821 indicates that the relationship between ‘region’ and ‘is\_promoted’ insignificant. Thus it should be removed from the model. 

By doing so, the model will be less prone to the risk of overfitting. It may also improve the model's performance, as it would be using fewer input variables that may not have a significant effect on the response variable. However, it should be noted that removing the variable will reduce a input variable, which may be important to the business.
## <a name="_toc127032436"></a>GridSearch & Cross Validation
`	`GridSearch is a hyperparameter tuning method that is used to find the best hyperparameters for a machine learning model. GridSearch will be used to find the best hyperparameters for the logistic regression, XGBoost, and SVC models. 

GridSearch takes in a set of hyperparameters and uses cross-validation to find the combination of hyperparameters that gives the best performance for the model. This is done by training the model multiple times, each time with a different combination of hyperparameters, and evaluating the performance of the model on a validation set. The best hyperparameters are then selected based on the performance of the model. Moreover, by applying the use of cross-validation, overfitting is reduced at the same time.

However, GridSearch has disadvantages, namely, computation requirements. GridSearch may find a set of hyperparameters that are well suited for the train dataset but may not be the actual best hyperparameters that could be used. To combat this, a larger set of possible parameters can be provided, however this significantly increases computational time, as each new variation must be tested.

Above is an implementation of GridSearch for XGBoost Classifier. Even though the number of hyperparameters is relatively minimal, the block of code takes ~6seconds to run. This emphasizes the need to know an estimate of what the range of possible best parameters will be and run GridSearch on that to save computational time. GridSearch will be applied to all models.

## <a name="_toc127032437"></a>Voting Classifier
1\.3.4

`	`Voting Classifier is an ensemble machine learning algorithm that combines predictions from multiple models. It aggregates the predictions and chooses the best one, to improve the accuracy of the model. The logistic regression, XGBoost, and SVC models will be used. 

`	`The voting classifier will use soft voting, where predictions made by each model is combined into a final prediction. This means that the classifier with the highest predicted probability for a particular class will have more influence in the final prediction than classifiers with lower probabilities for that class. 

Since this model is only a combination of the previous models, it has been only made here since it is best to use the most accurate possible version of the models to aggregate. Below is the implementation of the voting classifier.



## <a name="_toc127032438"></a>Results	
1\.3.5

|Model|Train Score|Test Score|Score Difference|
| :- | :- | :- | :- |
|Logistic Regression|0\.73 +0.01|0\.72 -0.01|0\.01 +0.02|
|XGBoost Classifier|0\.84 -0.09|0\.82 +0.02|0\.02 -0.11|
|Support Vector Classifier|0\.81 +0.09|0\.78 +0.07|0\.03 +0.02|
|Voting Classifier|0\.82 |0\.81|0\.01|

`	`The above table are the accuracies of the models when fitted with the new hyper parameters obtained from performing GridSearch and cross-validation. It shows the Train and Test scores and difference between the scores. The changes in score from the initial fitting is colored.

The logistic regression model can be seen to have a minor increase in train and score difference, with the test score dropping. The results now show that the model is no longer underfitting and is well fitted, with a small score difference of 0.01. However, while the model no longer has fitting issues, it has a low accuracy.

`	`The XGBoost classifier can be seen to have the largest change in score, with the train and test scores change in the opposite direction of each other, contributing to make the score difference between the two 0.02, down from 0.13. These results indicate that the model is no longer overfitting and is able to effectively generalize the data. It also has the highest accuracy, with a test score of 0.82.



`	`The SVC can be seen to have a large increase in both the train and test score. This increases the test score closer to 0.80, but still falls short of it. Moreover, whereas previously it had a very small score difference of 0.01, it now has a larger albeit still minor score difference of 0.03. Much like logistic regression, though the model is does not have fitting issues, it lacks a high accuracy, though not to the same level as logistic regression.

The voting classifier has a train score of 0.82 and a test score of 0.81. The score difference is 0.01, which indicates that the model is not overfitting or underfitting. 

The results suggest that the XGBoost Classifier has gained the most after improvement. The voting classifier is almost as accurate as XGBoost, without as much of a score difference. It has a good balance between the train and test scores, but it would be worth exploring other improvement techniques for the individual models to further improve the performance.



##
##

# <a name="_toc127032439"></a>**Summary**
`	`The problem statement to be addressed was **“Given these factors, should I plan around this employee being promoted in the future?”** This problem requires an accurate model to identify whether an employee would be promoted. To do so, HR data with variables that would affect a change of promotion was used to train and test the models.

`	`First, the data was cleaned, transformed, and scaled. Afterwards, stratified sampling was done to reduce the imbalanced nature of the data, to ensure that minority groups of data were represented appropriately. The sampled data was then split into train and test subsets.



`	`Three models were built, logistic regression, as the base model, XGBoost Classifier, and SVC. The logistic regression model, the base model, served as a baseline of performance to compare the other models. The initial results obtained were as unsatisfactory, with all models hovering around 0.70-0.80 with overfitting and underfitting issues.

`	`To improve the models, input variables with a high P value were removed. This is because variables with a high P value are an indication of insignificance to the target variable. It also assists in reducing overfitting. Finally, GridSearch and cross-validation was done to get the best hyper parameters for each model as well as reduce overfitting.

`	`Finally, a voting classifier was built using the models built, aggregating the results, and getting the best out of the models built previously. This is done on the improved versions of the models to attain the best possible results. The results of all the improved models and voting classifier were then discussed.

`	`Given the results of the models, only XGBoost classifier and voting classifier can answer the problem statement. Of the two, XGBoost classifier has a slightly higher accuracy, but also has a higher score difference between its train and test compared to voting classifier. A higher accuracy model would be required to effectively answer the problem statement, but out of the 2 models built, voting classifier is the better option.

`	`This is because it is vital that the model has a consistent performance and can generalize well to all data. The model can be relied upon to be correct 4 out of 5 times reliably. Compared to XGBoost classifier, which has a higher variability in its result and ability to train on a dataset, voting classifier is the better option.


# <a name="_toc127032440"></a>**Airbnb Listings**
# <a name="_toc127032441"></a>**Problem Understanding**
The dataset ‘listing,csv’ contains 16 variables, with price as its target variable. The data is from listings only in Singapore from May 2015 to Aug 2019, containing a total of just under 8 thousand rows. The data includes listings from all around Singapore, categorized under neighbourhood and broadly as neighbourhood\_group. This will result in some areas having a much higher influence in the final model as the data collected is not equally distributed across all regions.

|Column|Dtype|
| :- | :- |
|id|Category|
|name|Category|
|host\_id|Category|
|host\_name|Category|
|neighbourhood\_group|Category|
|neighbourhood|Category|
|latitude|float64|
|longitude|float64|
|room\_type|Category|
|price|int64|
|minimum\_nights|int64|
|number\_of\_reviews|int64|
|last\_review|Date|
|reviews\_per\_month|float64|
|calculated\_host\_listings\_count|int64|
|availability\_365|int64|
With this knowledge and given that the prediction target is ‘price’, is a continuous value representing the price of a listing. The following problem statement can be addressed: **“Given these factors, what should I price my listing at?”** This allows property owners to get an estimate of the valuation of their listing. This helps them attract potential customers as the price will be what the customer wants, allowing to maximize profits for the target demographic. Moreover, it allows owners to help customers make informed decisions about their choice of stay, allowing them to negotiate better prices.


# <a name="_toc127032442"></a>**Sample the Data**
## <a name="_toc127032443"></a>Outlier Handling
2\.1.3 

`	`Outlier handling will be done to remove data that is not relevant to the problem statement. This ensures that the data used to train and test the model will only contain data that the problem statement seeks to address. Below are visualizations of the target variable’s distribution.

`	`It can be seen from the histogram and boxplot that most data points fall below 2000, yet outliers exist beyond this point which skew the mean and median. If these points are not dealt with, it will heavily impact the results of the models, leading to overfitting and poor predictions. Moreover, realistically, the average customer will not be looking for a listing that costs thousands of dollars. Accounting for these values will not be helpful for addressing the problem statement. 

`	`To fix this sub setting will be done. Sub setting is simply removing values that are outside of the set boundaries. This will allow for a controlled range of the target variable, ensure that only the relevant data that we want is kept. Moreover, percentiles can be used as well, to capture as much of the data as possible, whilst avoiding outliers and adhering to the goal of addressing the target variable.

`	`The above code subsets the dataset. The data that is past the upper boundary is removed and the data below the lower boundary is removed, leaving only the data that is relevant. It should be noted that a significant number of rows have been lost, which may have to be taken into consideration in the future if additional requirements arise. Below is the distribution after sub setting. 

## <a name="_toc127032444"></a>Train Test Split
2\.2.1 

`	`The next step in the process is to split the sampled data into training and test sets using train/test split. The purpose of the train/test split is to evaluate the performance of the model on unseen data.

The training set is used to train the model, while the test set is used to evaluate the model's performance. A ratio of 70:30 train to test split will be used. Both the train and test split will have the target variable dropped, which can then be used to verify the accuracy of the model. The random sample seed will be fixed to ensure reproducibility but will be tested for edge cases.
# <a name="_toc127032445"></a>**Building the Regression Model**
2\.2

`	`Now that the data has been prepared and split, the next step is to build the regression model to predict price of a listing based on the listings dataframe. Three different models will be used to make predictions, linear regression, Random Forest Regressor (RFR), and XGBoost regressor. Linear regression will be used as the base model, serving as the baseline comparison, while XGBoost regressor and RFR will be tested, and the best will be chosen to suit the problem statement.

## <a name="_toc127032446"></a>Linear Regression
2\.2.2

Linear regression models the relationship between the dependent variable and one or more independent variables by fitting a straight line to the data. It uses a linear equation to model the relationship between the predictors and the response variable. It is simple to implement, fast, and interpretable, making it a good choice for a base model.

## <a name="_toc127032447"></a>Random Forest Regressor
2\.2.3

Random Forest Regressor is an ensemble machine learning algorithm that builds multiple decision trees and combines the results of these trees to improve the accuracy of the predictions. It uses a decision tree-based method and generates many trees, with each tree contributing to the overall prediction. RFR is designed to be both fast and accurate, and it is well-suited for large datasets with many features.

##
##
##
##
##
##
## <a name="_toc127032448"></a>XGBoost
2\.2.4

XGBoost Regressor is a gradient boosting algorithm that uses a decision tree-based method, combining the predictions of many weak decision trees. XGBoost is designed to be both fast and accurate, and it is well-suited for large datasets with many features. It is a powerful algorithm that can handle non-linear data and is commonly used in machine learning competition and real-world applications.


# <a name="_toc127032449"></a>**Evaluation**
## <a name="_toc127032450"></a>Results
### <a name="_toc127032451"></a>Without Scaling

|Model|Train Score|Test Score|Score Difference|Test RMSE|Test MAE|
| :- | :- | :- | :- | :- | :- |
|Linear Regression|0\.50|0\.54|-0.04|25\.6754|20\.5471|
|Random Forest Regressor|0\.52|0\.58|-0.06|24\.5843|19\.6038|
|XGBoost Regressor|0\.90|0\.64|0\.26|22\.7788|16\.8977|

`	`The above table are the accuracies of the models when fitted with the parameters in the previous section. It shows the Train and Test scores and difference between the scores. 

### <a name="_toc127032452"></a>With Scaling
For reference, these are the metrics when the target variable is scaled. Note lack of change outside of RMSE and MAE.

|Model|Train Score|Test Score|Score Difference|Test RMSE|Test MAE|
| :- | :- | :- | :- | :- | :- |
|Linear Regression|0\.50|0\.54|-0.04|0\.1990|0\.1592|
|Random Forest Regressor|0\.52|0\.58|-0.06|0\.1905|0\.1518|
|XGBoost Regressor|0\.90|0\.64|0\.26|0\.1773|0\.1311|
##
## <a name="_toc127032453"></a>Linear Regression
`	`The linear regression model has a train score of 0.50 and test score of 0.54. There is a score difference of -0.04, where the test does better than the train. This means the model is not learning enough from the training dataset and is underfitting. Moreover, the overall score is low, at around ~0.50, which means it not very accurate. The test RMSE and MAE of 25.6754 and 20.5471 are relatively high, which means the model is not making very accurate predictions.
## <a name="_toc127032454"></a>Random Forest Regressor
The RFR model has a train score of 0.52 and test score of 0.58. The difference between the scores is -0.06, which much like linear regression, means the test does better than the train. This indicates the model is not learning enough from the dataset and is underfitting even more than linear regression. However, the scores on average generally higher, meaning the model has a potential to be more drastically improved. The test RMSE and MAE of 24.5843 and 19.6038 are still relatively high albeit slightly better than linear regression, which means that the model is not making very accurate predictions.

<a name="_toc127032455"></a>XGBoost

The XGBoost regressor model has a train score of 0.90 and a test score of 0.64. There is a large difference between the train and test scores at 0.26, which means that the model is severely overfitting the data and is unable to generalize well. However, since the test score is so high, it has the largest potential to improve. The test RMSE and MAE of 22.7788 and 16.8977 are the lowest of all the models, but is still rather high, meaning the model is not making to most accurate of predictions. Because of its high train score and low RMSE and MAE, XGBoost has the highest chance of becoming the most suited model.


# <a name="_toc127032456"></a>**Improvement** 
## <a name="_toc127032457"></a>Removing Input Variables
Removing input variables based on P>|z| is a way to improve the performance of a model by reducing the number of input variables that are used in the model. The P value is used to determine if the relationship between the input variable and the target variable is significant. The |z| is the absolute value of the Z-score, which measures the number of standard deviations away from the mean a data point is. This means that a high P value indicates that the variable is insignificant. 

`	`Above is a summary of the input variables provided using the ‘summary’ method from the linear regression model. The variables ‘neighbourhood\_group’, ’number\_of\_reviews’, and ‘calculated\_host\_listings\_count’ have a P value of around 0.200-0.360. These P values are high, especially compared to other variables with 0 P value. However, since the P value does not cross the 0.400 threshold, no variables will be dropped. This allows for all the data to be kept, which may be important to the business. 
##
## <a name="_toc127032458"></a>GridSearch & Cross Validation
`	`GridSearch is a hyperparameter tuning method that is used to find the best hyperparameters for a machine learning model. GridSearch will be used to find the best hyperparameters for the linear regression, RFR, and XGBoost models. 

GridSearch takes in a set of hyperparameters and uses cross-validation to find the combination of hyperparameters that gives the best performance for the model. This is done by training the model multiple times, each time with a different combination of hyperparameters, and evaluating the performance of the model on a validation set. The best hyperparameters are then selected based on the performance of the model. Moreover, by applying the use of cross-validation, overfitting is reduced at the same time.

However, GridSearch has disadvantages, namely, computation requirements. GridSearch may find a set of hyperparameters that are well suited for the train dataset but may not be the actual best hyperparameters that could be used. To combat this, a larger set of possible parameters can be provided, however this significantly increases computational time, as each new variation must be tested.

Above is an implementation of GridSearch for FRF. Even though the number of hyperparameters is relatively minimal, the block of code takes ~8seconds to run. This emphasizes the need to know an estimate of what the range of possible best parameters will be and run GridSearch on that to save computational time. GridSearch will be applied to all models.


`	`Additionally, by plotting the MSE by max depth and showing the cross-validation scores and overall scores, a optimal max depth can be chosen. The closer a point is to the X-axis, the lower the MSE, which means the model is more accurate. However, to achieve this, a higher max depth is required, which will increase overfitting. This can be seen how the blue line, the train, MSE decreases linearly as max depth decreases, meaning it is learning the dataset. However, at a certain point, the line in red, the test, does not decrease at a similar rate, meaning that the model is overfitting. By choosing a suitable max depth, it is possible reduce the MSE as far as possible whilst maintaining an acceptable gap between the training and test, to reduce overfitting.
##
## <a name="_toc127032459"></a>Voting Classifier
1\.3.4

`	`Voting Classifier is an ensemble machine learning algorithm that combines predictions from multiple models. It aggregates the predictions and chooses the best one, to improve the accuracy of the model. The linear regression, RFR, and XGBoost models will be used. 

Since this model is only a combination of the previous models, it has been only made here since it is best to use the most accurate possible version of the models to aggregate. Below is the implementation of the voting classifier.


## <a name="_toc127032460"></a>Results
### <a name="_toc127032461"></a>Without Scaling

|Model|Train Score|Test Score|Score Difference|Test RMSE|Test MAE|
| :- | :- | :- | :- | :- | :- |
|Linear Regression|<p>0\.50</p><p>+0.00</p>|0\.54<br>+0.00|<p>-0.04</p><p>+0.00</p>|<p>25\.6754</p><p>+0.00</p>|<p>20\.5471</p><p>+0.00</p>|
|Random Forest Regressor|<p>0\.68</p><p>+0.16</p>|<p>0\.66</p><p>+0.06</p>|<p>0\.02</p><p>+0.08</p>|<p>21\.9711</p><p>-3.7043</p>|<p>16\.8680</p><p>-2.7358</p>|
|XGBoost Regressor|<p>0\.72</p><p>-0.18</p>|<p>0\.64</p><p>+0.00</p>|<p>0\.08</p><p>-0.18</p>|<p>22\.7789</p><p>-0. 0001</p>|<p>17\.3293</p><p>+0.3780</p>|
|Voting Regressor|0\.69|0\.65|0\.04|22\.2235|17\.2757|

`	`The above table are the accuracies of the models when fitted with the improvements performed in the previous section. It shows the Train and Test scores, the difference between the scores as well as the RMSE and MAE changes. The changes in score from the initial fitting is colored.

### <a name="_toc127032462"></a>With Scaling
For reference, these are the metrics when the target variable is scaled. Note lack of change outside of RMSE and MAE.

|Model|Train Score|Test Score|Score Difference|Test RMSE|Test MAE|
| :- | :- | :- | :- | :- | :- |
|Linear Regression|<p>0\.50</p><p>+0.00</p>|0\.54<br>+0.00|<p>-0.04</p><p>+0.00</p>|<p>0\.1990</p><p>+0.00</p>|<p>0\.1592</p><p>+0.00</p>|
|Random Forest Regressor|<p>0\.68</p><p>+0.16</p>|<p>0\.66</p><p>+0.06</p>|<p>0\.02</p><p>+0.08</p>|<p>0\.1699</p><p>-0.0291</p>|<p>0\.1304</p><p>-0.0214</p>|
|XGBoost Regressor|<p>0\.72</p><p>-0.18</p>|<p>0\.64</p><p>+0.00</p>|<p>0\.08</p><p>-0.18</p>|<p>0\.1761</p><p>-0.0012</p>|<p>0\.1335</p><p>+0.0024</p>|
|Voting Regressor|0\.69|0\.65|0\.04|0\.1718|0\.1220|
### <a name="_toc127032463"></a>Model Results
The linear regression train score and test score are low, at ~0.50 with a difference of 0.04. Since the train is lower than the test, the model is underfitting. Moreover, the overall accuracy is low and RMSE and MAE scores for the test data are still relatively high, indicating that the predictions made by the linear regression model are not accurate.

The RFR train score is higher than the test score, which suggests that this model is overfitting the training data. However, the difference between the two scores is not very large, indicating that the overfitting is not very severe. The RMSE and MAE scores for the test data are lower than those of the linear regression model, indicating that the random forest regressor is a better fit for this data.

The XGBoost Regressor train score much higher than the test data, at 0.08 difference, which is better than the initial, but still indicates a high amount of overfitting. The RMSE score for the test data have improved, but the MAE has worsened. Due to the high overfit, it is will not generalize well to new data.

For voting regressor, train score and test score are relatively close to each other, with a difference of only 0.04. This suggests that there is no significant overfitting or underfitting with this model. The RMSE and MAE scores for the test data are lower than those of the XGBoost regressor, indicating that this model is a better fit for the data.

Overall, the Random Forest Regressor seem to be the best, with the lowest RMSE, MAE and highest test score with minimal overfitting. Although XGBoost had the largest change from the improvement, it did not have a increase in its actual predictive accuracy. 


## <a name="_toc127032464"></a>Scaling the Target Variable
`	`Scaling the target variable, price, leads to lower RMSE, MAE, and MSE. This is because these values are directly related to the original variable’s unit. Thus, when scaling to 0 – 1, these metrics are lowered accordingly. However, there will be no actual improvement to the model results. 

Moreover, this unless the price predicted from the model is rescaled, it is difficult to interpret which may lead to confusion and incorrect assumptions. 

`	`Since scaling the target variable leads to no real-world benefit, and causes difficulty in interpretation, it will not be done despite the decrease in RMSE and MAE. The effects of scaling can be seen in the results table previously, comparing the scaled against without scaling.


# <a name="_toc127032465"></a>**Summary**
`	`The problem statement to be addressed was **“Given these factors, what should I price my listing at?”** This problem requires a relatively accurate model to categorize what price range a listing should be listed as. To do so, Airbnb listing data with variables that would affect the price range of a listing were train and tested on.

`	`First, the data was cleaned, transformed, and scaled. Afterwards, sub setting was done to remove outliers and get only data that would be useful to most listing hosts. The sampled data was then split into train and test subsets.



`	`Three models were built, linear regression, as the base model, RFR, and XGBoost regressor. The linear regression model, the base model, served as a baseline of performance to compare the other models. The initial results obtained were as unsatisfactory, with all models hovering around 0.50-0.60 with overfitting and underfitting issues.

`	`To improve the models, GridSearch and cross-validation was done to get the best hyper parameters for each model as well as reduce overfitting. Charts were also plotted to manually identify what max depth was best suited for the model.

`	`Finally, a voting classifier was built using the models built, aggregating the results, and getting the best out of the models built previously. This is done on the improved versions of the models to attain the best possible results. The results of all the improved models and voting classifier were then discussed.

`	`Given the results of the models, only RFR and voting regressor can answer the problem statement. Of the two, RFR has a slightly higher accuracy, and has a lower score difference between its train and test compared to voting regressor meaning it will generalize better to new data. Due to these reasons, RFR is the best model to answer this problem statement.

`	`This is because it is vital that the model has a consistent performance and can generalize well to all data. The model can be relied upon to be correct 4 out of 5 times reliably. Compared to XGBoost classifier, which has a higher variability in its result and ability to train on a dataset, voting classifier is the better option.

`	`This is because it is the most accurate of all the models and generalizes the best to new data. Moreover, its RMSE and MAE indicate that the error generated when the model is incorrect is still barely acceptable. This is because a precise price figure is not needed. The problem statement only requires a range of the optimal price for a listing, and should be used to categorize the listing, and the exact price should be down to the host, based on their knowledge of the listing’s amenities, location, usability, and other factors not found in the dataset.
#
#

# <a name="_toc127032466"></a>**Conclusion**
`	`This report has documented the work done to build a classification model on the dataset HR Analytics and a regressor model on the dataset Airbnb Listings. The classification and regressor models aimed to address the following problem statements respectively, **“Given these factors, should I plan around this employee being promoted in the future?”**, and **“Given these factors, what should I price my listing at?”** 



`	`To build models that suited the data and could answer these problems effectively, the following general steps were taken for both models, sampling, model building, evaluation, model improvement, and model selection.



`	`The first step, sampling the data, is the most crucial step. This is reliant on the nature of the problem statement. Two different approaches were taken to complete this step. For HR analytics, the data was imbalanced, which would have led to a highly overfit model. To solve this stratified sampling was done to ensure that both majority and minority classes were balanced, allowing for the model to better generalize for different types of data. For Airbnb listings, the data was of a very large range, which would have led to a very inaccurate model. To solve this, the data was subset to include only the data that was relevant to the problem statement, ensuring that data points which did not affect the model.

`	`Once the data was prepared, 3 models were built for each problem. The initial model results were unsatisfactory, thus had to be improved. Several methods were implemented to improve the models. The methods used were, the removal of input variables that were insignificant to the target variable, utilizing GridSearch and cross-validation to get the best hyper parameters and reduce overfitting, and for the regressor problem, max depth was tuned manually by information gained from visualizations.

`	`Finally, a voting classifier and regressor was created, which is an aggregation of the improved model results, to combine their strengths into one model. 

`	`The model improved model results were then compared against one another, and the model that suited the data the best whilst answering the problem statement was selected.

