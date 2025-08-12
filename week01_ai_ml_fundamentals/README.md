# Week01 AI & ML Fundamentals

## Notes
### Types of ML Systems

- **Supervised learning**: models can make predictions after seeing lots of data with the correct answers and then discovering the connections between the elements in the data that produce the correct answers
  - **Regression model**: predicts a numeric value
  - **Classification models**: predict the likelihood that something belongs to a category. Classification models are divided into two groups: binary classification and multiclass classification. Binary classification models output a value from a class that contains only two values, for example, a model that outputs either `rain` or `no rain`. Multiclass classification models output a value from a class that contains more than two values, for example, a model that can output either `rain`, `hail`, `snow`, or `sleet`.
- **Unsupervised learning**: models make predictions by being given data that does not contain any correct answers. An unsupervised learning model's goal is to identify meaningful patterns among the data. In other words, the model has no hints on how to categorize each piece of data, but instead it must infer its own rules.
  - **Clustering model**: finds data points that demarcate natural groupings
- **Reinforcement learning**: models make predictions by getting *rewards* or penalties based on actions performed within an environment. A reinforcement learning system generates a *policy* that defines the best strategy for getting the most rewards.
- **Generative AI**: is a class of models that creates content from user input

### Supervised Learning

Supervised machine learning is based on the following core concepts:

- **Data**: Datasets are made up of individual *examples* that contain *features* and a *label*. You could think of an example as analogous to a single row in a spreadsheet. Features are the values that a supervised model uses to predict the label. The label is the "answer," or the value we want the model to predict. 
  A dataset is characterized by its *size* and *diversity*. Size indicates the number of examples. Diversity indicates the range those examples cover. Good datasets are both large and highly diverse.
- **Model**: In supervised learning, a model is the complex collection of numbers that define the mathematical relationship from specific input feature patterns to specific output label values. The model discovers these patterns through training.
- **Training**: Before a supervised model can make predictions, it must be trained. To train a model, we give the model a dataset with labeled examples. The model's goal is to work out the best solution for predicting the labels from the features. The model finds the best solution by comparing its predicted value to the label's actual value. Based on the difference between the predicted and actual values—defined as the *loss*—the model gradually updates its solution. In other words, the model learns the mathematical relationship between the features and the label so that it can make the best predictions on unseen data.
- **Evaluating**: We evaluate a trained model to determine how well it learned. When we evaluate a model, we use a labeled dataset, but we only give the model the dataset's features. We then compare the model's predictions to the label's true values.
- **Inference**: Once we're satisfied with the results from evaluating the model, we can use the model to make predictions, called *inferences*, on unlabeled examples. 

#### Linear Regression

Link: [source](https://developers.google.com/machine-learning/crash-course/linear-regression)

*Linear regression* is a statistical technique used to find the relationship between variables. In an ML context, linear regression finds the relationship between *features* and a *label*.

If we have single feature then it is as simple as plotting points `[f, l]` (where `f` is feature on x-axis and `l` is label value on y-axis) and drawing a line that approximates the points. In general in math the line is defined as: 
$$
y = mx + b
$$
where

* *y* is the label (output) value
* *m* is slope of the line
* *x* is feature (input) value
* *b* is the y-intercept

In ML, we write the equation for a linear regression model as follows:
$$
y' = b + w_1x_1
$$
where

* *y'* is the predicted label - the output
* *b* is *bias* of the model; it is sometimes also referred to as *w_0*
* *w_1* is the *weight* of the feature
* *x_1* is a feature - the input

##### Model with multiple features

a more sophisticated model might rely on multiple features, each having a separate weight (, , etc.). For example, a model that relies on five features would be written as follows:
$$
y' = b + w_1x_1 + w_2x_2 + w_3x_3 + w_4x_4 + w_5x_5
$$


## Resources

- 

## Deliverables
- 
