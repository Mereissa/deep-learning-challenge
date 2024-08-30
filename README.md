# Alphabet Soup Charity Funding Predictor

This project aims to help the nonprofit foundation Alphabet Soup identify which applicants for funding are most likely to succeed in their ventures. Using a dataset of over 34,000 organizations that have received funding from Alphabet Soup over the years, we have built a binary classifier model that predicts the success of future applicants based on various features.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Technologies Used](#technologies-used)
4. [Preprocessing](#preprocessing)
5. [Model Development](#model-development)
6. [Model Optimization](#model-optimization)
7. [Results](#results)

## Project Overview

Alphabet Soup seeks to maximize the impact of its funding by selecting applicants with the best chance of success. This project uses machine learning and neural networks to develop a binary classification model that can accurately predict whether a given applicant is likely to be successful. The target accuracy for this model is set at over 75%.

## Dataset

The dataset used for this project includes information about more than 34,000 organizations funded by Alphabet Soup. Key features in the dataset include:

- `EIN` and `NAME`: Identification columns
- `APPLICATION_TYPE`: Type of application submitted
- `AFFILIATION`: Affiliated sector of the industry
- `CLASSIFICATION`: Government organization classification
- `USE_CASE`: Purpose for the funding request
- `ORGANIZATION`: Type of organization
- `STATUS`: Active status
- `INCOME_AMT`: Income classification
- `SPECIAL_CONSIDERATIONS`: Whether there are any special considerations
- `ASK_AMT`: Amount of funding requested
- `IS_SUCCESSFUL`: Whether the funding was used effectively (target variable)

## Technologies Used

- Python
- Pandas
- TensorFlow and Keras
- scikit-learn

## Preprocessing

1. **Data Cleaning**: Dropped non-beneficial columns such as `EIN` and `NAME`.
2. **Categorical Encoding**: Categorical variables were converted to numerical using one-hot encoding with `pd.get_dummies()`.
3. **Handling Rare Categories**: Categorical values with low frequencies were grouped into an 'Other' category.
4. **Data Splitting**: The data was split into training and testing sets using `train_test_split`.
5. **Data Scaling**: Features were scaled using `StandardScaler()` to ensure the model is not biased towards certain features due to scale differences.

## Model Development

The neural network was designed using TensorFlow and Keras. The architecture includes:

- An input layer based on the number of features.
- One or more hidden layers with appropriate activation functions.
- An output layer with a sigmoid activation function to perform binary classification.

The model was compiled and trained using binary cross-entropy as the loss function, and accuracy as the metric.

## Model Optimization

To achieve a target accuracy of over 75%, several optimization techniques were applied:

- Adjusting the number of neurons in the hidden layers.
- Adding or removing hidden layers.
- Changing activation functions.
- Experimenting with different numbers of epochs.

## Results

- The initial model achieved an accuracy of around XX%.
- After optimization, the model's accuracy was improved to YY%.
- [Summary of any key findings, if applicable].

