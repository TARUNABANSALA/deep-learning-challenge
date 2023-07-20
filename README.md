# deep-learning-challenge
Background
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, We have used the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special considerations for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively
Instructions
Step 1: Preprocess the Data.
Step 2: Compile, Train, and Evaluate the Model
Step 3: Optimize the Model: Using TensorFlow, optimized the model to achieve a target predictive accuracy higher than 75%.
Step 4: Write a Report on the Neural Network Model


The detailed analysis report on AlphabetSoup is as follows:

Overview of the analysis: Explain the purpose of this analysis.
The goal of this analysis is to create a deep learning binary classifier that predicts the success rate of funding applicants for Alphabet Soup, a nonprofit organization. The dataset provided contains information on over 34,000 organizations, including application details, industry affiliations, government classification, funding use cases, income classification, funding amount requested, and fund utilization.

The analysis involves preprocessing the data by removing unnecessary columns, encoding categorical variables, and splitting the dataset into training and testing sets. A neural network model is then designed, trained, and evaluated for its loss and accuracy. Optimization techniques such as adjusting input data, modifying network architecture, activation functions, and training epochs are applied to improve the model's performance.

The ultimate objective is to achieve a predictive accuracy higher than 75%. Once optimized, the model is saved as an HDF5 file for future use.

Results:
Data Preprocessing
What variable(s) are the target(s) for your model?

The target variable for the model is IS_SUCCESSFUL, displaying whether a charity donation was successful or not.

What variable(s) are the features for your model?

The feature variables for the model are the rest of the columns in the DataFrame, excluding IS_SUCCESSFUL

What variable(s) should be removed from the input data because they are neither targets nor features

I believe that EIN (Employee Identification Number) does not contain relevant information for our predictive model, hence left out this variable from the feature and target selection.

Compiling, Training, and Evaluating the Model

How many neurons, layers, and activation functions did you select for your neural network model, and why?

For this neural network model, I chose four hidden layers with 40,80, 40, and 5 neurons respectively. After several iterations and tests with different numbers of neurons and layers, this combination produced the best results in terms of accuracy and loss. For the activation functions, I chose ReLU for the first hidden layer to introduce non-linearity in the model and improve its performance. I selected sigmoid for the second, tanh for the third and soft-max for the fourth to increase accuracy of the model. For the final output layer, I used sigmoid activation function to ensure the output is between 0 and 1, which is needed for binary classification.


Were you able to achieve the target model performance?

Yes, I was able to develop a successful deep neural network model using TensorFlow and Keras to predict if an Alphabet Soup-funded organization would be successful. Through several iterations for the optimal model, I was able to achieve a predictive accuracy higher than 75%, with a final accuracy score of 77.73%. This model could be a valuable tool for Alphabet Soup in selecting the applicants with the best chance of success in their ventures.


What steps did you take in your attempts to increase model performance?

During the optimization process, the EIN column was dropped as it was not relevant. However, keeping the NAME column improved model accuracy. To refine the data, a cutoff value was chosen, and names occurring less than 10 times were replaced with "Other". A similar approach was applied to the CLASSIFICATION column, where categories with fewer than 1000 occurrences were replaced with "Other". The resulting binning was verified for accuracy.


Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.
The deep learning model using TensorFlow and Keras achieved a 78% predictive accuracy in classifying the success of Alphabet Soup-funded organizations. The model underwent various optimization attempts, including column dropping, categorical variable binning, and adjusting layers and activation functions. While the target accuracy of 75% was eventually reached, it required significant optimization efforts.

To improve classification, alternative models like Random Forest Classifier or Support Vector Machine (SVM) can be explored. These models handle both numerical and categorical variables, outliers, and imbalanced datasets effectively. Considering these alternatives may offer a potential solution to enhance classification performance.

Step 5: Copy Files Into Your Repository
Now that you're finished with your analysis in Google Colab, you need to get your files into your repository for final submission.