# About

This is an AI project consist of two parts:

1. (Rater)Train a model that can rate the review of a restaurant(by returning a predicted associate number of stars from 1~5).
2. (Generator)Generate a review in complete sentences given the rating(a number of stars) and some starting words.

# Rater

## Data

Our review data are downloaded from 

https://www.kaggle.com/yelp-dataset/yelp-dataset#yelp_academic_dataset_review.json

https://www.kaggle.com/yelp-dataset/yelp-dataset#yelp_academic_dataset_business.json

The filtered data which contains only the reviews of restaurants can be downloaded from the Google Drive.

Put `restaurant_reviews.json` and `mini_reviews.json` into the `Rater/data/` directory.

## Data-preprocessing & Vectorization

The reviews are stemmed and lemmatized first. Then we collect all distinct words(remove stopwords and non-English words) from reviews. For each review i, there is a vector Vi that represents the frequencies of each word, i.e Vi,j means word j occurs Vi,j times in review i.

### Dimension reduction

Here PCA is used for dimension reduction. Now `vtrz.get_vectors()` accepts an argument `n_dimension` as the number of the dimension after the reduction.

`n_dimension` can be an integer or a string "mle". "mle" only works when the number of the samples are greater than the number of the features.

Sample code

    import vectorize as vtrz
    X, Y, word_dict, pca_model = vtrz.get_vectors() # X, Y are both numpy.ndarray
    X, Y, word_dict, pca_model = vtrz.get_vectors(200) # If you want a reduction.
    X, Y, word_dict, pca_model = vtrz.get_vectors("mle") # If you want a reduction with the lowest variance.


For those return values, `X` and `Y` are the inputs and outputs. `word_dict` is the instance of class `WordDict`, and `word_dict.wordmap` is the dictionary that stores the index of each word. `pca_model` is the dimension reduction model fitted by train set.

For the test set, you can also transform them into vectors by calling `vtrz.get_vectors_new_sentences(n_dimension, pca_model, new_sentences, word_dict)`, where the `new_sentences` is the list of the reviews in the test set.

## Naive-Bayes Classification

By following the convention, we will split the data into 2/3 of training data and 1/3 of testing data. Then, we will firstly use naive bayes classifier to classify the reviews. The accuracy of naive bayes' performance on the test data will be printed out.

Sample Code (X, Y from above)
```
import naive_bayes as nb
NBModel = nb.naive_bayes_classifier(X, Y)
```

You can also play around by giving new reviews and let the rater predict the rate. (Currently only available for features generated without PCA)
```
nb.new_input_classify(NBModel, wordDict, pca_model)
```

## Logistic Regression Classification
Sample Code
````
import logistic_regression as lgreg
logClassifier = lgreg.logistic_classifier(X, Y)
````

Predict rate for review given by the user manually:
````
lgreg.new_input_classify(logClassifier, wordDict, pca_model)
````

## Feed-Forward Neural Net Classification
Sample Code (you can try different numbers for epoch, here I use 50)
````
import neural_net as nn
NNClassifier = nn.neural_net_classifier(X, Y, epoch=50)
````

Predict rate for review given by the user manually:
````
nn.new_input_classify(NNClassifier, wordDict, pca_model)
````


# Generator

Implemented using RNN
