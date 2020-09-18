
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import vectorize as vtrz
import split_data
import nlp
import data_processing as dp


def naive_bayes_train(train_features, train_classes):

    NBModel = MultinomialNB()
    NBModel.fit(train_features, train_classes.ravel())
    return NBModel


def naive_bayes_test(NBModel, test_features, test_classes):
    # print(np.shape(test_features))
    pred_classes = NBModel.predict(test_features)
    # print(np.shape(pred_classes))
    accuracy = np.mean(test_classes.ravel() == pred_classes)
    print("Naive Bayes prediction accuracy: " + str(accuracy))
    print("Sum of square difference: " + str(sum(np.square(test_classes.ravel() - pred_classes))))


def naive_bayes_classifier(features, classes):
    train_features, test_features, train_classes, test_classes = split_data.splitData(features, classes)
    NBModel = naive_bayes_train(train_features, train_classes)
    naive_bayes_test(NBModel, test_features, test_classes)
    return NBModel


def new_input_classify(NBModel, wordDict, pca_model=None):

    review = input("Give me a review to be rated (press enter to terminate): ")
    review = [review]
    while review != [""]:
        feature_vector = vtrz.get_vectors_new_sentences(pca_model, review, wordDict)
        # feature_vector = np.reshape(feature_vector, (1, len(feature_vector)))
        pred_class = NBModel.predict(feature_vector)
        print("The predicted rate is: " + str(pred_class))
        review = input("Give me a review to be rated (press enter to terminate): ")
        review = [review]


def classify_list_data(NBModel, wordDict, pca_model, sentence_list, class_list):
    feature_vectors = vtrz.get_vectors_new_sentences(pca_model, sentence_list, wordDict)
    # print(feature_vector.shape)
    # feature_vector = np.reshape(feature_vector, (1, len(feature_vector)))
    pred_classes = NBModel.predict(feature_vectors)
    accuracy = np.mean(class_list.ravel() == pred_classes)
    print("Naive Bayes prediction accuracy: " + str(accuracy))
    print("Sum of square difference: " + str(sum(np.square(class_list.ravel() - pred_classes))))

    return pred_classes

files = ["data/Generator_nGram/bi_1stars.h5_reviews.json",
         "data/Generator_nGram/bi_2stars.h5_reviews.json",
         "data/Generator_nGram/bi_3stars.h5_reviews.json",
         "data/Generator_nGram/bi_4stars.h5_reviews.json",
         "data/Generator_nGram/bi_5stars.h5_reviews.json",
         "data/Generator_nGram/tri_1stars.h5_reviews.json",
         "data/Generator_nGram/tri_2stars.h5_reviews.json",
         "data/Generator_nGram/tri_3stars.h5_reviews.json",
         "data/Generator_nGram/tri_4stars.h5_reviews.json",
         "data/Generator_nGram/tri_5stars.h5_reviews.json"]

if __name__ == "__main__":
    # import time
    # start_time = time.time()
    features, classes, wordDict, pca_model = vtrz.get_vectors()
    NBModel = naive_bayes_classifier(features, classes)
    print()
    # new_input_classify(NBModel, wordDict, pca_model)
    for file in files:
        print(file + ": ")
        data = dp.load_file(file)
        n = len(data)
        sentences = []
        labels = np.zeros((n, 1))
        for i in range(n):
            sentences.append(data[i]["text"])
            labels[i] = data[i]["stars"]

        pred_classes = classify_list_data(NBModel, wordDict, pca_model, sentences, labels)
        print(pred_classes)

        # end_time = time.time()
        # print(end_time - start_time)