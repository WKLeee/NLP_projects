import split_data
from sklearn.linear_model import LogisticRegression
import vectorize as vtrz
import nlp
import numpy as np
import data_processing as dp


def logistic_classifier(features, classes):
    train_features, test_features, train_classes, test_classes = split_data.splitData(features, classes)
    logClassifier = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)
    logClassifier.fit(train_features, train_classes.ravel())
    accuracy = logClassifier.score(test_features, test_classes.ravel())
    print("Logistic regression classifier prediction accuracy: " + str(accuracy))
    print("Sum of square difference: " + str(sum(np.square(test_classes.ravel() - logClassifier.predict(test_features)))))
    return logClassifier


def new_input_classify(logClassifier, wordDict, pca_model):
    review = input("Give me a review to be rated (press enter to terminate): ")
    review = [review]
    while review != [""]:
        feature_vector = vtrz.get_vectors_new_sentences(pca_model, review, wordDict)
        # feature_vector = np.reshape(feature_vector, (1, len(feature_vector)))
        pred_class = logClassifier.predict(feature_vector)
        print("The predicted rate is: " + str(pred_class))
        review = input("Give me a review to be rated (press enter to terminate): ")
        review = [review]

def classify_list_data(logClassifier, wordDict, pca_model, sentence_list, class_list):
    feature_vectors = vtrz.get_vectors_new_sentences(pca_model, sentence_list, wordDict)
    # print(feature_vector.shape)
    # feature_vector = np.reshape(feature_vector, (1, len(feature_vector)))
    pred_classes = logClassifier.predict(feature_vectors)
    accuracy = np.mean(class_list.ravel() == pred_classes)
    print("Logistic Regression prediction accuracy: " + str(accuracy))
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

if __name__ == '__main__':
    import time
    start_time = time.time()
    features, classes, wordDict, pca_model = vtrz.get_vectors(1000)
    logClassifier = logistic_classifier(features, classes)
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

        pred_classes = classify_list_data(logClassifier, wordDict, pca_model, sentences, labels)
        print(pred_classes)

        # end_time = time.time()
        # print(end_time - start_time)