from keras.models import Sequential
from keras import layers
import vectorize as vtrz
import split_data
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import nlp
import numpy as np
import matplotlib.pyplot as plt
import data_processing as dp
plt.style.use('ggplot')


# helper function to make the plot
def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def neural_net_classifier(features, classes, epoch):
    train_features, test_features, train_classes, test_classes = split_data.splitData(features, classes)

    test_classes_copy = test_classes.copy()

    train_classes = train_classes.ravel()
    test_classes = test_classes.ravel()

    encoder = LabelEncoder()
    encoder.fit(train_classes)
    train_classes = encoder.transform(train_classes)
    train_classes = np_utils.to_categorical(train_classes)

    encoder.fit(test_classes)
    test_classes = encoder.transform(test_classes)
    test_classes = np_utils.to_categorical(test_classes)

    features_dim = train_features.shape[1]
    NNModel = Sequential()
    NNModel.add(layers.Dense(10, input_dim=features_dim, activation='relu'))
    NNModel.add(layers.Dense(5, activation='sigmoid'))
    NNModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    NNModel.summary()

    history = NNModel.fit(train_features, train_classes, epochs=epoch, verbose=False, validation_data=(test_features, test_classes), batch_size=10)
    loss_train, accuracy_train = NNModel.evaluate(train_features, train_classes, verbose='False')
    print("Training Accuracy: {:.4f}".format(accuracy_train))
    loss_test, accuracy_test = NNModel.evaluate(test_features, test_classes, verbose='False')
    print("Testing Accuracy:  {:.4f}".format(accuracy_test))
    # print(test_classes.ravel().shape)
    # print(test_features.shape)
    # print(NNModel.predict_classes(test_features).shape)
    print("Sum of square difference: " + str(sum(np.square(test_classes_copy.ravel() - NNModel.predict_classes(test_features)))))

    plot_history(history)


    return NNModel



def classify_new_data(NNModel, wordDict, pca_model):
    review = input("Give me a review to be rated (press enter to terminate): ")
    review = [review]
    while review != [""]:
        feature_vector = vtrz.get_vectors_new_sentences(pca_model, review, wordDict)
        # print(feature_vector.shape)
        # feature_vector = np.reshape(feature_vector, (1, len(feature_vector)))
        pred_class = NNModel.predict_classes(feature_vector)
        print("The predicted rate is: " + str(pred_class+1))
        review = input("Give me a review to be rated (press enter to terminate): ")
        review = [review]

def classify_list_data(NNModel, wordDict, pca_model, sentence_list, class_list):
    feature_vectors = vtrz.get_vectors_new_sentences(pca_model, sentence_list, wordDict)
    # print(feature_vector.shape)
    # feature_vector = np.reshape(feature_vector, (1, len(feature_vector)))
    pred_classes = NNModel.predict_classes(feature_vectors)
    pred_classes = pred_classes+1
    accuracy = np.mean(class_list.ravel() == pred_classes)
    print("Neural Networks prediction accuracy: " + str(accuracy))
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
    features, classes, wordDict, pca_model = vtrz.get_vectors(100)
    NNModel = neural_net_classifier(features, classes, epoch=50)
    classify_new_data(NNModel, wordDict, pca_model)
    # print()
    # for file in files:
    #     print(file + ": ")
    #     data = dp.load_file(file)
    #     n = len(data)
    #     sentences = []
    #     labels = np.zeros((n, 1))
    #     for i in range(n):
    #         sentences.append(data[i]["text"])
    #         labels[i] = data[i]["stars"]
    #
    #     pred_classes = classify_list_data(NNModel, wordDict, pca_model, sentences, labels)
    #     print(pred_classes)

        # end_time = time.time()
        # print(end_time - start_time)