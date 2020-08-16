# -*- coding: utf-8 -*-

from gensim.models import word2vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
import logging

vec_dim = 200


def text8Train():
    print("start train word2vec model")
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus(u"text8")

    model = word2vec.Word2Vec(min_count=0, window=10,
                              size=vec_dim, workers=3, iter=10)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=10)
    model.save("model.bin")
    # y1 = model.self.wv.similarity("public", "private")
    # print('Similarity between "public" and "private":', y1)


def moreTrain():
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    new_model = word2vec.Word2Vec.load('new_model.bin')
    more_sentences = word2vec.Text8Corpus(u"addcode-1.txt")

    new_model.build_vocab(more_sentences, update=True)
    new_model.train(
        more_sentences, total_examples=new_model.corpus_count, epochs=5)
    print(new_model['ergb'])
    new_model.save('new_model.bin')


def printResult():
    model = word2vec.Word2Vec.load('model.bin')
    X = model.wv[model.wv.vocab]

    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    pyplot.scatter(result[:, 0], result[:, 1])
    words = list(model.wv.vocab)

    print(len(words))
    print('The most similar word to "device":',
          model.wv.most_similar(u"device"))

    for i, word in enumerate(words):
        # for (syn, num) in model.wv.most_similar(u"device"):
        print(word)
        if (word == 'device' or word == 'devices' or word == 'controller'):
            pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()


if __name__ == "__main__":
    # text8Train()
    # moreTrain()
    printResult()
