# -*- coding:utf-8 -*-

import nltk
import jieba
import codecs
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


# 文本预处理
def sent_tokenizer(texts):
    start = 0
    i = 0  # 每个字符的位置
    sentences = []
    punt_list = ".!?。！？；"
    for text in texts:
        if i < len(texts) - 1:
            if text in punt_list:
                sentences.append(texts[start:i + 1])  # 当前标点符号位置
                start = i + 1  # start标记到下一句的开头
                i += 1
            else:
                i += 1  # 若不是标点符号，则字符位置继续前移
        else:
            sentences.append(texts[start:])  # 处理文本末尾没有标点符号的情况
            break
    return sentences


# 停用词
def load_stop_words_list(path):
    stop_list = [
        line.strip()
        for line in codecs.open(path, 'r', encoding='utf8').readlines()
    ]
    stop_words = {}.fromkeys(stop_list)
    return stop_words


# 摘要
def summarize(text):
    stopwords = load_stop_words_list("stopwords.txt")
    sentences = sent_tokenizer(text)
    words = [
        w for sentence in sentences for w in jieba.cut(sentence)
        if w not in stopwords if len(w) > 1 and w != '\t'
    ]
    word_frequency = nltk.FreqDist(words)
    top_words = [
        w[0] for w in sorted(
            word_frequency.items(), key=lambda d: d[1], reverse=True)
    ][:n]
    scored_sentences = _score_sentences(sentences, top_words)
    # 利用均值和标准差过滤非重要句子
    avg = np.mean([s[1] for s in scored_sentences])
    std = np.std([s[1] for s in scored_sentences])
    mean_scored = [(sent_idx, score) for (sent_idx, score) in scored_sentences
                   if score > (avg + 1.5 * std)]
    top_n_scored = sorted(scored_sentences,
                          key=lambda s: s[1])[-num_top_sentence:]
    top_n_scored = sorted(top_n_scored, key=lambda s: s[0])
    return dict(
        top_n_summary=[sentences[idx] for (idx, score) in top_n_scored],
        mean_scored_summary=[sentences[idx] for (idx, score) in mean_scored])


# 句子得分
def _score_sentences(sentences, top_words):
    scores = []
    sentence_idx = -1
    for s in [list(jieba.cut(s)) for s in sentences]:
        sentence_idx += 1
        word_idx = []
        for index in range(len(s)):
            if s[index] in top_words:
                word_idx.append(index)
            else:
                pass
        word_idx.sort()
        if len(word_idx) == 0:
            continue
        # 对于两个连续的单词，利用单词位置索引，通过距离阈值计算簇
        clusters = []
        cluster = [word_idx[0]]
        i = 1
        while i < len(word_idx):
            if word_idx[i] - word_idx[i - 1] < cluster_distance:
                cluster.append(word_idx[i])
            else:
                clusters.append(cluster[:])
                cluster = [word_idx[i]]
            i += 1
        clusters.append(cluster)
        # 对每个簇打分，每个簇类的最大分数是对句子的打分
        max_cluster_score = 0
        for c in clusters:
            key_words = len(c)
            total_words = c[-1] - c[0] + 1
            score = key_words * key_words / total_words
            if score > max_cluster_score:
                max_cluster_score = score
        scores.append((sentence_idx, max_cluster_score))
    return scores


# TextRank算法
def text_rank_summarize(text):
    stopwords = load_stop_words_list('E:/Python Files/stopwords/中文停用词表.txt')
    sentences = sent_tokenizer(text)
    words_list = []
    for sentence in sentences:
        word_list = [
            w for w in jieba.cut(sentence) if w not in stopwords
            if len(w) > 1 and w != '\t'
        ]
        words_list.append(word_list)

    # 加载word2vec词向量
    word_embeddings = {}
    with codecs.open("sgns.renmin.char", encoding='utf-8') as f:
        for line in f:
            # 把第一行的内容去掉
            if '355996 300\n' not in line:
                values = line.split()
                word = values[0]
                embedding = np.asarray(values[1:], dtype='float32')
                word_embeddings[word] = embedding
        f.close()

    # 得到句子的向量表示
    sentence_vectors = []
    for word_list in words_list:
        if len(word_list) != 0:
            # 如果句子中的词语不在字典中，那就把embedding设为300维元素为0的向量。
            # 得到句子中全部词的词向量后，求平均值，得到句子的向量表示
            v = sum(
                [word_embeddings.get(w, np.zeros([
                    300,
                ])) for w in word_list]) / (len(word_list))
        else:
            # 如果句子为[]，那么就向量表示为300维元素为0个向量。
            v = np.zeros([
                300,
            ])
        sentence_vectors.append(v)

    # 计算句子之间的余弦相似度，构成相似度矩阵
    sim_mat = np.ones([len(words_list), len(words_list)])
    for i in range(len(words_list)):
        for j in range(len(words_list)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(
                    sentence_vectors[i].reshape(1, 300),
                    sentence_vectors[j].reshape(1, 300))[0, 0]

    # 利用句子相似度矩阵构建图结构，句子为节点，句子相似度为转移概率
    nx_graph = nx.from_numpy_array(sim_mat)

    # 得到所有句子的TextRank值
    scores = nx.pagerank(nx_graph)

    # 根据TextRank值对句子进行排序
    ranked_sentences = sorted(
        ((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    result = []
    for i in range(num_top_sentence):
        result.append(ranked_sentences[i][1])

    return result


# 读取文本文件
def read_text(path):
    with codecs.open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


if __name__ == '__main__':
    n = 30  # 关键词数量
    cluster_distance = 5  # 单词间的距离
    num_top_sentence = 5  # 返回句子的数量
    text = read_text("text.txt")
    algorithm1 = summarize(text)
    algorithm2 = text_rank_summarize(text)
    print('--------------top_n_summary----------------')
    for sent in algorithm1['top_n_summary']:
        print(sent)
    print('-----------mean_scored_summary-------------')
    for sent in algorithm1['mean_scored_summary']:
        print(sent)
    print('----------------TextRank-------------------')
    for sent in algorithm2:
        print(sent)
