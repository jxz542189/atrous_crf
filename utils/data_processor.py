import codecs
import os
import jieba


data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')

class DataProcessor(object):
    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

    @classmethod
    def read_data(cls, input_file):
        word_res = []
        label_res = []
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                tokens = contends.split(' ')
                if len(tokens) == 2:
                    word = line.strip().split(' ')[0]
                    label = line.strip().split(' ')[-1]
                else:
                    if len(contends) == 0:
                        l = ''.join([label for label in labels if len(label) > 0])
                        w = ''.join([word for word in words if len(word) > 0])
                        assert len(words) == len(labels)
                        label_res.append(l)
                        word_res.append(w)
                        words = []
                        labels = []
                        continue
                if contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
                words.append(word)
                labels.append(label)
        print(len(word_res))
        return label_res, word_res


def get_seg_features(string):
    seg_feature = []
    for word in jieba.cut(string):
        if len(word) == 1:
            seg_feature.append(0)
        else:
            tmp = [2] * len(word)
            tmp[0] = 1
            tmp[-1] = 3
            seg_feature.extend(tmp)
    return seg_feature


def get_vocab_file(files, out_file='vocab.txt'):
    res = []
    word_dict = {}
    for file in files:
        with open(os.path.join(data_path, file)) as f:
            for line in f:
                contends = line.strip()
                tokens = contends.split(' ')
                if len(tokens) == 2:
                    word = line.strip().split(' ')[0]
                    if word in word_dict:
                        word_dict[word] += 1
                    else:
                        word_dict[word] = 1
    items = word_dict.items()
    items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
    words, _ = zip(*items_sorted)
    words =[ word + '\n' for word in list(words)]
    with open(os.path.join(data_path, out_file), 'w') as f:
        f.writelines(words)
    return words

if __name__ == '__main__':
    # keys = get_vocab_file(['example.train', 'example.test', 'example.dev'])
    # print(keys[:200])
    print(data_path)
    data = DataProcessor()
    label_res, word_res = data.read_data(os.path.join(data_path, "example.dev"))
    # print(len(lines))
    # print(lines[:2])
