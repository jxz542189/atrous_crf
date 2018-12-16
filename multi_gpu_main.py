import tensorflow as tf
from tensorflow import data
from datetime import datetime
import multiprocessing
import json
import shutil
import os
from model.model import Model
from utils.data_processor import DataProcessor, get_seg_features
import numpy as np
from utils import tf_metrics



tf.reset_default_graph()
path = os.path.dirname(os.path.realpath(__file__))
print(path)
config_path = os.path.join(path, 'config')
params_path = os.path.join(config_path, 'params.json')
with open(params_path) as param:
    params_dict = json.load(param)
config = tf.contrib.training.HParams(**params_dict)
os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_visible_devices
model_dir = 'trained_models/{}'.format(config.model_name)
total_steps = int((config.train_size/config.batch_size) * config.num_epochs)
test_steps = int((config.num_test_size/config.batch_size))
target_labels = ['O','B-PER','I-PER','B-ORG','I-ORG','I-LOC','B-LOC']
PAD_WORD = '<pad>'
UNK = '<unk>'
N_WORDS = 4467
EVAL_AFTER_SEC = 60
RESUME_TRAINING = False
VOCAB_LIST_FILE = os.path.join(path, "data", "vocab.txt")
label_dict = {}
for i, label in enumerate(target_labels):
    label_dict[label] = i
word_dict = {}
k = 0
with open(VOCAB_LIST_FILE) as f:
    for word in f:
        word = word.strip()
        word_dict[word] = k
        k += 1


def input_fn(filename, mode=tf.estimator.ModeKeys.EVAL,
             num_epochs=1,
             batch_size=32):
    labels, lines = DataProcessor().read_data(filename)
    shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False
    num_threads = multiprocessing.cpu_count()
    buffer_size = 2 * batch_size + 1
    print("")
    print("* data input_fn:")
    print("================")
    print("Batch size: {}".format(batch_size))
    print("Epoch Count: {}".format(num_epochs))
    print("Mode: {}".format(mode))
    print("Thread Count: {}".format(num_threads))
    print("Shuffle: {}".format(shuffle))
    print("================")
    print("")
    max_seq_length = config.max_seq_length
    labels_id = []
    for label in labels:
        label_list = []
        for ch in label:
            if ch in label_dict:
                label_list.append(label_dict[ch])
            else:
                label_list.append(label_dict['O'])
        label_list = label_list[:max_seq_length] if len(label_list) >= max_seq_length else label_list + [label_dict['O']] * (
                    max_seq_length - len(label_list))
        labels_id.append(np.array(label_list, dtype=np.int32))
    words_id = []
    segs_id = []
    lengths = []
    for line in lines:
        seg_id = get_seg_features(line)
        word_id = []
        for word in line:
            if word in word_dict:
                word_id.append(word_dict[word])
            else:
                word_id.append(word_dict[UNK])
        lengths.append(len(word_id))
        seg_id = seg_id[:max_seq_length] if len(seg_id) >= max_seq_length else seg_id + [2] * (max_seq_length - len(seg_id))
        word_id = word_id[:max_seq_length] if len(word_id) >= max_seq_length else word_id + [word_dict[PAD_WORD]] * (max_seq_length - len(word_id))
        segs_id.append(np.array(seg_id, dtype=np.int32))
        words_id.append(np.array(word_id, dtype=np.int32))
        assert len(seg_id) == len(word_id)
    assert len(words_id) == len(labels_id)
    #words_id:(None,max_seq_length) segs_id:(None, max_seq_length) lengths:(None)
    res = np.concatenate([np.array(words_id), np.array(segs_id), np.reshape(np.array(lengths), (-1, 1))], axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(({"instances":res}, np.array(labels_id, dtype=np.int32)))
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.repeat(None)
    else:
        dataset = dataset.repeat(1)
    dataset = dataset.prefetch(buffer_size)
    return dataset


def model_fn(features, labels, mode, params):
    max_seq_length = params.max_seq_length
    words_id, segs_id, lengths = tf.split(features['instances'], axis=-1, num_or_size_splits=[max_seq_length, max_seq_length, 1])
    lengths = tf.squeeze(lengths)
    model = Model(words_id, segs_id, labels, lengths, params)
    if mode == tf.estimator.ModeKeys.PREDICT:
        paths = model.decode(model.logits, model.lengths, model.trans)
        predictions = {
            'predictions': paths
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode,
                                          predictions=predictions,
                                          export_outputs=export_outputs)

    model.get_loss(model.logits, labels, lengths)
    tf.summary.scalar('loss', model.loss)
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=model.loss,
                                          train_op=model.train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        def metric_fn(label_ids, logits, trans):
            pred_ids = model.decode(model.logits, model.lengths, model.trans)
            pred_ids = tf.cast(pred_ids, tf.int32)
            precision = tf_metrics.precision(label_ids, pred_ids, len(target_labels), [1, 2, 3, 4, 5, 6])
            recall = tf_metrics.recall(label_ids, pred_ids, len(target_labels), [1, 2, 3, 4, 5, 6])
            f = tf_metrics.f1(label_ids, pred_ids, len(target_labels), [1, 2, 3, 4, 5, 6])

            return {
                "eval_precision": precision,
                "eval_recall": recall,
                "eval_f": f,
                # "eval_loss": loss,
            }

        eval_metrics = metric_fn(labels, model.logits, model.trans)
        return  tf.estimator.EstimatorSpec(
            mode=mode,
            loss=model.loss,
            eval_metric_ops=eval_metrics)


def create_estimator(run_config, hparams):
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       params=hparams,
                                       config=run_config)
    print("")
    print("Estimator Type: {}".format(type(estimator)))
    print("")

    return estimator


def serving_input_fn():
    receiver_tensor = {
        'instances': tf.placeholder(tf.int32, [None, None])
    }
    features = {
        key: tensor
        for key, tensor in receiver_tensor.items()
    }

    return tf.estimator.export.ServingInputReceiver(
        features, receiver_tensor)



if __name__ == '__main__':
    # ==============另一训练方式===============
    if not RESUME_TRAINING:
        print("Removing previous artifacts...")
        shutil.rmtree(model_dir, ignore_errors=True)
    else:
        print("Resuming training...")
    distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=4)

    # run_config = tf.estimator.RunConfig(log_step_count_steps=config.train['log_step_count_steps'],
    #                                     tf_random_seed=config.train['tf_random_seed'],
    #                                     model_dir=model_dir,
    #                                     )

    run_config = tf.estimator.RunConfig(log_step_count_steps=config.log_step_count_steps,
                                        tf_random_seed=config.tf_random_seed,
                                        model_dir=model_dir,
                                        session_config=tf.ConfigProto(allow_soft_placement=True,
                                                                      log_device_placement=True),
                                        train_distribute=distribution)
    estimator = create_estimator(run_config, config)
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(os.path.join(path, 'data', 'example.train'),
                                  mode=tf.estimator.ModeKeys.TRAIN,
                                  num_epochs=config.num_epochs,
                                  batch_size=config.batch_size),
        max_steps=total_steps,
        hooks=None
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(os.path.join(path, 'data', 'example.dev'),
                                  mode=tf.estimator.ModeKeys.EVAL,
                                  batch_size=config.batch_size),
        exporters=[tf.estimator.LatestExporter(name="predict",
                                               serving_input_receiver_fn=serving_input_fn,
                                               exports_to_keep=1,
                                               as_text=True)],
        steps=test_steps,
        throttle_secs=EVAL_AFTER_SEC
    )
    # eval_spec = tf.estimator.EvalSpec(
    #     input_fn=lambda: input_fn(os.path.join(path, 'data', 'example.dev'),
    #                               mode=tf.estimator.ModeKeys.EVAL,
    #                               batch_size=config.batch_size),
    #     steps=None,
    #     throttle_secs=EVAL_AFTER_SEC
    # )
    tf.estimator.train_and_evaluate(estimator=estimator,
                                    train_spec=train_spec,
                                    eval_spec=eval_spec)


