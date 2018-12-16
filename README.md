IDCNN+CRF是根据[Fast and Accurate Entity Recognition with Iterated Dilated Convolutions](https://arxiv.org/abs/1702.02098) 编写的

#### 亮点

+ 支持多GPU训练

+ 使用tf.data和tf.estimator编写

+ 支持tensorflow-serving

+ 使用虫洞卷积加CRF编写

#### 存在一个bug

  ```python
  NotFoundError (see above for traceback): Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint.
  ```

#### 参看

[Fast and Accurate Entity Recognition with Iterated Dilated Convolutions](https://arxiv.org/abs/1702.02098)

https://github.com/crownpku/Information-Extraction-Chinese/tree/master/NER_IDCNN_CRF