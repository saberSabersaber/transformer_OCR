# transformer_OCR
利用transformer 进行ocr识别。项目主题框架参考deep-text-recognition-benchmark（https://github.com/clovaai/deep-text-recognition-benchmark），训练和测试数据在deep-text-recognition-benchmark 项目中可以进行下载，Transformer 部分 参考pytorch-seq2seq(https://github.com/bentrevett/pytorch-seq2seq)
transformer 结构采用 pytorch-seq2seq 中的demo 参数配置，利用deep-text-recognition-benchmark 中模型对backbone部分进行初始化，平均acc 达到0.85.

# 项目依赖
PyTorch 1.3.1 CUDA 10.1, python 3.6 and Ubuntu 16.04 lmdb, pillow, torchvision, nltk, natsort

# 模型地址
链接：https://pan.baidu.com/s/1RzWpU_0-OQcezTKuMQqmUA 
提取码：olze 

# 训练测试
CUDA_VISIBLE_DEVICES=0 python3 train.py \
--train_data data_lmdb_release/training --valid_data data_lmdb_release/validation \
--select_data MJ-ST --batch_ratio 0.5-0.5 \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling None --Prediction Transformer

CUDA_VISIBLE_DEVICES=0 python3 test.py \
--eval_data data_lmdb_release/evaluation --benchmark_all_eval \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling None --Prediction Transformer \
--saved_model TPS-ResNet-None-Transformer.pth

# 模型效果
accuracy: IIIT5k_3000: 87.567 SVT: 87.172 IC03_860: 95.465 IC03_867: 94.810 IC13_857: 93.816 IC13_1015: 92.414 IC15_1811: 77.361 IC15_2077: 74.506 SVTP: 78.915 CUTE80: 73.519 total_accuracy: 85.039 averaged_infer_time: 28.099 # parameters: 58.723

# todo
由于transformer 结构只是采用了 pytorch-seq2seq 中的demo 参数配置，参数配置还有一定的调优空间，以及学习率策略、优化器等都还需实验进一步尝试。
