import TFODStruct
import os
import tensorflow as tf
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection.utils import config_util

#
# FUNÇÃO QUE TREINA O MODELO.
#
def trainObjectDetectionModel(trainPipelinePath, numLabels, batchSize, numSteps,
                        warmupSteps, checkpointPath, modelType, labelMapPath,
                        trainRecordPath, testRecordPath, learningRateBase, warmupLearningRate):

    # Carrega-se o 'pipeline.config'.
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(trainPipelinePath, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    # Cálculo do número total de épocas realizadas.
    totalSteps = int(pipeline_config.train_config.num_steps) + numSteps

    pipeline_config.model.ssd.num_classes = numLabels  # Definição do número de classes a distinguir.
    pipeline_config.train_config.batch_size = batchSize  # Definição do tamanho do batch.

    # Configurações do optimizador.
    pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate. \
        cosine_decay_learning_rate.total_steps = totalSteps
    pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate. \
        cosine_decay_learning_rate.warmup_steps = int(pipeline_config.train_config.num_steps) + warmupSteps
    pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate. \
        cosine_decay_learning_rate.learning_rate_base = learningRateBase
    pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate. \
        cosine_decay_learning_rate.warmup_learning_rate = warmupLearningRate

    pipeline_config.train_config.num_steps = totalSteps # Definição do número total de épocas realizadas.
    pipeline_config.train_config.fine_tune_checkpoint = checkpointPath  # Definição do caminho até ao último checkpoint.
    pipeline_config.train_config.fine_tune_checkpoint_type = modelType  # Definição do tipo de modelo (deteção).
    pipeline_config.train_input_reader.label_map_path = labelMapPath  # Definição do caminho até ao label map.
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[
    :] = trainRecordPath  # Definição do caminho até ao record de treino.
    pipeline_config.eval_input_reader[0].label_map_path = labelMapPath  # Definição do caminho até ao label map.
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[
    :] = testRecordPath  # Definição do caminho até ao record de teste.

    # Escrevem-se as configurações no 'pipeline.config'
    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(TFODStruct.files['PIPELINE_CONFIG'], "wb") as f:
        f.write(config_text)

    # Corre-se o script de treino.
    TRAINING_SCRIPT = os.path.join(TFODStruct.paths['APIMODEL_PATH'], 'research', 'object_detection',
                                   'model_main_tf2.py')
    command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps={}". \
        format(TRAINING_SCRIPT, TFODStruct.paths['TRAIN_OUTPUT_PATH'], TFODStruct.files['PIPELINE_CONFIG'], totalSteps)
    os.system(command)


config = config_util.get_configs_from_pipeline_file(TFODStruct.files['PIPELINE_CONFIG'])
trainObjectDetectionModel(trainPipelinePath=TFODStruct.files['PIPELINE_CONFIG'], numLabels=len(TFODStruct.labels),
                          batchSize=4, numSteps=1000, warmupSteps=500,
                          checkpointPath=os.path.join(TFODStruct.paths['CHECKPOINT_PATH'], 'ckpt-10'),
                          modelType="detection", labelMapPath=TFODStruct.files['LABELMAP'],
                        trainRecordPath=[os.path.join(TFODStruct.paths['ANNOTATION_PATH'], 'train.record')],
                        testRecordPath=[os.path.join(TFODStruct.paths['ANNOTATION_PATH'], 'test.record')],
                          learningRateBase=0.03, warmupLearningRate=0.01)
# os.path.join(TFODStruct.paths['PRETRAINED_MODEL_PATH'], TFODStruct.PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
# os.path.join(TFODStruct.paths['CHECKPOINT_PATH'], 'ckpt-2')
