import TFODStruct
import os
import shutil
import tensorflow as tf
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection.utils import config_util
from object_detection.builders import model_builder

#
# FUNÇÃO QUE LÊ O CONTEÚDO DE UM CHECKPOINT.
#
def readCheckPointContent(checkpointPath):

    model_config = config['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Carrega-se o checkpoint.
    checkpoint = tf.compat.v2.train.Checkpoint(model=detection_model)
    checkpoint.restore(checkpointPath).expect_partial()

    # Carregam-se as variáveis desejadas.
    reader = tf.train.load_checkpoint(checkpointPath)
    try:
        value = reader.get_tensor("optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE")
    except:
        value = 0

    return value

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

    pipeline_config.model.ssd.num_classes = numLabels  # Definição do número de classes a distinguir.
    pipeline_config.train_config.batch_size = batchSize  # Definição do tamanho do batch.

    # Configurações do optimizador.
    pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate. \
        cosine_decay_learning_rate.total_steps = numSteps
    pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate. \
        cosine_decay_learning_rate.warmup_steps = warmupSteps
    pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate. \
        cosine_decay_learning_rate.learning_rate_base = learningRateBase
    pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate. \
        cosine_decay_learning_rate.warmup_learning_rate = warmupLearningRate

    pipeline_config.train_config.num_steps = numSteps # Definição do número de épocas a realizar.
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

    totalSteps = readCheckPointContent(checkpointPath=checkpointPath)
    totalSteps = totalSteps + numSteps

    # Corre-se o script de treino.
    TRAINING_SCRIPT = os.path.join(TFODStruct.paths['APIMODEL_PATH'], 'research', 'object_detection',
                                   'model_main_tf2.py')
    command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps={}". \
        format(TRAINING_SCRIPT, TFODStruct.paths['TRAIN_OUTPUT_PATH'], TFODStruct.files['PIPELINE_CONFIG'], totalSteps)
    os.system(command)


config = config_util.get_configs_from_pipeline_file(TFODStruct.files['PIPELINE_CONFIG'])
trainObjectDetectionModel(trainPipelinePath=TFODStruct.files['PIPELINE_CONFIG'], numLabels=len(TFODStruct.labels),
                          batchSize=4, numSteps=200, warmupSteps=100,
                          checkpointPath=os.path.join(TFODStruct.paths['CHECKPOINT_PATH'], 'ckpt-6'),
                          modelType="detection", labelMapPath=TFODStruct.files['LABELMAP'],
                        trainRecordPath=[os.path.join(TFODStruct.paths['ANNOTATION_PATH'], 'train.record')],
                        testRecordPath=[os.path.join(TFODStruct.paths['ANNOTATION_PATH'], 'test.record')],
                          learningRateBase=0.03, warmupLearningRate=0.01)

# Lista todos os arquivos no diretório de saída do processo de treino.
all_files = os.listdir(TFODStruct.paths["TRAIN_OUTPUT_PATH"])
# Carregam-se apenas os arquivos que terminam com ".index".
checkpoint_files_index = [file for file in all_files if file.endswith(".index")]
# Carregam-se apenas os arquivos que terminam com ".data".
checkpoint_files = [file for file in all_files if file.endswith(".data-00000-of-00001")]

# Se houver checkpoints copia-se o último '.index' para o diretório desejado.
#if checkpoint_files_index:
    #latest_checkpoint = max(checkpoint_files_index)

    #source_checkpoint = os.path.join(TFODStruct.paths["TRAIN_OUTPUT_PATH"], latest_checkpoint)
    #target_checkpoint = os.path.join(TFODStruct.paths["CHECKPOINT_PATH"], latest_checkpoint)

    # Copia o arquivo de checkpoint e seus arquivos associados
    #shutil.copytree(source_checkpoint, target_checkpoint)

# Se houver checkpoints copia-se o último '.data' para o diretório desejado.
#if checkpoint_files:
    #latest_checkpoint = max(checkpoint_files)

    #source_checkpoint = os.path.join(TFODStruct.paths["TRAIN_OUTPUT_PATH"], latest_checkpoint)
    #target_checkpoint = os.path.join(TFODStruct.paths["CHECKPOINT_PATH"], latest_checkpoint)

    # Copia o arquivo de checkpoint e seus arquivos associados
    #shutil.copytree(source_checkpoint, target_checkpoint)