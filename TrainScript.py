import TFODStruct
import os
import shutil
import tensorflow as tf
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

#
# FUNÇÃO QUE LISTA AS VARIÁVEIS DE UM CHECKPOINT.
#
def printCheckPointVariables(checkpointPath):

    reader = tf.train.load_checkpoint(checkpointPath)
    var_names = reader.get_variable_to_shape_map().keys()

    for var_name in var_names:
        var_value = reader.get_tensor(var_name)
        print(f"Variable name: {var_name}")
        print("Variable value:\n", var_value)
        print("=" * 50)

#
# FUNÇÃO QUE LÊ A QUANTIDADE DE STEPS DE UM CHECKPOINT.
#
def readCheckPointVariables(checkpointPath):

    reader = tf.train.load_checkpoint(checkpointPath)
    try:
        steps = reader.get_tensor("optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE")
        checkpointCounter = reader.get_tensor("save_counter/.ATTRIBUTES/VARIABLE_VALUE")
    except:
        steps = 0
        checkpointCounter = 0

    return steps, checkpointCounter

#
# FUNÇÃO QUE GUARDA NO DISCO O ÚLTIMO CHECKPOINT.
#
def saveLastCheckPoint():

    for item in os.listdir(TFODStruct.paths["CHECKPOINT_PATH"]):
        item_path = os.path.join(TFODStruct.paths["CHECKPOINT_PATH"], item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

    all_files = os.listdir(TFODStruct.paths["TRAIN_OUTPUT_PATH"])
    checkpoint_files_index = [file for file in all_files if file.endswith(".index")]
    checkpoint_files = [file for file in all_files if file.endswith(".data-00000-of-00001")]

    if checkpoint_files_index:
        latest_checkpoint = max(checkpoint_files_index)

        source_checkpoint = os.path.join(TFODStruct.paths["TRAIN_OUTPUT_PATH"], latest_checkpoint)
        target_checkpoint = os.path.join(TFODStruct.paths["CHECKPOINT_PATH"], 'latest-ckpt.index')

        shutil.copy(source_checkpoint, target_checkpoint)

    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files)

        source_checkpoint = os.path.join(TFODStruct.paths["TRAIN_OUTPUT_PATH"], latest_checkpoint)
        target_checkpoint = os.path.join(TFODStruct.paths["CHECKPOINT_PATH"], 'latest-ckpt.data-00000-of-00001')

        shutil.copy(source_checkpoint, target_checkpoint)


#
# FUNÇÃO QUE TREINA O MODELO E CONFIGURA A PIPELINE DE TREINO.
#
def trainSSDMobNet(batchSize, numSteps, warmupSteps, learningRateBase, warmupLearningRate):

    checkpointSteps = readCheckPointVariables(checkpointPath=
                                              os.path.join(TFODStruct.paths['CHECKPOINT_PATH'], 'latest-ckpt'))
    totalSteps = checkpointSteps[0] + numSteps
    totalWarmUpSteps = checkpointSteps[0] + warmupSteps

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(TFODStruct.files['PIPELINE_CONFIG'], "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    pipeline_config.model.ssd.num_classes = len(TFODStruct.labels)
    pipeline_config.train_config.batch_size = batchSize

    pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate. \
        cosine_decay_learning_rate.total_steps = totalSteps
    pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate. \
        cosine_decay_learning_rate.warmup_steps = totalWarmUpSteps
    pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate. \
        cosine_decay_learning_rate.learning_rate_base = learningRateBase
    pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate. \
        cosine_decay_learning_rate.warmup_learning_rate = warmupLearningRate

    pipeline_config.train_config.num_steps = totalSteps
    pipeline_config.train_config.fine_tune_checkpoint = \
        os.path.join(TFODStruct.paths['CHECKPOINT_PATH'], 'latest-ckpt')
    pipeline_config.train_config.fine_tune_checkpoint_type = 'detection'
    pipeline_config.train_input_reader.label_map_path = TFODStruct.files['LABELMAP']
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = \
        [os.path.join(TFODStruct.paths['ANNOTATION_PATH'], 'train.record')]
    pipeline_config.eval_input_reader[0].label_map_path = TFODStruct.files['LABELMAP']
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = \
        [os.path.join(TFODStruct.paths['ANNOTATION_PATH'], 'test.record')]

    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(TFODStruct.files['PIPELINE_CONFIG'], "wb") as f:
        f.write(config_text)

    TRAINING_SCRIPT = os.path.join(TFODStruct.paths['APIMODEL_PATH'], 'research', 'object_detection',
                                   'model_main_tf2.py')
    command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps={}". \
        format(TRAINING_SCRIPT, TFODStruct.paths['TRAIN_OUTPUT_PATH'],
               TFODStruct.files['PIPELINE_CONFIG'], totalSteps)
    os.system(command)


printCheckPointVariables(os.path.join(TFODStruct.paths['CHECKPOINT_PATH'], 'latest-ckpt'))
trainSSDMobNet(batchSize=10, numSteps=1000, warmupSteps=500, learningRateBase=0.03, warmupLearningRate=0.007)
saveLastCheckPoint()