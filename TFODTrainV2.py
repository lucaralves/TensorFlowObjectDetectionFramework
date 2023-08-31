import TFODStruct
import os
import shutil
import tensorflow as tf
from object_detection import model_lib_v2
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

num_steps = 200
warmup_steps = 100
learning_rate_base = 0.03
warmup_learning_rate = 0.007
checkpoint_every_n = 100
checkpoint_max_to_keep = 10
batch_size = 4

checkpoint_path = os.path.join(TFODStruct.paths['CHECKPOINT_PATH'], 'latest-ckpt')
num_classes = len(TFODStruct.labels)
label_map_path = TFODStruct.files['LABELMAP']
train_record_path = [os.path.join(TFODStruct.paths['ANNOTATION_PATH'], 'train.record')]
test_record_path = [os.path.join(TFODStruct.paths['ANNOTATION_PATH'], 'test.record')]

pipeline_config_path = TFODStruct.files['PIPELINE_CONFIG']
model_dir = TFODStruct.paths['TRAIN_OUTPUT_PATH']
use_tpu = False
record_summaries = True

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
    print()


#
# FUNÇÃO QUE LÊ A QUANTIDADE DE STEPS DE UM CHECKPOINT.
#
def readCheckPointSteps(checkpointPath):

    reader = tf.train.load_checkpoint(checkpointPath)
    try:
        steps = reader.get_tensor("optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE")
    except:
        steps = 0

    return steps


#
# FUNÇÃO QUE LÊ O NÚMERO ASSOCIADO A UM CHECKPOINT.
#
def readCheckpointNumber(checkpointPath):

    reader = tf.train.load_checkpoint(checkpointPath)
    try:
        checkpointCounter = reader.get_tensor("save_counter/.ATTRIBUTES/VARIABLE_VALUE")
    except:
        checkpointCounter = 0

    return checkpointCounter


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
    checkpoint_files_path = [os.path.join(TFODStruct.paths["TRAIN_OUTPUT_PATH"], os.path.splitext(file)[0])
                        for file in all_files if file.endswith(".index")]

    if checkpoint_files_path:
        latest_checkpoint = max(checkpoint_files_path, key=readCheckpointNumber)

        source_checkpoint_index = latest_checkpoint + '.index'
        target_checkpoint_index = os.path.join(TFODStruct.paths["CHECKPOINT_PATH"],
                                         'latest-ckpt.index')
        source_checkpoint_data = latest_checkpoint + '.data-00000-of-00001'
        target_checkpoint_data = os.path.join(TFODStruct.paths["CHECKPOINT_PATH"],
                                          'latest-ckpt.data-00000-of-00001')

        shutil.copy(source_checkpoint_index, target_checkpoint_index)
        shutil.copy(source_checkpoint_data, target_checkpoint_data)


def configTrainPipeline(train_steps, warmup_steps, learning_rate_base, warmup_learning_rate,
                        checkpoint_path, num_classes, batch_size, label_map_path,
                        train_record_path, test_record_path):

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(TFODStruct.files['PIPELINE_CONFIG'], "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    pipeline_config.model.ssd.num_classes = num_classes
    pipeline_config.train_config.batch_size = batch_size

    pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate. \
        cosine_decay_learning_rate.total_steps = train_steps
    pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate. \
        cosine_decay_learning_rate.warmup_steps = warmup_steps
    pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate. \
        cosine_decay_learning_rate.learning_rate_base = learning_rate_base
    pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate. \
        cosine_decay_learning_rate.warmup_learning_rate = warmup_learning_rate

    pipeline_config.train_config.num_steps = train_steps
    pipeline_config.train_config.fine_tune_checkpoint = checkpoint_path
    pipeline_config.train_config.fine_tune_checkpoint_version = 'V2'
    pipeline_config.train_config.fine_tune_checkpoint_type = 'detection'
    pipeline_config.train_input_reader.label_map_path = label_map_path
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = train_record_path
    pipeline_config.eval_input_reader[0].label_map_path = label_map_path
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = test_record_path

    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(TFODStruct.files['PIPELINE_CONFIG'], "wb") as f:
        f.write(config_text)


def main(unused_argv):

    printCheckPointVariables(checkpointPath=checkpoint_path)

    total_steps = readCheckPointSteps(checkpointPath=checkpoint_path) + num_steps
    total_warmup_steps = readCheckPointSteps(checkpointPath=checkpoint_path) + warmup_steps
    configTrainPipeline(train_steps=total_steps, warmup_steps=total_warmup_steps,
                        learning_rate_base=learning_rate_base, warmup_learning_rate=warmup_learning_rate,
                        checkpoint_path=checkpoint_path, num_classes=num_classes, batch_size=batch_size,
                        label_map_path=label_map_path, train_record_path=train_record_path,
                        test_record_path=test_record_path)

    strategy = tf.compat.v2.distribute.MirroredStrategy()
    with strategy.scope():
      model_lib_v2.train_loop(
          pipeline_config_path=pipeline_config_path,
          model_dir=model_dir,
          train_steps=total_steps,
          use_tpu=use_tpu,
          checkpoint_every_n=checkpoint_every_n,
          record_summaries=record_summaries,
          checkpoint_max_to_keep=checkpoint_max_to_keep)

    saveLastCheckPoint()

if __name__ == '__main__':
    tf.compat.v1.app.run()