import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder

# Caminho para o diretório onde o checkpoint está armazenado
checkpoint_directory = r'C:\Users\TECRA\Desktop\Uni\3ano\ESTAGIO\TensorFlowObjectDetectionToolV2\Tensorflow\workspace\models\my_ssd_mobnet\checkpoints'
checkpoint_path = os.path.join(checkpoint_directory, 'ckpt-1')

# Carregar as configurações do modelo
configs = config_util.get_configs_from_pipeline_file('C:\\Users\\TECRA\\Desktop\\Uni\\3ano\\ESTAGIO\\TensorFlowObjectDetectionToolV2\\Tensorflow\\' +
            'workspace\\models\\my_ssd_mobnet\\pipeline.config')
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Carregar um checkpoint
checkpoint = tf.compat.v2.train.Checkpoint(model=detection_model)
checkpoint.restore(checkpoint_path).expect_partial()

# Carregar as variáveis do checkpoint usando tf.train.load_checkpoint
reader = tf.train.load_checkpoint(checkpoint_path)
var_names = reader.get_variable_to_shape_map().keys()

# optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
try:
    value = reader.get_tensor("optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE")
    print(value)
except:
    value = 0
    print(value)

# Imprimir os valores das variáveis
# for var_name in var_names:
    # var_value = reader.get_tensor(var_name)
    # print(f"Variable name: {var_name}")
    # print("Variable value:\n", var_value)
    # print("=" * 50)