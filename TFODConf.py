import os
import wget
import TFODStruct

#
# CRIAÇÃO DAS DIRETORIAS QUE CONSTITUEM A FERRAMENTA TFDO.
#
for path in TFODStruct.paths.values():
    if not os.path.exists(path):
        if os.name == 'posix':
            os.system("mkdir -p {}".format(path))
        if os.name == 'nt':
            os.system("mkdir {}".format(path))

#
# CRIAÇÃO DAS DIRETORIAS ONDE SÃO GUARDADAS AS IMAGENS DE TREINO E TESTE.
#
if not os.path.exists(TFODStruct.paths['IMAGES_PATH']):
    if os.name == 'posix':
        os.system("mkdir -p {}".format(TFODStruct.paths['IMAGES_PATH']))
    if os.name == 'nt':
        os.system("mkdir {}".format(TFODStruct.paths['IMAGES_PATH']))
for label in TFODStruct.labels:
    path = os.path.join(TFODStruct.paths['IMAGES_PATH'], label['name'])
    if not os.path.exists(path):
        os.system("mkdir {}".format(path))

#
# CRIAÇÃO DE UM LABELMAP.
#
with open(TFODStruct.files['LABELMAP'], 'w') as f:
    for label in TFODStruct.labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

#
# DOWNLOAD DO SCRIPT QUE GERA OS TFRECORDS.
#
if not os.path.exists(TFODStruct.files['TF_RECORD_SCRIPT']):
    os.system("git clone https://github.com/nicknochnack/GenerateTFRecord {}".format(TFODStruct.paths['SCRIPTS_PATH']))

#
# DOWNLOAD DO MODELO QUE VAI SER UTILIZADO.
#
if os.name =='posix':
    os.system("wget {}".format(TFODStruct.PRETRAINED_MODEL_URL))
    os.system("mv {}.tar.gz {}".format(TFODStruct.PRETRAINED_MODEL_NAME, TFODStruct.paths['PRETRAINED_MODEL_PATH']))
    os.system("cd {} && tar -zxvf {}.tar.gz".format(TFODStruct.paths['PRETRAINED_MODEL_PATH'], TFODStruct.PRETRAINED_MODEL_NAME))
if os.name == 'nt':
    wget.download(TFODStruct.PRETRAINED_MODEL_URL)
    os.system("move {}.tar.gz {}".format(TFODStruct.PRETRAINED_MODEL_NAME, TFODStruct.paths['PRETRAINED_MODEL_PATH']))
    os.system("cd {} && tar -zxvf {}.tar.gz".format(TFODStruct.paths['PRETRAINED_MODEL_PATH'], TFODStruct.PRETRAINED_MODEL_NAME))

#
# COPIA-SE O MODELO PRE TREINADO PARA UMA OUTRA DIRETORIA.
#
if os.name =='posix':
    os.system("cp {} {}".format(os.path.join(TFODStruct.paths['PRETRAINED_MODEL_PATH'], TFODStruct.PRETRAINED_MODEL_NAME, 'pipeline.config'),
                                os.path.join(TFODStruct.paths['CHECKPOINT_PATH'])))
if os.name == 'nt':
    os.system("copy {} {}".format(os.path.join(TFODStruct.paths['PRETRAINED_MODEL_PATH'], TFODStruct.PRETRAINED_MODEL_NAME, 'pipeline.config'),
                                  os.path.join(TFODStruct.paths['CHECKPOINT_PATH'])))

#
# DOWNLOAD E INSTALAÇÃO DA API TFDO.
#
if not os.path.exists(os.path.join(TFODStruct.paths['APIMODEL_PATH'], 'research', 'object_detection')):
    os.system("git clone https://github.com/tensorflow/models {}".format(TFODStruct.paths['APIMODEL_PATH']))
url="https://github.com/protocolbuffers/protobuf/releases/download/v3.19.3/protoc-3.19.3-win64.zip"
wget.download(url)
os.system("move protoc-3.19.3-win64.zip {}".format(TFODStruct.paths['PROTOC_PATH']))
os.system("cd {} && tar -xf protoc-3.19.3-win64.zip".format(TFODStruct.paths['PROTOC_PATH']))
os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(TFODStruct.paths['PROTOC_PATH'], 'bin'))
os.system("cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && copy object_detection\\packages\\tf2\\"
          "setup.py setup.py && python setup.py build && python setup.py install")
os.system("cd Tensorflow/models/research/slim && pip install -e .")

#
# DOWNLOAD E INSTALAÇÃO DE ALGUMAS DEPENDÊNCIAS.
#
os.system("pip install tensorflow")
os.system("pip install scipy")
os.system("pip install matplotlib")
os.system("pip install PyYAML")
os.system("pip install pytz")
os.system("pip install pycocotools")
os.system("pip install opencv-python")
os.system("pip install gin-config==0.1.1")

#
# SCRIPT QUE VERIFICA SE A INSTALAÇÃO DA API TFDO FOI BEM SUCEDIDA.
#
VERIFICATION_SCRIPT = os.path.join(TFODStruct.paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')
os.system("python {}".format(VERIFICATION_SCRIPT))