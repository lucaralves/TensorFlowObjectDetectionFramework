import os
import TFODStruct

#
# GERAM-SE OS TFRECORDS DAS IMAGENS DE TREINO E TESTE.
#
os.system("python {} -x {} -l {} -o {}".format(TFODStruct.files['TF_RECORD_SCRIPT'],
                                               os.path.join(TFODStruct.paths['IMAGE_PATH'], 'train'),
                                               TFODStruct.files['LABELMAP'],
                                               os.path.join(TFODStruct.paths['ANNOTATION_PATH'], 'train.record')))
os.system("python {} -x {} -l {} -o {}".format(TFODStruct.files['TF_RECORD_SCRIPT'],
                                               os.path.join(TFODStruct.paths['IMAGE_PATH'], 'test'),
                                               TFODStruct.files['LABELMAP'],
                                               os.path.join(TFODStruct.paths['ANNOTATION_PATH'], 'test.record')))