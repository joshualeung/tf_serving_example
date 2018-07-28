# coding: utf8
"""
Exporting a pretrained inception model for online serving.

Usage:
  python export_inception.py --export_path /tmp/inception --version 1
"""
import os.path
import tensorflow as tf

FLAGS = None
# Download the pretrained model from http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz and put it under FLAGS.model_dir
tf.app.flags.DEFINE_string('model_dir', '', 'Directory where you untar the pretrained model.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
tf.app.flags.DEFINE_string('export_path', '', 'Export path.')
tf.app.flags.DEFINE_integer('version', 1, 'Model version.')
FLAGS = tf.app.flags.FLAGS

def create_graph():
  model_path = os.path.join(FLAGS.model_dir, "classify_image_graph_def.pb")
  with tf.gfile.FastGFile(model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def export_model():
  create_graph()
  with tf.Session() as sess:
    input_tensor = sess.graph.get_tensor_by_name('DecodeJpeg/contents:0')
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    #predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

    export_path = os.path.join(FLAGS.export_path, FLAGS.version)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    prediction_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
        inputs = {'image': tf.saved_model.utils.build_tensor_info(input_tensor)},
        outputs = {'softmax': tf.saved_model.utils.build_tensor_info(softmax_tensor)},
        method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME
      )
    )

    legacy_init_op = tf.group(tf.tables_initializer(), name = "legacy_init_op")
    builder.add_meta_graph_and_variables(sess,
                                         [tf.saved_model.tag_constants.SERVING],
                                         signature_def_map={
                                           'predict_images': prediction_signature,
                                           tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature,
                                         },
                                         legacy_init_op=legacy_init_op)
    builder.save()
    print("Done exporting!")

if __name__ == '__main__':
  export_model()
