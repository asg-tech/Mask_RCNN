
import os
import sys
#import json
#import datetime
#import numpy as np
#import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

import keras.models

import tensorflow as tf
from tensorflow.python.framework import graph_io

# Path to trained weights file
#COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

DEFAULT_SAVE_PB = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class ConvConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "ASG_Conv"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9



#def train(model):
#    """Train the model."""
#    # Training dataset.
#    dataset_train = BalloonDataset()
#    dataset_train.load_balloon(args.dataset, "train")
#    dataset_train.prepare()
#
#    # Validation dataset
#    dataset_val = BalloonDataset()
#    dataset_val.load_balloon(args.dataset, "val")
#    dataset_val.prepare()
#
#    # *** This training schedule is an example. Update to your needs ***
#    # Since we're using a very small dataset, and starting from
#    # COCO trained weights, we don't need to train too long. Also,
#    # no need to train all layers, just the heads should do it.
#    print("Training network heads")
#    model.train(dataset_train, dataset_val,
#                learning_rate=config.LEARNING_RATE,
#                epochs=30,
#                layers='heads')



def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name='frozen_model.pb', save_pb_as_text=False):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=False)
        if save_pb_as_text:
            graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name+"txt", as_text=True)
        return graphdef_frozen




############################################################
#  Main - Converter
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Convert H5 to PB')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--save', required=False,
                        default=DEFAULT_SAVE_PB,
                        metavar="/path/to/pb/",
                        help='Path to save pb too)')
    args = parser.parse_args()

    print("Weights: ", args.weights)
    print("Logs: ", args.logs)
    print("save: ", args.save)

    class InferenceConfig(ConvConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()

    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)

    weights_path = args.weights

    model.load_weights(args.weights, by_name=True)

    #model.keras_model.save('./mymodel.hdf5')

    session = tf.keras.backend.get_session()

    input_names = [t.op.name for t in model.keras_model.inputs]
    output_names = [t.op.name for t in model.keras_model.outputs]

    # Prints input and output nodes names, take notes of them.
    #print(input_names, output_names)

    print ("Input Names: {}".format(input_names))
    print ("Output Names: {}".format(output_names))

    text_file = open(args.save.rstrip()+"/IO_layers.txt", "w")
    text_file.write("Input Names: {}\n\n".format(input_names))
    text_file.write("Output Names: {}".format(output_names))
    text_file.close()

    frozen_graph = freeze_graph(session.graph, session, [out.op.name for out in model.keras_model.outputs], save_pb_dir=args.save, save_pb_as_text=True)
    #frozen_graph_txt = freeze_graphTxt(session.graph, session, [out.op.name for out in model.keras_model.outputs], save_pb_dir=args.save)

    trt_graph = trt.create_inference_graph(
        input_graph_def=frozen_graph,
        outputs=output_names,
        max_batch_size=2,
        max_workspace_size_bytes=1 << 25,
        precision_mode='FP16',
        minimum_segment_size=50)

    graph_io.write_graph(trt_graph, args.save.rstrip(), "trt_graph.pb", as_text=False)
    graph_io.write_graph(trt_graph, args.save.rstrip(), "trt_graph.pbtxt", as_text=True)
    #graph_io.write_graph(frozen_graph, "./model/", "test_frozen_graph.pb", as_text=False)
    #graph_io.write_graph(frozen_graph_txt, "./model/", "test_frozen_graph.pbtxt", as_text=True)