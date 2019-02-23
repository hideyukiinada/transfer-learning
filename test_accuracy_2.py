#
# This file is based on label_image.py
# Copyright 2019, Hide Inada for the changes from and addition to the original code.
#
# --- Original copyright notice ---
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import sys
import re
from pathlib import Path
import numpy as np
import tensorflow as tf

RESULT_FILE = "accuracy_result.txt"

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))  # Change the 2nd arg to INFO to suppress debug logging


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def predict(sess=None, graph=None, input_operation=None, output_operation=None, label_file_name=None,
            image_file_name=None):
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255

    t = read_tensor_from_image_file(
        image_file_name,
        input_height=input_height,
        input_width=input_width,
        input_mean=input_mean,
        input_std=input_std)

    results = sess.run(output_operation.outputs[0], {
        input_operation.outputs[0]: t
    })
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file_name)

    top_labels = list()
    for i in top_k:
        top_labels.append(labels[i])

    top_labels = list(map(lambda e: e.replace(" ", ""), top_labels))

    # print(top_labels)
    return top_labels


def main():
    if len(sys.argv) != 2:
        print("Usage: test.accuracy.py <image base directory name>")
        sys.exit(1)

    image_base_dir = sys.argv[1]
    image_base_dir_path = Path(image_base_dir)
    if image_base_dir_path.exists() is False:
        raise ValueError("Image base directory is not found.")

    model_file_name = "/tmp/output_graph.pb"
    label_file_name = "/tmp/output_labels.txt"
    input_layer = "Placeholder"
    output_layer = "final_result"

    graph = load_graph(model_file_name)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.Session(graph=graph) as sess:

        with open(RESULT_FILE, "w") as f_out:

            score = dict()
            for i, d in enumerate([d for d in image_base_dir_path.iterdir() if d.is_dir()]):

                # if i > 0:
                #     break
                #
                # log.info("Processing " + str(d))

                normalized_directory_name = d.name.lower().replace("_", "")
                print("Dir: " + normalized_directory_name)
                score[normalized_directory_name] = dict()

                score[normalized_directory_name]["total"] = 0
                score[normalized_directory_name]["top5"] = 0
                score[normalized_directory_name]["top1"] = 0

                for f in d.glob("*.jpg"):
                    score[normalized_directory_name]["total"] += 1

                    log.info("Processing " + str(f))

                    top5 = predict(sess=sess,
                                   graph=graph,
                                   input_operation=input_operation,
                                   output_operation=output_operation,
                                   label_file_name=label_file_name,
                                   image_file_name=str(f))

                    if normalized_directory_name in top5:
                        score[normalized_directory_name]["top5"] += 1

                    if normalized_directory_name == top5[0]:
                        score[normalized_directory_name]["top1"] += 1

                score[normalized_directory_name]["top5_accuracy"] = score[normalized_directory_name]["top5"] / \
                                                                    score[normalized_directory_name]["total"]
                score[normalized_directory_name]["top1_accuracy"] = score[normalized_directory_name]["top1"] / \
                                                                    score[normalized_directory_name]["total"]

                f_out.write("%s\t%0.4f\t%0.4f" % (
                    normalized_directory_name, score[normalized_directory_name]["top1_accuracy"],
                    score[normalized_directory_name]["top5_accuracy"]))


if __name__ == "__main__":
    main()
