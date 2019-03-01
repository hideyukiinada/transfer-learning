# Note on label_image.py script (DRAFT???)

This is a follow up note on the transfer learning article.
As I mentioned in the article, once you are done with re-training, you use label_image.py to classify your image.

I thought it may be helpful for some readers as there are a few sections of the code that may not be straightforward.

Overall structure of the code is

1. Load the model
1. Read the image
1. Run the image by the model
1. Pick top 5 from the predicted result


# 1. Loading the model
During the retraining, a model is saved as a protobuf file.
In the process of saving, variables are also replaced with constants (??? mention the name of the function).  This makes sense as during prediction,
you don't update weights anymore so they don't have to be variables.

Reading the binary protobuf file is a four-step process.

1. Open the model file in the binary protobuf format
1. Read data into a buffer
1. Convert to GraphDef object from data
1. Convert to Graph object from GraphDef object


```
def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph
```

graph_def.ParseFromString(f.read()) does step 2 and step 3.
tf.import_graph_def(graph_def) does step 4.
tf.import_graph_def(graph_def) restores nodes to the default graph so in this code, they are setting the graph
object to default before calling tf.import_graph_def ???

Also nodes that are restored will be in the "import/" namespace, you will need to refer to.

graph_def.ParseFromString(f.read()) may look like this is a text operation, but it's not.
This is for binary.
ProtoBuf page also mentions this (??? add quote).

# 2. Reading the image
This part is straightforward.
The below code just reads the file (bmp, gif, png, jpg) and reads the file.

Input image dimension is set to 299 by 299 pixels for Inception-v3 model.

Here are the things that happen

Open and read the image file to the buffer???
Decode the data as png, gif, bmp or jpeg based on the file extention???
Cast the uint8 data to float32 to prepare for the range conversion later
Add one more dimension. 1024x800x3 becomes 1x1024x800x3
Resize image 1x299x299x3 using bilinear option. (check to see if the doc says anything aboub bilinear, if not
mention pillow or opencv's reference).
Normalize data. By default, shift the value from 0-255 to 0-1
Run the session to actually read the data.

```
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
```

# 3. Load label file

```

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

```

# 4. Running the image by the model

Once the model is loaded and the image file is read, you can run the image through the model.
In order to do that you need to :

know the name of the node in the graph.
get the reference to the node

```

  graph = load_graph(model_file)
  t = read_tensor_from_image_file(
      file_name,
      input_height=input_height,
      input_width=input_width,
      input_mean=input_mean,
      input_std=input_std)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)

  with tf.Session(graph=graph) as sess:
    results = sess.run(output_operation.outputs[0], {
        input_operation.outputs[0]: t
    })
  results = np.squeeze(results)
```

# 5. Picking top 5 from the predicted result









# License
This is the license for the code that appears in this article:

```
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
```