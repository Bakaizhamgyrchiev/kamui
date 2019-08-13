import tensorflow as tf
import numpy as np
import os
import json
import click
import cv2


CONV_2D = "CONV_2D"
RELU = "RELU"
DEPTHWISE_CONV_2D = "DEPTHWISE_CONV_2D"
ADD = "ADD"
PAD = "PAD"
MAX_POOL_2D = "MAX_POOL_2D"
RESHAPE = "RESHAPE"
CONCATENATION = "CONCATENATION"


def add_input_tensor(tensor_description, endpoints):
    name = tensor_description["name"]
    dtype = tf.float32
    shape = [None] + tensor_description["shape"][1:]

    click.echo(f"--> Adding input_tensor: {name} {dtype} {shape}")

    endpoints['0'] = tf.placeholder(name=name, dtype=dtype, shape=shape)

    return endpoints


def add_conv2d(op, tensors_description, endpoints, interpreter):
    input_ind, kernel_ind, bias_ind = op["inputs"]
    output_ind = op["outputs"][0]

    input = endpoints[str(input_ind)]

    kernel_desc = tensors_description[kernel_ind]
    bias_desc = tensors_description[bias_ind]
    layer_desc = tensors_description[output_ind]

    name = layer_desc["name"]
    ksize_w = kernel_desc["shape"][1]
    ksize_h = kernel_desc["shape"][2]
    filters = kernel_desc["shape"][0]
    stride_w = op["builtin_options"]["stride_w"]
    stride_h = op["builtin_options"]["stride_h"]
    padding = op["builtin_options"]["padding"]

    kernel_init = tf.keras.initializers.Constant(interpreter.get_tensor(kernel_ind))
    bias_init = tf.keras.initializers.Constant(interpreter.get_tensor(bias_ind))

    endpoints[str(output_ind)] = tf.keras.layers.Conv2D(
        kernel_size=[ksize_h, ksize_w],
        filters=filters,
        strides=[stride_h, stride_w],
        padding=padding,
        kernel_initializer=kernel_init,
        bias_initializer=bias_init,
        name=name
    )(input)

    return endpoints


def add_relu(op, tensors_description, endpoints, interpreter):
    input_ind = op["inputs"][0]
    output_ind = op["outputs"][0]

    input = endpoints[str(input_ind)]

    layer_desc = tensors_description[output_ind]

    name = layer_desc["name"]

    endpoints[str(output_ind)] = tf.keras.layers.ReLU(name=name)(input)

    return endpoints


def add_dephtwise_conv2d(op, tensors_description, endpoints, interpreter):
    input_ind, kernel_ind, bias_ind = op["inputs"]
    output_ind = op["outputs"][0]

    input = endpoints[str(input_ind)]

    kernel_desc = tensors_description[kernel_ind]
    bias_desc = tensors_description[bias_ind]
    layer_desc = tensors_description[output_ind]

    name = layer_desc["name"]
    ksize_w = kernel_desc["shape"][1]
    ksize_h = kernel_desc["shape"][2]
    stride_w = op["builtin_options"]["stride_w"]
    stride_h = op["builtin_options"]["stride_h"]
    padding = op["builtin_options"]["padding"]
    depth_multiplier = op["builtin_options"]["depth_multiplier"]

    kernel_init = tf.keras.initializers.Constant(interpreter.get_tensor(kernel_ind))
    bias_init = tf.keras.initializers.Constant(interpreter.get_tensor(bias_ind))

  #  filters = tf.Variable(tf.random_normal((5,5,100,10)))
  #  out = tf.nn.depthwise_conv2d(
  #      inputs,
  #      filters,
  #      strides=[1,1,1,1],
  #      padding='SAME')

    endpoints[str(output_ind)] = tf.keras.layers.DepthwiseConv2D(
        kernel_size=[ksize_h, ksize_w],
        strides=[stride_h, stride_w],
        depth_multiplier=depth_multiplier,
        padding=padding,
        kernel_initializer=kernel_init,
        bias_initializer=bias_init,
        name=name
    )(input)

    return endpoints


def add_add(op, tensors_description, endpoints, interpreter):
    input1_ind, input2_ind = op["inputs"]
    output_ind = op["outputs"][0]

    input1 = endpoints[str(input1_ind)]
    input2 = endpoints[str(input2_ind)]

    layer_desc = tensors_description[output_ind]

    name = layer_desc["name"]

    endpoints[str(output_ind)] = tf.add(x=input1, y=input2, name=name)

    return endpoints


def add_max_pool_2d(op, tensors_description, endpoints, interpreter):
    input_ind = op["inputs"][0]
    output_ind = op["outputs"][0]

    input = endpoints[str(input_ind)]

    layer_desc = tensors_description[output_ind]

    name = layer_desc["name"]
    stride_w = op["builtin_options"]["stride_w"]
    stride_h = op["builtin_options"]["stride_h"]
    filter_width = op["builtin_options"]["filter_width"]
    filter_height = op["builtin_options"]["filter_height"]
    padding = op["builtin_options"]["padding"]

    endpoints[str(output_ind)] = tf.keras.layers.MaxPool2D(
        pool_size=[filter_height, filter_width],
        padding=padding,
        strides=[stride_h, stride_w],
        name=name
    )(input)

    return endpoints


def add_pad(op, tensors_description, endpoints, interpreter):
    input_ind, padding_ind = op["inputs"]
    output_ind = op["outputs"][0]

    input = endpoints[str(input_ind)]

    layer_desc = tensors_description[output_ind]

    name = layer_desc["name"]
    padding = interpreter.get_tensor(padding_ind)

    endpoints[str(output_ind)] = tf.pad(input, padding, name=name)

    return endpoints


def add_reshape(op, tensors_description, endpoints, interpreter):
    input_ind = op["inputs"][0]
    output_ind = op["outputs"][0]

    input = endpoints[str(input_ind)]

    layer_desc = tensors_description[output_ind]

    name = layer_desc["name"]
    new_shape = op["builtin_options"]["new_shape"]

    endpoints[str(output_ind)] = tf.reshape(input, shape=new_shape, name=name)

    return endpoints


def add_concatenate(op, tensors_description, endpoints, interpreter):
    input1_ind, input2_ind = op["inputs"]
    output_ind = op["outputs"][0]

    input1 = endpoints[str(input1_ind)]
    input2 = endpoints[str(input2_ind)]

    layer_desc = tensors_description[output_ind]

    name = layer_desc["name"]
    axis = op["builtin_options"]["axis"]

    endpoints[str(output_ind)] = tf.concat([input1, input2], axis=axis, name=name)

    return endpoints


@click.command()
@click.option('--json_graph', default=None, type=str)
@click.option('--flatbuffer_graph', default=None, type=str)
def fmain(json_graph, flatbuffer_graph):

    click.echo(f"--> Reading deserialized flatbuffer(tflite) json file {json_graph}..")
    with open(json_graph, "r") as f:
        json_dict = json.load(f)

    interpreter = tf.contrib.lite.Interpreter(flatbuffer_graph)
    interpreter.allocate_tensors()

    operator_codes = json_dict['operator_codes']

    subgraph = json_dict['subgraphs'][0]

    inputs_indices = subgraph["inputs"][0]
    output_indices = subgraph["outputs"]
    tensors_description = subgraph["tensors"]
    operators = subgraph["operators"]

    graph = tf.Graph()

    endpoints = {}

    fns = {
        CONV_2D: add_conv2d,
        RELU: add_relu,
        DEPTHWISE_CONV_2D: add_dephtwise_conv2d,
        ADD: add_add,
        PAD: add_pad,
        RESHAPE: add_reshape,
        CONCATENATION: add_concatenate,
        MAX_POOL_2D: add_max_pool_2d
    }

    click.echo(f"--> Creating graph")
    with graph.as_default():
        endpoints = add_input_tensor(tensors_description[inputs_indices], endpoints)

        for op in operators:
            print(op)
            opcode = operator_codes[op["opcode_index"]]["builtin_code"]
            endpoints = fns[opcode](op, tensors_description, endpoints, interpreter)

    input_data = cv2.imread("/home/bakai/oz/tmp/src/test.png")
    input_data = cv2.resize(src=input_data,dsize=(128, 128))
    input_data = np.expand_dims(input_data, 0)
    input_data = np.float32(input_data)

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        cls, reg = sess.run(["classificators:0", "regressors:0"], feed_dict={"input:0": input_data})

        print(cls[0][0], reg[0][0])
        #print(tf.trainable_variables())
        #print(sess.run("conv2d/bias:0"))

if __name__ == '__main__':
    fmain()
