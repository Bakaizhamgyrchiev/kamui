import tensorflow as tf

tf.enable_resource_variables()
import os
import json
import click
from typing import List

SESSION_CONFIG = tf.ConfigProto(allow_soft_placement=True, device_count={"GPU": 0})


class ConverterModes:
    SAVED_MODEL = 0
    LITE = 1


class Converter:
    lite_filename = "model.tflite"

    def __init__(self, graph: tf.Graph, input_tensors: List[str], output_tensors: List[str], target: str, convert_mode: int):
        self.graph = graph
        self.input_tensors = input_tensors
        self.output_tensors = output_tensors
        self.target = target
        self.convert_mode = convert_mode

    @classmethod
    def for_saved_model(cls, graph: tf.Graph, input_tensors: List[str], output_tensors: List[str], target: str):
        return cls(
            graph,
            input_tensors,
            output_tensors,
            target,
            ConverterModes.SAVED_MODEL
        )

    @classmethod
    def for_lite(cls, graph: tf.Graph, input_tensors: List[str], output_tensors: List[str], target: str):
        return cls(
            graph,
            input_tensors,
            output_tensors,
            target,
            ConverterModes.LITE
        )

    def convert(self):
        if self.convert_mode == ConverterModes.SAVED_MODEL:
            self.convert_saved_model()
        elif self.convert_mode == ConverterModes.LITE:
            self.convert_lite()
        else:
            pass

    def convert_saved_model(self):

        builder = tf.saved_model.builder.SavedModelBuilder(self.target)

        with tf.Session(config=SESSION_CONFIG, graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            input_signatures, output_signatures = self._get_input_output_signatures()

            signature = tf.saved_model.signature_def_utils.predict_signature_def(input_signatures,
                                                                                 output_signatures)

            builder.add_meta_graph_and_variables(
                sess,
                [tf.saved_model.tag_constants.SERVING],
                {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature},
                clear_devices=True
            )

        builder.save()

        return None

    def convert_lite(self):
        with tf.Session(config=SESSION_CONFIG, graph=self.graph) as sess:
            input_signatures, output_signatures = self._get_input_output_signatures()

            converter = tf.contrib.lite.TFLiteConverter.from_session(
                sess,
                list(input_signatures.values()),
                list(output_signatures.values())
            )
            converter.allow_custom_ops = True

            tflite_model = converter.convert()
            with open(os.path.join(self.target, self.lite_filename), "wb") as f:
                f.write(tflite_model)

        return None

    def _get_input_output_signatures(self):
        input_signatures = {input: self.graph.get_tensor_by_name("{}:0".format(input)) for input in self.input_tensors}
        output_signatures = {output: self.graph.get_tensor_by_name("{}:0".format(output)) for output in self.output_tensors}

        return input_signatures, output_signatures


CONV_2D = "CONV_2D"
RELU = "RELU"
PRELU = "PRELU"
DEPTHWISE_CONV_2D = "DEPTHWISE_CONV_2D"
ADD = "ADD"
PAD = "PAD"
MAX_POOL_2D = "MAX_POOL_2D"
RESHAPE = "RESHAPE"
CONCATENATION = "CONCATENATION"

BATCH = 1


def add_input_tensor(tensor_description, endpoints):
    name = tensor_description["name"]
    dtype = tf.float32
    shape = [BATCH] + tensor_description["shape"][1:]

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

    kernels = tf.constant(interpreter.get_tensor(kernel_ind))
    kernels = tf.transpose(kernels, [1, 2, 3, 0])
    kernels = tf.get_variable(name + "_kernel", initializer=kernels)

    bias = tf.constant(interpreter.get_tensor(bias_ind))
    bias = tf.get_variable(name + "_bias", initializer=bias)

    out = tf.nn.conv2d(
        input,
        kernels,
        strides=[1, stride_h, stride_w, 1],
        padding=padding,
        name=name)

    endpoints[str(output_ind)] = tf.nn.bias_add(out, bias)

    return endpoints


def add_relu(op, tensors_description, endpoints, interpreter):
    input_ind = op["inputs"][0]
    output_ind = op["outputs"][0]

    input = endpoints[str(input_ind)]

    layer_desc = tensors_description[output_ind]

    name = layer_desc["name"]

    endpoints[str(output_ind)] = tf.keras.layers.ReLU(name=name)(input)

    return endpoints

def add_prelu(op, tensors_description, endpoints, interpreter):
    input_ind, alpha_ind = op["inputs"]
    output_ind = op["outputs"][0]

    input = endpoints[str(input_ind)]

    layer_desc = tensors_description[output_ind]

    name = layer_desc["name"]

    alpha = tf.get_variable(name+"_prelu", tf.constant(interpreter.get_tensor(alpha_ind)))

    pos = tf.keras.layers.ReLU()(input)
    neg = -alpha * tf.keras.layers.ReLU()(-input)
    res = pos + neg

    endpoints[str(output_ind)] = res

    return endpoints


    return pos + neg


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

    kernels = tf.constant(interpreter.get_tensor(kernel_ind))
    kernels = tf.transpose(kernels, [1, 2, 3, 0])
    kernels = tf.get_variable(name+"_kernel", initializer=kernels)


    bias = tf.constant(interpreter.get_tensor(bias_ind))
    bias = tf.get_variable(name+"_bias", initializer=bias)

    out = tf.nn.depthwise_conv2d(
        input,
        kernels,
        strides=[1, stride_h, stride_w, 1],
        padding=padding,
        name=name)

    endpoints[str(output_ind)] = tf.nn.bias_add(out, bias)

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
    in_shape = tf.shape(input)
    new_shape = op["builtin_options"]["new_shape"]

    batch = BATCH
    num_channels = new_shape[2]
    num_box = in_shape[1] * in_shape[2] * tf.cast(in_shape[3] / new_shape[2], tf.int32)

    output_shape = tf.stack([batch, num_box, num_channels], axis=0)

    endpoints[str(output_ind)] = tf.reshape(input, shape=output_shape, name=name)

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
@click.option('--target_dir', default=None, type=str)
@click.option('--frozen_graph_name', default=None, type=str)
def fmain(json_graph, flatbuffer_graph, target_dir, frozen_graph_name):

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
        PRELU: add_prelu,
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
            opcode = operator_codes[op["opcode_index"]]["builtin_code"]
            endpoints = fns[opcode](op, tensors_description, endpoints, interpreter)

    click.echo(f"--> Saving graph to protocol buffer {os.path.join(target_dir, frozen_graph_name)}..")
    os.makedirs(target_dir, exist_ok=True)
    converter = Converter.for_saved_model(graph=graph, target=os.path.join(target_dir, "saved_model_var"),
                                          input_tensors=["input"], output_tensors=["classificators", "regressors"])
    converter.convert()

if __name__ == '__main__':
    fmain()
