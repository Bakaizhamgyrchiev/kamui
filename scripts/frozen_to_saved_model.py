import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat
from tensorflow.tools.graph_transforms import TransformGraph
import os
import copy
from typing import List, Dict

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


class ImporterGraph:

    @staticmethod
    def load_frozen_graph(frozen_graph_filename: str, output_tensors: List[str]):
        graph_def = ImporterGraph.get_graph_def_from_frozen_graph(frozen_graph_filename)

        [out] = tf.import_graph_def(graph_def, name="", return_elements=["{}:0".format(output_tensors[0])])

        return out.graph

    @staticmethod
    def add_graph_as_operation(pb_path: str, scope_name: str, input_map: Dict[str, tf.Tensor], output_names: List[str]) -> List[tf.Tensor]:
        """Позволяет встроить граф из экспортированной модели в нынешнюю модель.

            :param pb_path: полный путь к protobuf файлу, например, path/to/name.pb
            :param scope_name: название scopе, под которым будут подгружены операции из графа
            :param input_map: словарь, в котором ключом является имя входного тензора графа, а значением необходимый тензор из исходого графа
            :param output_names: список имен выходных тензоров графа
            :type pb_path: str
            :type scope_name: str
            :type input_map: Dict[str, tf.Tensor]
            :type output_names: List[str]
            :return: Список тензоров, соответствующих параметру output_names
            :rtype: List[tf.Tensor]
        """
        graph = tf.get_default_graph()

        with graph.as_default():
            try:
                graph_def = ImporterGraph.get_graph_def_from_frozen_graph(pb_path)
            except:
                try:
                    graph_def = ImporterGraph.get_graph_def_from_saved_model(pb_path)
                except:
                    raise NotImplementedError("--> Failed to import graph from protobuf!"
                                              "Exported protobuf type is not frozen or saved model")

            transforms = ['strip_unused_nodes','remove_attribute(attribute_name=_class)']
            graph_def = TransformGraph(graph_def, list(input_map.keys()), output_names, transforms)

            new_graph_def = tf.GraphDef()
            for node in graph_def.node:
                new_graph_def.node.extend([copy.deepcopy(node)])

        return tf.import_graph_def(new_graph_def, name=scope_name,
                                   input_map={f'{k}:0': v for k, v in input_map.items()} if input_map else None,
                                   return_elements=[f"{x}:0" for x in output_names] if output_names else None)

    @staticmethod
    def get_graph_def_from_frozen_graph(graph_pb_path: str):
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_pb_path, "rb") as f:
            graph_def.ParseFromString(f.read())

        return graph_def

    @staticmethod
    def get_graph_def_from_saved_model(graph_pb_path: str):
        with gfile.FastGFile(graph_pb_path, 'rb') as f:
            data = compat.as_bytes(f.read())
            sm = saved_model_pb2.SavedModel()
            sm.ParseFromString(data)

        return sm.meta_graphs[0].graph_def