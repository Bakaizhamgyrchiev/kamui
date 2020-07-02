#!/bin/bash
binary=$1
schema=$2
input_file=$3
output_folder=$4

$binary -t --strict-json --defaults-json -o $output_folder $schema -- $input_file

#./flatbuffer_to_json.sh /home/bakai/oz/projects/flatbuffers/flatc /home/bakai/oz/projects/tensorflow/tensorflow/lite/schema/schema.fbs /media/sensor_models_local/facemesh/face_landmark.tflite /media/sensor_models_local/facemesh/facemesh.json