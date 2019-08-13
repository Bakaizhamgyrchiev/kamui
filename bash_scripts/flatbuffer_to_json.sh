#!/bin/bash
binary=$1
schema=$2
input_file=$3
output_folder=$4

$binary -t --strict-json --defaults-json -o $output_folder $schema -- $input_file
