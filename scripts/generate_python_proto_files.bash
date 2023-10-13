#!/usr/bin/env bash


python -m grpc_tools.protoc -I./protos --python_out=./product_prediction/prediction_service --pyi_out=./product_prediction/prediction_service --grpc_python_out=./product_prediction/prediction_service ./protos/shoe_category_prediction.proto