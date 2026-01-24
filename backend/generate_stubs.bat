@echo off
REM Generate Python gRPC stubs from proto files
REM Run from g20_demo/backend/

echo Generating Python gRPC stubs...

if not exist generated mkdir generated

python -m grpc_tools.protoc ^
    -I../protos ^
    --python_out=./generated ^
    --grpc_python_out=./generated ^
    ../protos/control.proto ^
    ../protos/inference.proto

echo Done! Stubs in: backend/generated/
