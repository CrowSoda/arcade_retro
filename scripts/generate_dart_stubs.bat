@echo off
REM Generate Dart gRPC stubs from proto files
REM Run from g20_demo/

echo Generating Dart gRPC stubs...

REM Ensure Flutter protoc_plugin is installed
dart pub global activate protoc_plugin

REM Create output directory
if not exist lib\core\grpc\generated mkdir lib\core\grpc\generated

REM Generate Dart stubs
protoc --proto_path=protos ^
    --dart_out=grpc:lib/core/grpc/generated ^
    protos/control.proto ^
    protos/inference.proto

echo Done! Stubs in: lib/core/grpc/generated/
echo Add to pubspec.yaml:
echo   grpc: ^4.0.0
echo   protobuf: ^3.0.0
