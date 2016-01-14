In order to upgrade the protobuf support you will need an upgraded version of google's protoc compiler.

The version of the protobuf library in the project.clj file *must* match the version of the protoc compiler (protoc --version).

then simply:

```protoc --java_out=java resources/caffe.proto```