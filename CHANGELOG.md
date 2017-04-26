# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [0.9.5] - 2017-04-26
### Added
- CPU-only support. Cortex can now run on the CPU without CUDA drivers being installed.
- docker-example -- A simple example of how to run a cortex project in a docker container.
- multi-thread -- The execution context now supports specifying the device, allowing for more advanced asynchronous computations like pipeline parallelism and using multiple devices.
### Bugs fixed
- Calling `execute/run` with a large dataset and a small batch size would throw stack-overflow exception. (https://github.com/thinktopic/think.resource/commit/0c435b2361cb5cf8f68410a9681625f7cfa7baff)
