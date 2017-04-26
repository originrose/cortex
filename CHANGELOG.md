# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [0.9.5] - 2017-04-26
### Added
- CPU-only support. Cortex can now run on the CPU without CUDA drivers being installed.
- docker-example -- A simple example of how to run a cortex project in a docker container.
- multi-thread -- The execution context now supports specifying the device, allowing for more
advanced asynchronous computations like pipeline parallelism and using multiple devices.
