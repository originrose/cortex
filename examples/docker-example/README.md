# docker-example

An example cortex project that demonstrates how to run cortex from a docker container.

## Overview

Frequently infrastructure uses Docker containers to deploy applications and
services. As there are some extra requirements to get projects working using
the GPU inside of a Docker container with the requried dependencies, this
project demonstrates how to do this with a very simple network.

## Preparing your project

The first step is to build and train a network. Nominally, this will result
in a saved network that can either be stored in the container (e.g. with the
ADD command in the Dockerfile) or perhaps through other means such as downloading
the latest network from a CDN. For the purposes of this example, the network
is trained on the fly.

Once the network has been trained, you will create a `-main` function and
specify that this is the function to run in your (project.clj)[project.clj].
Additionally, it is convenient to specify the jar name with the `:uberjar-name`
key. In this example we have chosen `docker-example.jar`. Ensure that your
network runs locally with `lein run`.

## Building the container

Next, you will create a Docker container that is based off of a container
provided by ThinkTopic that has all of the required Cuda/CudNN and necessary
libraries to run your container. Simply add the appropriate `.jar` file
and specify how to run it.

A very small example (Dockerfile)[Dockerfile] can be found in this project.

Next, build your project with the `docker build`  command (Remember to change
the tag to something appropriate for your project).

```
$ docker build -t docker-example .
```

## Running the container

Next we will use `nvidia-docker` to run your container in a manner that allows
it to access the GPU. Follow the install instructions here for installing
`nvidia-docker`: https://github.com/NVIDIA/nvidia-docker . Note that in addition
to installing nvidia-docker, you will also need to run the plugin. For now we can
simply run that in the background.

```
$ sudo nvidia-docker-plugin &
$ sudo nvidia-docker run docker-example
```

This should run your program in a Docker container while still leveraging the
underlying GPU.


## License

Copyright © 2017 ThinkTopic LLC
