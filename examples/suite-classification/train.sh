#!/bin/bash

lein uberjar

java -jar target/suite-classification-0.1.0-SNAPSHOT-standalone.jar
