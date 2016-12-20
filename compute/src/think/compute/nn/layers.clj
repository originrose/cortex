(ns think.compute.nn.layers
  "Base set of layers expected to work across all backends.  These layers implement the
cortex protocols around nn layers and provide some implementation of their respective types
in order to ease the implementation burden across backends and ensure as much of a unified
implementation as possible."
  (:require [think.compute.nn.backend :as nn-backend]
            [think.compute.math :as math]
            [think.compute.driver :as drv]
            [clojure.core.matrix :as m]
            [cortex.util :as util]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defprotocol ComputeLayer
  "Interface to connect the execution context to either a shared implementation
(with sharing in this file) or a backend-specific implementation."
  (forward [layer parameter-buffers input-buffers output-buffers])
  (backward [layer parameter-buffers input-buffers output-buffers]))


(defn create [backend node batch-size]
  ;;For layers that share no implementation between backends
  ;;We just create the backend representation.
  (nn-backend/create backend node batch-size))
