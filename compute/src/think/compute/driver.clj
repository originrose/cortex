(ns think.compute.driver
  "Base set of protocols required to move information from the host to the device as well as enable
  some form of computation on a given device.  There is a cpu implementation provided for reference.

  This file describes three basic datatypes:
  Driver - Enables enumeration of devices as well as creation of streams and host or device buffers.
  Stream - Stream of execution occuring on the device.
  Event - Synchronization primitive.  Events are created un-triggered and get triggered their associated
  stream gets to the point where the event was created."
  (:require [think.datatype.core :as dtype]
            [clojure.core.matrix :as m]
            [think.resource.core :as resource]))


(defprotocol PDriver
  "A driver is a generic compute abstraction.  Could be a group of threads,
  could be a machine on a network or it could be a CUDA or OpenCL driver.
  A stream is a stream of execution (analogous to a thread) where
  subsequent calls are serialized.  All buffers are implement a few of the datatype
  interfaces, at least get-datatype and ecount.  Host buffers are expected to implement
  enough of the datatype interfaces to allow a copy operation from generic datatypes
  into them.  This means at least PAccess."
  (get-devices [impl]
    "Get a list of devices accessible to the system.")
  (set-current-device [impl device]
    "Set the current device.  In for cuda this sets the current device in thread-specific storage so expecting
this to work in some thread independent way is a bad assumption.")
  (get-current-device [impl]
    "Get the current device from the current thread.")
  (create-stream [impl]
    "Create a stream of execution.  Streams are indepenent threads of execution.  They can be synchronized
with each other and the main thread using events.")
  (allocate-host-buffer [impl elem-count elem-type]
    "Allocate a host buffer.  Transfer from host to device requires data first copied into a host buffer
and then uploaded to a device buffer.")
  (allocate-device-buffer [impl elem-count elem-type]
    "Allocate a device buffer.  This is the generic unit of data storage used for computation.")
  (sub-buffer-impl [impl device-buffer offset length]
    "Create a sub buffer that shares the backing store with the main buffer.")
  (allocate-rand-buffer [impl elem-count]
    "Allocate a buffer used for rands.  Random number generation in general needs a divisible-by-2 element count
and a floating point buffer (cuda cuRand limitation)"))

(defprotocol PDriverProvider
  "Get a driver from an object"
  (get-driver [impl]))

(defprotocol PStream
  "Basic functionality expected of streams.  Streams are an abstraction of a stream of execution
and can be synchonized with the host or with each other using events."
  (copy-host->device [stream host-buffer host-offset
                      device-buffer device-offset elem-count])
  (copy-device->host [stream device-buffer device-offset
                      host-buffer host-offset elem-count])
  (copy-device->device [stream dev-a dev-a-off dev-b dev-b-off elem-count])
  (memset [stream device-buffer device-offset elem-val elem-count])
  (create-event [stream]
    "Create an event for synchronization.  The event is triggered when the stream
executes to the event.")
  (sync-event [stream event]
    "Have this stream pause until a given event is triggered."))

(defprotocol PStreamProvider
  "Get a stream from an object"
  (get-stream [impl]))

(defprotocol PEvent
  (wait-for-event [event]
    "Wait on the host until an event is triggered."))


(defn sub-buffer
  "Create a view of a buffer which shares the backing store
  with the original.  Offset must be >= 0."
  [impl device-buffer ^long offset ^long length]
  (let [original-size (dtype/ecount device-buffer)
        new-max-length (- original-size offset)]
    (when-not (<= length new-max-length)
      (throw (Exception. "Sub buffer out of range.")))
    (sub-buffer-impl impl device-buffer offset length)))


(defn host-array->device-buffer
  "Synchronously make a device buffer with these elements in it."
  [device stream upload-ary]
  (let [datatype (dtype/get-datatype upload-ary)
        elem-count (m/ecount upload-ary)
        upload-buffer (allocate-host-buffer device elem-count datatype)
        device-buffer (allocate-device-buffer device elem-count datatype)]
    (dtype/copy! upload-ary 0 upload-buffer 0 elem-count)
    (copy-host->device stream upload-buffer 0 device-buffer 0 elem-count)
    (wait-for-event (create-event stream))
    (resource/release upload-buffer)
    device-buffer))


(defn device-buffer->host-array
  "Synchronously transfer a device buffer to a host array"
  [device stream device-buffer]
  (let [elem-count (m/ecount device-buffer)
        datatype (dtype/get-datatype device-buffer)
        download-buffer (allocate-host-buffer device elem-count datatype)
        download-ary (dtype/make-array-of-type datatype elem-count)]
    (copy-device->host stream device-buffer 0 download-buffer 0 elem-count)
    (wait-for-event (create-event stream))
    (dtype/copy! download-buffer 0 download-ary 0 elem-count)
    (resource/release download-buffer)
    download-ary))
