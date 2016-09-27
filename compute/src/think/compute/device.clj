(ns think.compute.device
  (:require [think.compute.datatype :as dtype]
            [clojure.core.matrix :as m]
            [resource.core :as resource]))


(defprotocol PDevice
  "A device is a generic compute abstraction.  Could be a group of threads,
  could be a machine on a network or it could be a CUDA or OpenCL device.
  A stream is a stream of execution (analogous to a thread) where
  subsequent calls are serialized."
  (get-devices [impl])
  (set-current-device [impl device])
  (get-current-device [impl])
  (create-stream [impl])
  (allocate-host-buffer [impl elem-count elem-type])
  (allocate-device-buffer [impl elem-count elem-type])
  ;;Create a sub buffer that shares backing store with this main buffer but can
  ;;represent a subsection.
  (sub-buffer-impl [impl buffer offset length])
  ;;The random number generators of different devices will usually be required to produce
  ;;random numbers of a particular type; for CUDA in general it is float values.
  (allocate-rand-buffer [impl elem-count]))

(defprotocol PDeviceProvider
  "Get a device from an object"
  (get-device [impl]))

(defprotocol PStream
  ;;Offsets are in elements.  Moving data onto a device is potentially
  ;;a multiple step process from
  ;;getting the data in a packed java array and then onto a device-specific host buffer and then
  ;;finally a host->device call.  Moving data back until you can manipulate it on the JVM is
  ;;potentially another multiple step process.  In all these steps the types must match.
  (copy-host->device [stream host-buffer host-offset
                      device-buffer device-offset elem-count])
  (copy-device->host [stream device-buffer device-offset
                      host-buffer host-offset elem-count])
  (copy-device->device [stream dev-a dev-a-off dev-b dev-b-off elem-count])
  ;;Set the value in a buffer to a constant value.
  (memset [stream device-buffer device-offset elem-val elem-count])
  ;;Create an event in a stream
  (create-event [stream])
  ;;Ensure this stream cannot proceed until this event is triggered.
  (sync-event [stream event]))

(defprotocol PStreamProvider
  "Get a stream from an object"
  (get-stream [impl]))

(defprotocol PEvent
  ;;Wait on the host till the event has been triggered
  (wait-for-event [event]))


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
