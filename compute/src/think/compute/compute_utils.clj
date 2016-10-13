(ns think.compute.compute-utils
  (:require [clojure.core.async :as async]
            [think.resource.core :as resource]))


(defn- async-channel-to-lazy-seq
  [to-chan]
  (when-let [item (async/<!! to-chan)]
    (when-let [except (:thread-exception item)]
      (throw (Exception. "Exception killed compute thread" ^Throwable except)))
    (cons item (lazy-seq (async-channel-to-lazy-seq to-chan)))))


(defn compute-map
  "Similar to map except runs the compute operation in its own thread.  This is necessary
to protect the GPU resources against random threading issues (the CUDA libraries store
information in thread-dependent areas).  Process-fn takes 1 extra argument (first argument)
which is the result of init-fn.  Execution is protected by a resource context so all resources
allocated by init-fn or process-fn are released when the sequences are all terminated.
If the thread exits with an exception then that exception is thrown on when it is read
into the return sequence."
  [init-fn process-fn & args]
  (let [ret-chan (async/chan 5)]
    (async/thread
      (try
       (resource/with-resource-context
         (let [init-data (init-fn)]
           (doall (apply map (fn [& arg-data]
                               (async/>!! ret-chan
                                          (apply process-fn init-data arg-data)))
                         args))))
       (catch Throwable e
         (async/>!! ret-chan {:thread-exception e})))
      (async/close! ret-chan))
    (async-channel-to-lazy-seq ret-chan)))
