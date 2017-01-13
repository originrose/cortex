(ns cortex.loss
  "Definitions and implementations of cortex loss function terms.  The loss function
for a network is a weighted summation across a vector of terms.  The weight on any term is :lambda
and is defined with a per-term-default in the loss metadata for that function type.  The default
is 1."
  (:require [clojure.core.matrix :as m]))


;;Utilities for dealing with map constructors

(defn arg-list->arg-map
  [args]
  (when-not (= 0 (rem (count args) 2))
    (throw (ex-info "Argument count must be evenly divisble by 2"
                    {:arguments args})))
  (->> (partition 2 args)
       (map vec)
       (into {})))


(defn merge-args
  [desc args]
  (merge desc (arg-list->arg-map args)))


(defmulti loss-metadata
  "Get the metadata for a particular loss function.

There are some standard argument definitions:
:output - loss-fn contains? :node-id output of the node.
:stream - loss-fn contains? :stream the data coming from the data stream.
:parameter - loss-fn contains? :node-id :parameter and this is the value of the parameter of the node.

In addition loss functions themselves can have parameters in that the network stores in its
definition in parameter buffers.  This is useful in the case where the loss itself has some running
means associated with it like in the form of center loss.

All losses have a default lambda which weights them against any other elements in the loss.

In general, the loss function for a given network

The map must contain:
{:arguments - The keys and potentially some extra information about the data in the keys passed
to the loss function.
}"
  :type)


(defmethod loss-metadata :default
  [loss-fn]
  {:arguments #{:output :stream}
   :lambda 1.0})


(defn mse-loss
  "Mean squared error loss.  Applied to a node and a matching-size output stream."
  [& args]
  (merge-args
   {:type :mse-loss}
   args))


(defn softmax-loss
  "Softmax loss.  Applied to a node and a softmax (1-hot encoded) output stream."
  [& args]
  (merge-args
   {:type :softmax-loss}
   args))


(defn l1-regularization
  "Penalize the network for the sum of the absolute values of a given buffer.  This pushes all entries of the buffer
at a constant rate towards zero and is purported to lead to more sparse representations.  This could be applied
to either a trainable parameter or to a node in which case it will be applied to the node's output buffer."
  [& args]
  (merge-args
   {:type :l1-regularization}
   args))


(defn- reg-loss-metadata
  "Regularizer-type loss functions can be applied to either a node in which case there
will be no parameter entry in the loss function and the output of the node is assumed
or to a parameter buffer (like weights) in which case the function should have a parameter
entry in addition to a node-id."
  [loss-fn]
  (let [arguments (if (contains? loss-fn :parameter)
                    #{{:parameter (get loss-fn :parameter)}}
                    #{:output})]
    {:arguments arguments
     :lambda 0.001}))


(defmethod loss-metadata :l1-regularization
  [loss-fn]
  (reg-loss-metadata loss-fn))


(defn l2-regularization
  "Penalize the network for the magnitude of a given buffer.  This will penalize large entries in the buffer
  exponentially more than smaller entries leading to a buffer that tends to produce an even distribution of small
  entries.  Can be applied to either a trainable parameter or a node in which case it will be applied to
  the node's output buffer."
  [& args]
  (merge-args
   {:type :l2-regularization}
   args))

(defmethod loss-metadata :l2-regularization
  [loss-fn]
  (reg-loss-metadata loss-fn))


(defn center-loss
  "Center loss is a way of specializing an activation for use as a grouping/sorting
mechanism.  It groups activations by class and develops centers for the activations
over the course of training.  Alpha is a number between 1,0 that stands for the exponential
decay factor of the running centers (essentially running means).  The network is penalized
for the distance of the current activations from their respective centers.  The result is that
the activation itself becomes very grouped by class and thus make far better candidates for
LSH or a distance/sorting system.  Note that this is a loss used in the middle of the graph,
not at the edges.  This is applied to a given node and needs an softmax-type output stream.
http://ydwen.github.io/papers/WenECCV16.pdf"
  [{:keys [alpha] :as arg-map
    :or {alpha 0.5}}]
  (merge {:type :center-loss
          :alpha alpha}
         arg-map))


(defn- get-center-loss-center-buffer-shape
  "Get the shape of the centers of the network.  The network must be built and enough
information must be known about the dataset to make a stream->size map."
  [loss-fn node-id->node-map stream->size-map]
  (let [node-output-size (get-in node-id->node-map [(get loss-fn :node-id) :output-size])
        stream-size (get stream->size-map (get loss-fn :stream))]
    (when-not (and node-output-size
                   stream-size)
      (throw (ex-info "Center loss failed to find either node output size or stream size"
                      {:loss-function loss-fn
                       :output-size node-output-size
                       :stream-size stream-size})))
    ;;We keep track of stream-size centers each of node output size.
    [(long stream-size) (long node-output-size)]))


(defmethod loss-metadata :center-loss
  [loss-fn]
  {:parameters {:centers {:shape-fn get-center-loss-center-buffer-shape
                          :initialization {:type :constant
                                           :value 0}}}
   :arguments #{:output :stream}
   :lambda 0.1})


(defmulti loss
  "Implement a specific loss based on the type of the loss function.  They are passed map
containing the buffer coming from the network.
{:output output
 :target stream}"
  (fn [loss-fn buffer-map]
    (:type loss-fn)))


(defmethod loss :mse-loss
  [loss-fn buffer-map]
  (let [v (get buffer-map :output)
        target (get buffer-map :stream)]
   (/ (double (m/magnitude-squared (m/sub v target)))
      (m/ecount v))))


(defn log-likelihood-softmax-loss
  ^double [softmax-output answer]
  (let [answer-num (m/esum (m/mul softmax-output answer))]
    (- (Math/log answer-num))))


(defmethod loss :softmax-loss
  [loss-fn buffer-map]
  (let [output-channels (long (get loss-fn :output-channels 1))
        v (get buffer-map :output)
        target (get buffer-map :stream)]
      (if (= output-channels 1)
        (log-likelihood-softmax-loss v target)
        (let [n-pixels (quot (long (m/ecount v)) output-channels)]
          (loop [pix 0
                 sum 0.0]
            (if (< pix n-pixels)
              (recur (inc pix)
                     (double (+ sum
                                (log-likelihood-softmax-loss
                                 (m/subvector v (* pix output-channels) output-channels)
                                 (m/subvector target (* pix output-channels) output-channels)))))
              (double (/ sum n-pixels))))))))


(defn average-loss
  "V is inferences, target is labels.  Calculate the average loss
across all inferences and labels."
  ^double [loss-fn v-seq target-seq]
  (double
   (/ (->> (map (fn [v target]
                  {:output v
                   :stream target})
                v-seq target-seq)
       (map (partial loss loss-fn))
       (reduce +))
      (count v-seq))))


(defn max-index
  [coll]
  (second (reduce (fn [[max-val max-idx] idx]
                    (if (or (nil? max-val)
                            (> (coll idx) max-val))
                      [(coll idx) idx]
                      [max-val max-idx]))
                  [nil nil]
                  (range (count coll)))))

(defn softmax-result-to-unit-vector
  [result]
  (let [zeros (apply vector (repeat (first (m/shape result)) 0))]
    (assoc zeros (max-index (into [] (seq result))) 1.0)))


(defn softmax-results-to-unit-vectors
  [results]
  (let [zeros (apply vector (repeat (first (m/shape (first results))) 0))]
    (mapv #(assoc zeros (max-index (into [] (seq  %))) 1.0)
          results)))

(defn evaluate-softmax
  "Provide a percentage correct for softmax.  This is much easier to interpret than
the actual log-loss of the softmax unit."
  [guesses answers]
  (if (or (not (pos? (count guesses)))
          (not (pos? (count answers)))
          (not= (count guesses) (count answers)))
    (throw (Exception. (format "evaluate-softmax: guesses [%d] and answers [%d] count must both be positive and equal."
                               (count guesses)
                               (count answers)))))
  (let [results-answer-seq (mapv vector
                                 (softmax-results-to-unit-vectors guesses)
                                 answers)
        correct (count (filter #(m/equals (first %) (second %)) results-answer-seq))]
    (double (/ correct (count results-answer-seq)))))
