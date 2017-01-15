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
:output - loss-term contains? :node-id output of the node.
:stream - loss-term contains? :stream the data coming from the data stream.
:parameter - loss-term contains? :node-id :parameter and this is the value of the parameter of the node.

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
  [loss-term]
  {:arguments #{:output :stream}
   :lambda 1.0})

(defn get-loss-lambda
  [loss-term]
  (or (get loss-term :lambda)
      (get (loss-metadata loss-term) :lambda 1.0)))

(defn get-loss-term-arguments
  [loss-term]
  (get (loss-metadata loss-term) :arguments #{:output :stream}))

(defn get-loss-parameters
  [loss-term]
  (->> (get (loss-metadata loss-term) :parameters)
       (map (fn [{:keys [key] :as param}]
              (merge param
                     (get loss-term param))))))


(defmulti generate-loss-term
  "Given a map and a specific key, generate a loss term or return nil.  Used for auto
generating the loss terms from analyzing the graph nodes."
  (fn [item-key]
    item-key))


(defmethod generate-loss-term :default
  [& args]
  nil)


(defn generic-loss-term
  "Generate a generic loss term from a loss type"
  [loss-type]
  {:type loss-type
   :lambda (get-loss-lambda {:type loss-type})})


(defmulti get-stream-size
  "Get the stream size we expect for a node.  Defaults to the output size of the node.
Can be nil in which case it is assumed that the loss term cannot decifer the valid
output size from information in the graph node."
  (fn [loss-term node]
    (get loss-term :type)))


(defmethod get-stream-size :default
  [loss-term node]
  (get node :output-size))

(defn mse-loss
  "Mean squared error loss.  Applied to a node and a matching-size output stream."
  [& args]
  (merge-args
   {:type :mse-loss}
   args))


(defmethod generate-loss-term :mse-loss
  [item-key]
  (generic-loss-term item-key))


(defn softmax-loss
  "Softmax loss.  Applied to a node and a softmax (1-hot encoded) output stream."
  [& args]
  (merge-args
   {:type :softmax-loss}
   args))

(defmethod generate-loss-term :softmax-loss
  [item-key]
  (generic-loss-term item-key))

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
  [loss-term]
  (let [arguments (if (contains? loss-term :parameter)
                    #{{:parameter (get loss-term :parameter)}}
                    #{:output})]
    {:arguments arguments
     :lambda 0.001}))


(defmethod loss-metadata :l1-regularization
  [loss-term]
  (reg-loss-metadata loss-term))



(defmethod generate-loss-term :l1-regularization
  [item-key]
  (generic-loss-term item-key))


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
  [loss-term]
  (reg-loss-metadata loss-term))


(defmethod generate-loss-term :l2-regularization
  [item-key]
  (generic-loss-term item-key))


(defn center-loss
  "Center loss is a way of specializing an activation for use as a grouping/sorting
mechanism.  It groups activations by class and develops centers for the activations
over the course of training.  Alpha is a number between 1,0 that stands for the exponential
decay factor of the running centers (essentially running means).  The equation to update a mean
is alpha*current + (1 - alpha)*new-mean.  The network is penalized for the distance of the current
activations from their respective centers.  The result is that the activation itself becomes very
grouped by class and thus make far better candidates for LSH or a distance/sorting system.  Note
that this is a loss used in the middle of the graph, not at the edges.  This is applied to a
given node and needs an softmax-type output stream.
http://ydwen.github.io/papers/WenECCV16.pdf"
  [& {:keys [alpha] :as arg-map
      :or {alpha 0.5}}]
  (merge {:type :center-loss
          :alpha alpha}
         arg-map))

(defn stream-map-entry->size
  ^long [entry]
  (long
   (if (number? entry)
     (long entry)
     (get entry :size))))

(defn stream->size
  [stream-map stream]
  (if-let [entry (get stream-map stream)]
    (stream-map-entry->size entry)
    (throw (ex-info "Failed to find stream:"
                    {:stream stream
                     :stream-map stream-map}))))


(defn- get-center-loss-center-buffer-shape
  "Get the shape of the centers of the network.  The network must be built and enough
information must be known about the dataset to make a stream->size map."
  [loss-term node-id->node-map stream-map]
  (let [node-output-size (get-in node-id->node-map [(get loss-term :node-id) :output-size])
        stream-size (stream->size stream-map (get loss-term :stream))]
    (when-not (and node-output-size
                   stream-size)
      (throw (ex-info "Center loss failed to find either node output size or stream size"
                      {:loss-function loss-term
                       :output-size node-output-size
                       :stream-size stream-size})))
    ;;We keep track of stream-size centers each of node output size.
    [(long stream-size) (long node-output-size)]))


(defmethod loss-metadata :center-loss
  [loss-term]
  {:parameters [{:key :centers
                 :shape-fn get-center-loss-center-buffer-shape
                 :initialization {:type :constant
                                  :value 0}}]
   :arguments #{:output :stream}
   :lambda 0.1})


(defmethod generate-loss-term :center-loss
  [item-key]
  (generic-loss-term item-key))


(defmethod get-stream-size :center-loss
  [loss-term node]
  nil)


(defmulti loss
  "Implement a specific loss based on the type of the loss function.  They are passed map
containing the buffer coming from the network.
{:output output
 :target stream}"
  (fn [loss-term buffer-map]
    (:type loss-term)))


(defmethod loss :mse-loss
  [loss-term buffer-map]
  (let [v (get buffer-map :output)
        target (get buffer-map :stream)]
   (/ (double (m/magnitude-squared (m/sub v target)))
      (m/ecount v))))


(defn log-likelihood-softmax-loss
  ^double [softmax-output answer]
  (let [answer-num (m/esum (m/mul softmax-output answer))]
    (- (Math/log answer-num))))


(defmethod loss :softmax-loss
  [loss-term buffer-map]
  (let [output-channels (long (get loss-term :output-channels 1))
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
  ^double [loss-term v-seq target-seq]
  (double
   (/ (->> (map (fn [v target]
                  {:output v
                   :stream target})
                v-seq target-seq)
       (map (partial loss loss-term))
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
