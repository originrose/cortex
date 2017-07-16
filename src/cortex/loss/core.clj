(ns cortex.loss.core
  "Definitions and implementations of cortex loss function terms.  A loss term is a function
that takes a map of arguments and returns a single double number.  The loss function for a
network is a weighted summation across a vector of terms.  The weight on any term is :lambda
and is defined with a per-term-default in the loss metadata for that function type.  The
default is 1.

The loss terms can have any number of uniquely named arguments and each argument can bind to
one of four different things.
1.  A node output
2.  A node parameter
3.  A stream
4.  Data tracked by the implementation with an initializer.
5. An augmented stream.  A pure transformation from a source stream to another stream.

The loss term can state which of it's arguments it produces gradients for when asked to produce
gradients. This makes it a little easier for implementations to manage gradients when
evaluating loss terms."
  (:require [clojure.core.matrix :as m]
            [cortex.util :refer [merge-args]]
            [cortex.graph :as graph]))


(defn loss-metadata
  "Get the metadata for a particular loss function.

There are some standard argument definitions:
:output - loss-term contains? :node-id output of the node.
:stream - loss-term contains? :stream the data coming from the data stream.
:parameter - loss-term contains? :node-id :parameter and this is the value of the parameter of
the node.

In addition loss functions themselves can have parameters in that the network stores in its
definition in parameter buffers.  This is useful in the case where the loss itself has some
running means associated with it like in the form of center loss.

All losses have a default lambda which weights them against any other elements in the loss.

In general, the loss function for a given network

The map must contain:
{:arguments - The keys and potentially some extra information about the data in the keys passed
to the loss function.
  }"
  [loss-term]
  (graph/get-node-metadata loss-term))


(defn is-loss-node?
  [node]
  (contains? (-> (graph/get-node-metadata node)
                 :passes
                 set)
             :loss))


(defn get-loss-lambda
  [loss-term]
  (or (get loss-term :lambda)
      (get (loss-metadata loss-term) :lambda 1.0)))



(defn get-loss-term-arguments
  "Flatten the metadata of the loss term with any specific argument information included
with the loss term itself."
  [loss-term]
  (graph/get-node-arguments loss-term))


(defn get-loss-term-args-of-type
  "Get loss term arguments of a specific type."
  [loss-term type]
  (->> (get-loss-term-arguments loss-term)
       (filter #(= (get % :type) type))))


(defn get-loss-term-parameters
  "A parameter for a loss term is an argument that has an initialization key.  This means it is
data specific to that loss term that will be updated during the term's execution but it also
needs to be saved if the network is saved.  Data specific to the loss term that does not need to
be saved is expected to be created by the implementation of the term itself.  These are mutually
exlusive so we should never see a loss term with two or more of them."
  [loss-term]
  (get-loss-term-args-of-type loss-term :parameter))

(defn get-loss-term-node-outputs
  "Get loss term arguments that connect to node outputs."
  [loss-term]
  (get-loss-term-args-of-type loss-term :node-output))

(defn get-loss-term-node-parameters
  "Get loss term arguments that connect to node parameters."
  [loss-term]
  (get-loss-term-args-of-type loss-term :node-parameter))

(defn get-loss-term-streams
  "Get loss term arguments that connect to data streams."
  [loss-term]
  (get-loss-term-args-of-type loss-term :stream))

(defn get-loss-term-augmented-streams
  "Loss loss term arguments that are formed from augmented streams of data"
  [loss-term]
  (get-loss-term-args-of-type loss-term :stream-augmentation))


(defmulti generate-loss-term
  "Given a map and a specific key, generate a loss term or return nil.  Used for auto
generating the loss terms from analyzing the graph nodes."
  (fn [item-key]
    item-key))


(defmethod generate-loss-term :default
  [& args]
  nil)


(defn loss-term-from-map-key-val
  [map-key loss-val]
  (when-let [retval (generate-loss-term map-key)]
    (when-not (or (map? loss-val)
                  (number? loss-val))
      (throw (ex-info "Loss values in nodes or parameters must be either maps or values"
                      {:key map-key
                       :value loss-val})))
    (if (map? loss-val)
      (merge retval loss-val)
      (assoc retval :lambda (double loss-val)))))


(defmulti loss
  "Implement a specific loss based on the type of the loss function.  They are passed map
containing the buffer coming from the network.
{:output output
 :target stream}"
  (fn [loss-term buffer-map]
    (:type loss-term)))


(defn average-loss
  "output is inferences, target is labels.  Calculate the average loss
  across all inferences and labels."
  ^double [loss-term output-seq label-seq]
  (let [loss-sequence (map (fn [v target]
                             (loss loss-term
                                   {:output v
                                    :labels target}))
                           output-seq label-seq)]
    (double
     (* (double (get-loss-lambda loss-term))
        (/ (apply + loss-sequence)
           (count output-seq))))))


(defn mse-loss
  "Mean squared error loss.  Applied to a node and a matching-size output stream."
  [& args]
  (merge-args
   {:type :mse-loss}
   args))


(defn l1-regularization
  "Penalize the network for the sum of the absolute values of a given buffer.
This pushes all entries of the buffer at a constant rate towards zero and is purported
to lead to more sparse representations.  This could be applied to either a trainable
parameter or to a node in which case it will be applied to the node's output buffer."
  [& args]
  (merge-args
   {:type :l1-regularization}
   args))


(defn l2-regularization
  "Penalize the network for the magnitude of a given buffer.  This will penalize large entries
in the buffer exponentially more than smaller entries leading to a buffer that tends to produce
an even distribution of small  entries.  Can be applied to either a trainable parameter or a
node in which case it will be applied to the node's output buffer."
  [& args]
  (merge-args
   {:type :l2-regularization}
   args))


(defn censor-loss
  [& {:as arg-map}]
  (merge {:type :censor-loss}
         arg-map))


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


(defn softmax-loss
  "Softmax loss.  Applied to a node and a softmax (1-hot encoded) output stream."
  [& args]
  (merge-args
   {:type :softmax-loss}
   args))


(defn loss-fn->table-str
  [loss-fn]
  (with-out-str
    (clojure.pprint/print-table
      [:type :value :lambda :node-id :argument]
                        (mapv (fn [loss-term]
                                (assoc loss-term
                                       :lambda
                                       (get-loss-lambda
                                        loss-term)
                                       :node-id
                                       (get-in loss-term
                                               [:output
                                                :node-id])
                                       :argument
                                       (get-in loss-term
                                               [:output
                                                :argument])))
                              loss-fn))))


(defn generate-loss-function
  "Given a graph with loss terms attached filter the nodes by if they are losses"
  [graph]
  (->> (get graph :nodes)
       vals
       (filter is-loss-node?)
       ;;When loss functions are just standard nodes then this will change
       (map #(dissoc %
                     :input-dimensions
                     :output-dimensions))
       vec))
