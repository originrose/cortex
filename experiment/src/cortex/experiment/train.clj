(ns cortex.experiment.train
  (:require [clojure.java.io :as io]
            [think.resource.core :as resource]
            [think.parallel.core :as parallel]
            [cortex.util :as util]
            [cortex.graph :as graph]
            [cortex.loss.core :as loss]
            [cortex.optimize :as opt]
            [cortex.optimize.adam :as adam]
            [cortex.nn.execute :as execute]
            [cortex.nn.compute-binding :as compute-binding]
            [cortex.nn.traverse :as traverse]
            [cortex.nn.network :as network])
  (:import [java.io File]))

(def default-network-filestem "trained-network")
(def trained-networks-folder "trained-networks/")

(set! *warn-on-reflection* true)

(defn load-network
  "Loads a map of {:cv-loss :network-description}."
  [network-filename]
  (if (.exists (io/file network-filename))
    (util/read-nippy-file network-filename)
    (throw (ex-info "Saved network not found." {:filename network-filename}))))


(defn save-network
  "Saves a trained network out to the filesystem."
  [network network-filename]
  (println "Saving network to" network-filename)
  (util/write-nippy-file network-filename network)
  network)


(defn default-network-test-fn
  "Given the context, old network, the new network and a test dataset,
  return a map indicating if the new network is indeed the best one
  and the network with enough information added to make comparing
  networks possible.
    {:best-network? boolean
     :network (assoc new-network :whatever information-needed-to-compare).}"
  ;; TODO: No need for context here.
  [simple-loss-print? network-filename
   ;; global arguments 
   {:keys [batch-size context]}  
   ;per-epoch arguments
   {:keys [new-network old-network test-ds]} ] 
  (let [batch-size (long batch-size)
        labels (execute/run new-network test-ds
                 :batch-size batch-size
                 :loss-outputs? true
                 :context context)
        loss-fn (execute/execute-loss-fn new-network labels test-ds)
        loss-val (apply + (map :value loss-fn))
        current-best-loss (if-let [best-loss (get old-network :cv-loss)]
                            ;; TODO: Is there a bug here? What if the best-loss isn't sequential?
                            (when (sequential? best-loss)
                              (apply + (map :value best-loss))))
        best-network? (or (nil? current-best-loss)
                          (< (double loss-val)
                             (double current-best-loss)))
        updated-network (assoc new-network :cv-loss loss-fn)]
    (println " saving current network iteration ")
    (save-network updated-network network-filename)
    
    (println (format "Loss for epoch %s: %s" (get new-network :epoch-count) loss-val))
    (when-not simple-loss-print?
      (println (loss/loss-fn->table-str loss-fn)))
    {:best-network? best-network?
     :network updated-network}))

(defn- per-epoch-fn
  [test-network-fn training-context epoch-args] 
  (let [test-results (test-network-fn training-context epoch-args)
        {:keys [network]} test-results]
    network))


(defn backup-trained-network
  [network-filestem]
  (let [network-filename (str network-filestem ".nippy")]
    (when (.exists (io/file network-filename))
      (let [backup-filename (->> (rest (range))
                                 (map #(format "%s%s-%s.nippy" trained-networks-folder network-filestem %))
                                 (remove #(.exists (io/file %)))
                                 (first))]
        (io/make-parents backup-filename)
        (io/copy (io/file network-filename)
                 (io/file backup-filename))))))


(defn- create-n-callable-fn
  [src-fn max-call-count]
  (if max-call-count
    (let [counter (atom 0)
          max-call-count (long max-call-count)]
      (fn []
        (when (< @counter max-call-count)
          (swap! counter inc)
          (src-fn))))
    src-fn))


(defn- to-epoch-seq-fn
  [item epoch-count]
  (if-not (fn? item)
    (parallel/create-next-item-fn
     (let [retval (if (map? (first item))
                    (repeat item)
                    item)]
       (if epoch-count
         (take epoch-count retval)
         retval)))
    (create-n-callable-fn item epoch-count)))

(defn- recur-train-network
  [network train-ds-fn test-ds-fn optimizer train-fn epoch-eval-fn]
  ;[{:keys [network train-ds-fn test-ds-fn optimizer train-fn epoch-eval-fn] :as recur-args}]
  (let [train-data (train-ds-fn)
        test-data (test-ds-fn)
        old-network network]
    (when (and train-data test-data)
      (let [{:keys [network optimizer]} (train-fn network train-data optimizer)
            epoch-args {:new-network (update network :epoch-count inc)
                        :old-network old-network :train-ds train-data
                        :test-ds test-data}
            network (epoch-eval-fn epoch-args)]
        (cons network
              (lazy-seq
               (recur-train-network network train-ds-fn test-ds-fn
                                    optimizer train-fn epoch-eval-fn)))))))


(defn train-n
  "Given a network description, start training from scratch or given a trained
  network continue training. Keeps track of networks that are actually improving
  against a test-ds.

  Networks are saved with a `:cv-loss` that is set to the best cv loss so far.

  This system expects a dataset with online data augmentation so that it is
  effectively infinite although the cross-validation and holdout sets do not
  change. By default, the best network is saved to: `trained-network.nippy`

  Note, we have to have enough memory to store the cross-validation dataset
  in memory while training.

  Every epoch a test function is called with these 2 map arguments:

  (test-fn global-context epoch-context)

  It must return a map containing at least:
    {:best-network? true if this is the best network
     :network The new network with any extra information needed for comparison assoc'd onto it.}

  If epoch-count is provided then we stop training after that many epochs else
  we continue to train forever."
  [network train-ds test-ds
   & {:keys [batch-size epoch-count
             network-filestem
             optimizer
             reset-score
             force-gpu?
             simple-loss-print?
             test-fn
             context]
      :or {batch-size 128
           network-filestem default-network-filestem
           reset-score false}}]
  (let [context (or context (execute/compute-context))]
    (execute/with-compute-context context
      (let [optimizer (or optimizer (adam/adam))
            context (execute/compute-context)
            network-filename (str network-filestem ".nippy")
            ;; If someone is training with an infinite data sequence they have to
            ;; actually pass in a function that when called returns the next epoch
            ;; of data.  This is the only way so far to avoid 'holding onto head'
            ;; when the number of epochs rises.
            train-ds-fn (to-epoch-seq-fn train-ds epoch-count)
            test-ds-fn (to-epoch-seq-fn test-ds epoch-count)
            network (if (vector? network)
                      (do
                        (backup-trained-network network-filestem)
                        (network/linear-network network))
                      (if reset-score
                        (assoc network :cv-loss {})
                        network))
            network (if (number? (get network :epoch-count))
                      network
                      (assoc network :epoch-count 0))
            train-fn #(execute/train %1 %2
                                     :batch-size batch-size
                                     :optimizer %3
                                     :context context)
            training-context {:batch-size batch-size :context context}
            test-fn  (or test-fn
                         (partial default-network-test-fn simple-loss-print? 
                                  network-filename))
            epoch-eval-fn (partial per-epoch-fn test-fn training-context)]
        (println "Training network:")
        (network/print-layer-summary network (traverse/training-traversal network))
        (->> (recur-train-network network train-ds-fn test-ds-fn optimizer train-fn epoch-eval-fn)
             last)))))


(defn print-trained-networks-summary
  "Prints a summary of the different networks trained so far.
  Respects an (optional) `network-filestem`."
  [& {:keys [network-filestem
             cv-loss->number
             cv-loss-display-precision
             extra-keys]
      :or {network-filestem default-network-filestem
           cv-loss->number #(apply + (vals %))
           cv-loss-display-precision 3}}]
  (let [cv-loss-format-string (format "%%.%sf" cv-loss-display-precision)]
    (->> trained-networks-folder
         io/file
         file-seq
         (filter #(let [n (.getPath ^File %)]
                    (and (.contains ^String n (.concat ^String trained-networks-folder
                                                       ^String network-filestem))
                         (.endsWith ^String n ".nippy"))))
         (map (fn [f] [f (util/read-nippy-file f)]))
         (map (fn [[f network]] (assoc network :filename (.getName ^File f))))
         (map (fn [network] (update network :cv-loss cv-loss->number)))
         (sort-by :cv-loss)
         (map (fn [network] (update network :cv-loss #(format cv-loss-format-string %))))
         (clojure.pprint/print-table (concat [:filename :cv-loss :parameter-count] extra-keys)))))
