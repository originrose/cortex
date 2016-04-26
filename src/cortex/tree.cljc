(ns cortex.tree
  (:require [clojure.core.matrix :as mat]
            [clojure.zip :as zip]
            [rhizome.viz :as viz]))

(defn shuffle-with-seed
  "Return a random permutation of coll with a seed."
  [coll seed]
  (let [al (java.util.ArrayList. coll)
        rnd (java.util.Random. seed)]
    (java.util.Collections/shuffle al rnd)
    (clojure.lang.RT/vector (.toArray al))))

(defn take-rand
  [n coll & [seed]]
  (take n (if seed
            (shuffle-with-seed coll seed)
            (shuffle coll))))

(defn take-rand-with-replacement
  [n coll]
  (repeatedly n #(rand-nth coll)))

(defrecord TreeNode [feature-index split target children])

(defn walker
  [root]
  (zip/zipper
    ; only leaf nodes with a target value can't have children
    (fn branch? [node] (nil? (:target node)))
    (fn children [node] (:children node))
    (fn make-node [node children] (TreeNode. (:feature-index node) (:split node) (:target node) children))
    root))

(defn node-seq
  [root]
  (tree-seq
    (fn branch? [node] (nil? (:target node)))
    (fn children [node] (:children node))
    root))

(defn indexed-array
  [X indices]
  (map #(mat/get-row X %) indices))

(defn mode
  "Returns the most comment value in a seq of items."
  [items]
  (first (last (sort-by val (frequencies items)))))

(defn split-samples
  "Returns a pair of lazy seqs of row indices of X that fall either to the left or right of the
  given split value and feature dimension.

  [left-samples right-samples]
  "
  [X x-indices feature-index split-val]
  (reduce (fn [[le gt] x-index]
            (if (<= (mat/mget X x-index feature-index) split-val)
              [(conj le x-index) gt]
              [le (conj gt x-index)]))
          [[] []]
          x-indices))

(defn rand-splitter
  "Returns a vector of [feature-indices split-value] where the feature-index for the
  chosen dimension has been removed from feature-indices.

  Note: assumes the feature-indices are already in random order."
  [X Y feature-indices sample-indices options]
  (let [feature-index (first feature-indices) ;(rand-nth feature-indices)
        feature-vals (mat/select X :all feature-index)
        [min-val max-val] (reduce
                            (fn [[min-v max-v] v]
                              [(min min-v v) (max max-v v)])
                            [(first feature-vals) (first feature-vals)]
                            feature-vals)
        rand-split (+ min-val (* (- max-val min-val) (rand)))]
    [feature-index rand-split]))

(defn gini-score
  "Given all possible labels and a frequencies map of sample labels returns
  the Gini impurity measure.  Zero is the best score, where all samples have the
  same label."
  [labels freqs]
  (let [k (apply + (vals freqs))
        squared-fractions (map #(let [f (/ (get freqs % 0) k)]
                                  (* f f))
                               labels)]
    (- 1 (apply + squared-fractions))))

(defn best-splitter
 [X Y feature-indices sample-indices {:keys [max-features] :as options}]
 (let [labels (set (indexed-array Y sample-indices))
       scores (for [feature-index (take-rand (or max-features (count feature-indices)) feature-indices)]
                (for [split-val (map #(mat/mget X % feature-index) sample-indices)]
                  (let [split-sample-indices (split-samples X sample-indices feature-index split-val)
                        score (if (some empty? split-sample-indices)
                                1.0
                                (/ (apply + (map #(gini-score labels (frequencies (indexed-array Y %)))
                                                 split-sample-indices))
                                   2))]
                    [score feature-index split-val])))
       scores (apply concat scores)
       best-score (apply min-key first scores)]
   (next best-score)))

(defn decision-tree*
  [X Y {:keys [n-samples n-features split-fn feature-indices sample-indices] :as options}]
  (let [labels (doall (indexed-array Y sample-indices))]
    (cond
      ; All remaining samples have the same class => leaf node
      (= 1 (count (set labels))) (TreeNode. nil nil (first labels) [])

      ; All features have been split on already => leaf node
      (empty? feature-indices)   (TreeNode. nil nil (mode labels) [])

      ; TODO:
      ; - Add support for other stopping modes:
      ;   * max-depth: max depth of the tree
      ;   * max-leaves: max number of leaves in the tree
      ;   * min-branch-samples: min number of samples to still branch
      ;   * max-features: sample up to max-features without replacement to find
      ;   the best dimension on which to split at each node
      ;   * min-leaf-weight: if using weighted samples make a leaf when weight
      ;   combined sample weight at a node drops below a certain value

      ; Determine the next feature and split-val, then point to the children nodes
      :else
      (let [[feature-index split-val] (split-fn X Y feature-indices sample-indices options)
            feature-indices (filter #(not= feature-index %) feature-indices)
            [left-indices right-indices] (split-samples X sample-indices feature-index split-val)
            left-tree  (decision-tree* X Y (assoc options
                                                  :feature-indices feature-indices
                                                  :sample-indices left-indices))
            right-tree (decision-tree* X Y (assoc options
                                                  :feature-indices feature-indices
                                                  :sample-indices right-indices))
            root (TreeNode. feature-index split-val nil [left-tree right-tree])]
        root))))

; TODO:
; - sample weighting?
(defn decision-tree
  "Build a decision tree.

  - n-samples will be sampled with replacement from the samples X or the subset of X specified
  by passing sample-indices
  - n-features will be sampled from the columns of X or the specified feature-indices
  - the split-fn will be called with [X feature-indices] and it must return
    [feature-index rand-split]
  "
  [X Y {:keys [n-samples n-features split-fn sample-indices feature-indices] :as options}]
  (assert (= (mat/row-count X) (mat/row-count Y))
          "Training data X and labels Y must have the same number of rows.")
  (let [n-samples (or n-samples (mat/row-count X))
        sample-indices (take-rand-with-replacement n-samples
                                                   (or sample-indices (range (mat/row-count X))))
        n-features (or n-features (mat/column-count X))
        feature-indices (or feature-indices (take-rand n-features (range (mat/column-count X))))
        split-fn (or split-fn best-splitter)
        options (assoc options
                       :n-samples n-samples
                       :sample-indices sample-indices
                       :n-features n-features
                       :feature-indices feature-indices
                       :split-fn split-fn)]
    (decision-tree* X Y options)))

(defn view-tree
  [t]
  (let [node-label
        (fn [n]
          {:label
           (if-let [split (:split n)]
             (format "X[%d] <= %.2f" (:feature-index n) split)
             (format "Y = %s" (:target n)))})]
    (viz/view-graph (node-seq t) :children
                    :node->descriptor node-label)))

(defn tree-search
  [node sample]
  (if (:target node)
    node
    (let [children (:children node)
          left? (<= (mat/mget sample (:feature-index node))
                    (:split node))]
      (if left?
        (recur (first children) sample)
        (recur (last children) sample)))))

(defn tree-classify
  [tree sample]
  (:target (tree-search tree sample)))

(defn tree-classify-dataset
  [tree data labels]
  (frequencies
    (map-indexed
      (fn [i row]
        (= (tree-classify tree row)
           (mat/mget labels i)))
      data)))

; TODO:
; - support setting the n-features and n-samples for changing the boosting
; properties, or alternatively maybe support feature and/or sample weighting
(defn random-forest
  "A random forest is composed of many decision trees fit to the data X and labels Y."
  [X Y {:keys [n-trees n-samples n-features]:as options}]
  (let [[n-samples n-features] (mat/shape X)]
    (pmap (fn [_] (decision-tree X Y options)) (range n-trees))))

(defn forest-classify
  [forest sample]
  (mode (map #(tree-classify % sample) forest)))

(defn forest-classify-dataset
  [forest data]
  (apply merge-with + (map #(tree-classify-dataset % data) forest)))

;(defn predict
;  "Predict the regression target Y given a model and X by averaging
;  the predictions of the underlying trees."
;  [model X]
;  (let [estimates (pmap #(tree-predict % X) (:trees model))]
;    (/ (apply + estimates)
;       (count estimates))))
