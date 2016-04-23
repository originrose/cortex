(ns cortex.tree
  (:require [clojure.core.matrix :as mat]
            [clojure.zip :as zip]))

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

; Decision Tree Algorithm
; -

(defrecord TreeNode [split target children])

(defn walker
  [root]
  (zip/zipper
    ; only leaf nodes with a target value can't have children
    (fn branch? [node] (nil? (:target node)))
    (fn children [node] (:children node))
    (fn make-node [node children] (TreeNode. (:split node) (:target node) children))
    root))

(defn mode
  "Returns the most comment value in a seq of items."
  [items]
  (first (last (sort-by val (frequencies items)))))

(defn rand-splitter
  "Returns a vector of [feature-indices split-value] where the feature-index for the
  chosen dimension has been removed from feature-indices.

  Note: assumes the feature-indices are already in random order."
  [X feature-indices]
  (let [feature-index (first feature-indices) ;(rand-nth feature-indices)
        feature-vals (mat/select X :all feature-index)
        [min-val max-val] (reduce
                            (fn [[min-v max-v] v]
                              [(min min-v v) (max max-v v)])
                            [(first feature-vals) (first feature-vals)]
                            feature-vals)
        rand-split (+ min-val (* (- max-val min-val) (rand)))]
    [feature-index (next feature-indices) rand-split]))

(defn split-samples
  "Returns a pair of lazy seqs of row indices of X that fall either to the left or right of the
  given split value and feature dimension.

  [left-samples right-samples]
  "
  [X feature-index split-val]
  (let [indices (mat/le (mat/select X :all feature-index) split-val)]
    [(remove nil? (map-indexed (fn [i v] (if (zero? v) nil i)) indices))
     (remove nil? (map-indexed (fn [i v] (if (zero? v) i nil)) indices))]))

(defn decision-tree*
  [X Y {:keys [n-samples n-features split-fn x-indices feature-indices] :as options}]
  (cond
    (empty? indices))
  (let [feature-indices (take-rand n-features (or indices (range (mat/row-count X))))
        [feature-index feature-indices split-val] (split-fn X feature-indices)
        [left-indices right-indices] (split-samples X feature-index split-val)
        left-tree  (decision-tree* X Y (assoc options :x-indices left-indices))
        right-tree (decision-tree* X Y (assoc options :x-indices right-indices))
        root (TreeNode. split-val nil [left-tree right-tree])]
    root))

(defn decision-tree
  [X Y {:keys [n-samples n-features split-fn x-indices feature-indices] :as options}]
  (decision-tree* X Y ))

; TODO:
; - support setting the n-features and n-samples for changing the boosting
; properties, or alternatively maybe support feature and/or sample weighting
(defn random-forest
  "A random forest is composed of many decision trees fit to the data X and labels Y."
  [X Y & {:keys [n-trees n-samples n-features criterion-fn split-fn] :as options}]
  (let [[n-samples n-features] (mat/shape X)
        options (if split-fn options
                  (assoc options :split-fn rand-splitter))]
    (repeatedly n-trees #(decision-tree X Y options))))

(defn tree-predict
  [tree X]
  nil)

(defn predict
  "Predict the regression target Y given a model and X by averaging
  the predictions of the underlying trees."
  [model X]
  (let [estimates (pmap #(tree-predict % X) (:trees model))]
    (/ (apply + estimates)
       (count estimates))))
