(ns cortex.loss.yolo9000
  (:require [clojure.core.matrix :as m]
            [cortex.loss.core :as loss]
            [cortex.util :as util]
            [clojure.core.matrix.random :as m-rand]
            [cortex.tensor :as ct]
            [cortex.compute.cpu.tensor-math :as cpu-tm]))


(def grid-x 13)
(def grid-y 13)
(def grid-ratio [(/ grid-x) (/ grid-y)])
(def anchor-count 5)
(def anchors (m/reshape [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52] [5 2]))
(def classes ["boat" "sofa" "dog" "bird" "sheep" "cow" "bottle" "diningtable" "train" "bus" "chair" "motorbike"
              "person" "horse" "bicycle" "aeroplane" "car" "cat" "tvmonitor" "pottedplant"])
(def classes-count (count classes))
(def output-count (+ 5 classes-count))
(def width 416)
(def height 416)
(defn class-name->label [class] (mapv #(if (= class %) 1.0 0.0) classes))
(defn label->class-name [label] (nth classes (util/max-index label)))
(def SCALE_NOOB 0.5)
(def SCALE_CONF 5)
(def SCALE_COOR 5)
(def SCALE_PROB 1)
(def x-y [0 1])
(def w-h [2 3])
(def bb [0 1 2 3])
(def conf [4])


;; prediction has shape [grid-x grid-y anchor-count output-count]
;; normalization and ranges:
;; truth:
;; (x,y) are in range 0-1, corresponding to displacement from upper left corner of a cell,
;; (w,h) are in range 0-13, corresponding to the fact that a width may be anything.
;; pred:
;; (x,y) are arbirary, we sigmoid them to get to the (0,1) range.
;; (w,h) are factors that get multiplied by the anchor ratios,

;; This is the prediction formatted for loss - x,y,conf get sigmoid, prob gets softmax, and w-h gets weirdness.
;; Now on to creating the indicator function D.  This is a rank three tensor such that
;; (m/select D x y i) is 1 when an object has its center in cell (x, y), and anchor box i achieves the greatest
;; iou between the truth bounding box and the predicted bounding-boxes.

(defn sigmoid [x] (/ 1 (+ 1 (Math/exp (- x)))))

(defn un-sigmoid [x] (-> (/ 1 x) (- 1) (Math/log) (-)))

(defn sigmoid-gradient [x] (* x (- 1 x)))

(defn softmax [v]
  (let [exp-v (m/exp v)]
    (m/div exp-v (m/esum exp-v))))

(defn ul
  "Return x-y coords of upper left corner of box"
  [[x y w h]]
  [(- x (/ w 2)) (- y (/ h 2))])

(defn br
  "Return x-y coords of bottom corner of box"
  [[x y w h]]
  [(+ x (/ w 2)) (+ y (/ h 2))])

(defn area [[x y]]
  (* x y))

(defn iou
  "Calculates intersection over union of boxes b = [x y w h].
  **Be sure that x,y and w,h are using the same distance units.**"
  [b1 b2]
  (let [UL (m/emap max (ul b1) (ul b2))
        BR (m/emap min (br b1) (br b2))
        intersection (area (map max (m/sub BR UL) [0 0]))
        union (+ (area (drop 2 b1)) (area (drop 2 b2)) (- intersection))
        result (if (< 0 union) (/ intersection union) 0)]
    result))

(defn select-anchor
  "Given the vector iou of iou-s with the different anchors, returns a one-hot encoded vector at the max value."
  [iou]
  (let [max-iou (m/emax iou)]
    (if (< 0 max-iou)
      (m/eq iou max-iou)
      (m/zero-vector anchor-count))))


(defn ->weights [formatted-prediction truth]
  (let [formatted-pred-selector (partial m/select formatted-prediction :all :all :all)
        truth-selector (partial m/select truth :all :all :all)
        pred-boxes (-> (formatted-pred-selector bb)
                       (m/reshape [(* grid-x grid-y anchor-count) (count bb)]))
        truth-boxes (-> (truth-selector bb)
                        (m/reshape [(* grid-x grid-y anchor-count) (count bb)]))
        ious (-> (map iou pred-boxes truth-boxes)
                 ;; value at (+ (* i grid-x) j) is the vector of ious for each of the five anchors at cell (i,j)
                 (m/reshape [(* grid-x grid-y) anchor-count]))
        ;; value at (+ (* i grid-x) j) is [0 0 1 0 0] if third anchor achieves iou max at cell (i,j)
        D (-> (map select-anchor ious)
              ;; value at [i j anchor-idx] is 1 if anchor-idx achieves max for cell (i,j), otherwise 0.
              (m/reshape [grid-x grid-y anchor-count])
              ;; tensor on 1's to get shape [grid-x grid-y anchor-count output-count]
              (m/outer-product (m/assign (m/new-vector output-count) 1.0)))
        scale-vec-ob (->> (m/join
                            (m/assign (m/zero-vector (count bb)) SCALE_COOR)
                            (m/assign (m/zero-vector (count conf)) SCALE_CONF)
                            (m/assign (m/zero-vector classes-count) SCALE_PROB))
                          (m/broadcast-like formatted-prediction))
        scale-vec-noob (->> (m/join
                              (m/zero-vector (count bb))
                              (m/assign (m/zero-vector (count conf)) SCALE_NOOB)
                              (m/zero-vector classes-count))
                            (m/broadcast-like formatted-prediction))]
    (m/add
      (m/emul D scale-vec-ob)
      (m/emul (m/sub 1 D) scale-vec-noob))))


(defn format-prediction
  [pred]
  (let [pred-selector (partial m/select pred :all :all :all)
        x-y-vals      (->> [0 1] pred-selector (m/emap sigmoid))
        w-h-vals      (->> [2 3] pred-selector (m/emul grid-ratio) (m/exp) (m/emul anchors) (m/sqrt))
        conf-vals     (->> [4] pred-selector (m/emap sigmoid))
        prob-vals     (-> (pred-selector (range 5 output-count))
                          (m/reshape [(* grid-x grid-y anchor-count) classes-count])
                          ((fn [v] (map softmax v)))
                          (m/reshape [grid-x grid-y anchor-count classes-count]))]
    (m/join-along 3 x-y-vals w-h-vals conf-vals prob-vals)))


(defn ct-sigmoid!
  [tens]
  (ct/unary-op! tens 1.0 tens :logistic))

(defn ct-emul!
  [val tens]
  (ct/binary-op! tens 1.0 tens 1.0 val :*))

(defn ct-exp!
  [tens]
  (ct/unary-op! tens 1.0 tens :exp))

(defn ct-sqrt!
  [tens]
  (ct/unary-op! tens 1.0 tens :sqrt))

(defn ct-reshape
  [tens shape]
  (ct/in-place-reshape tens shape))

(defn ct-softmax!
  [tens]
  (ct/softmax! tens tens))


(defn format-prediction-ct!
  [pred ct-grid-ratio ct-anchors]
  (let [pred-selector (partial ct/select pred :all :all :all)]
    ;;x-y-vals
    (->> [0 1] pred-selector ct-sigmoid!)
    ;;w-h-vals
    (->> [2 3] pred-selector (ct-emul! ct-grid-ratio) (ct-exp!) (ct-emul! ct-anchors) (ct-sqrt!))
    ;;conf-vals
    (->> [4] pred-selector ct-sigmoid!)
    (comment
     ;;prob-vals
     (-> (pred-selector (range 5 output-count))
         (ct-reshape [(* grid-x grid-y anchor-count) classes-count])
         ct-softmax!))
    pred))


(defmethod loss/loss :yolo9000
  [loss-term buffer-map]
  (cpu-tm/tensor-context
   (let [truth (get buffer-map :labels)
         label (get buffer-map :output)
         formatted-prediction (format-prediction label)
         ct-grid-ratio (ct/->tensor grid-ratio)
         ct-anchors (ct/->tensor anchors)
         ct-formatted-prediction (format-prediction-ct! (ct/->tensor label) ct-grid-ratio ct-anchors)
         weights (->weights formatted-prediction truth)]
     (let [correct-format (m/to-double-array formatted-prediction)
           ct-format (ct/to-double-array ct-formatted-prediction)]
       (clojure.pprint/pprint ["survey says..." {:equality-check  (m/equals correct-format
                                                                            ct-format
                                                                            1e-4)
                                                 :correct-ct-pairs (mapv vector
                                                                         (vec (take output-count
                                                                                    (drop output-count correct-format)))
                                                                         (vec (take output-count
                                                                                    (drop output-count ct-format))))}]))
     (->> (m/sub formatted-prediction truth)
          (m/square)
          (m/emul weights)
          (m/esum)))))


(defn- read-label
  []
  (let [data (util/read-nippy-file "yololoss.nippy")]
    (m/reshape (get data :data) (get data :shape))))


(defn- test-loss
  []
  (let [pred (m-rand/sample-uniform [grid-x grid-y anchor-count output-count])
        truth (read-label)]
    (loss/loss {:type :yolo9000} {:labels truth
                                  :output pred})))
