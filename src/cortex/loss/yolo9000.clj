(ns cortex.loss.yolo9000
  (:require [clojure.core.matrix :as m]
            [cortex.loss.core :as loss]
            [cortex.util :as util]
            [clojure.core.matrix.random :as m-rand]
            [cortex.tensor :as ct]
            [cortex.compute.cpu.tensor-math :as cpu-tm]
            [cortex.compute.cuda.tensor-math :as gpu-tm]
            [clojure.math.combinatorics :as combo]
            [cortex.compute.driver :as drv]))


(def grid-x 13)
(def grid-y 13)
(def grid-ratio [(/ grid-x) (/ grid-y)])
(def anchor-count 5)
(def anchors (m/reshape [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52] [5 2]))
(def classes ["boat" "sofa" "dog" "bird" "sheep" "cow" "bottle" "diningtable" "train" "bus" "chair" "motorbike"
              "person" "horse" "bicycle" "aeroplane" "car" "cat" "tvmonitor" "pottedplant"])
(def classes-count (count classes))
(def output-count (+ 5 classes-count))
(def SCALE_NOOB 0.5)
(def SCALE_CONF 5)
(def SCALE_COOR 5)
(def SCALE_PROB 1)
(def x-y [0 1])
(def w-h [2 3])
(def bb [0 1 2 3])
(def conf [4])
(def class-dims (range 5 output-count))


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

(defn softmax-gradient
  "Softmax gradient is a matrix whose ij-entry is v_jdelta_ij - v_jv_i.
  Written globally that is Diagonal(v) minus v-tensor-v."
  [v]
  (m/sub (m/diagonal-matrix v) (m/outer-product v v)))

(comment
  [1 2 3]
  [[0 -2 -3]
   [-2 2 -6]
   [-3 -6 6]])


(defn ul
  "Return x-y coords of upper left corner of box"
  [[x y w h]]
  [(- x (/ w 2)) (- y (/ h 2))])


(defn coords-ct
  "Return x-y coords of upper left corner of box and lower right."
  [box-vec ul-vec br-vec]
  (let [wh-vec (ct/select box-vec :all (range 2 4))
        xy-vec (ct/select box-vec :all (range 2))]
    (ct/unary-op! br-vec 0.5 wh-vec :noop)
    (ct/binary-op! ul-vec 1.0 xy-vec 1.0 br-vec :-)
    (ct/binary-op! br-vec 1.0 xy-vec 1.0 br-vec :+)))


(defn br
  "Return x-y coords of bottom corner of box"
  [[x y w h]]
  [(+ x (/ w 2)) (+ y (/ h 2))])

(defn area [[x y]]
  (* x y))

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

(defn ct-max!
  [a b]
  (ct/binary-op! a 1.0 a 1.0 b :max))

(defn ct-min!
  [a b]
  (ct/binary-op! a 1.0 a 1.0 b :min))

(defn ct-sub!
  [a b]
  (ct/binary-op! a 1.0 a 1.0 b :-))

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

(defn area-ct
  [result wh-vec]
  (ct/binary-op! result
                 1.0 (ct/select wh-vec :all 0)
                 1.0 (ct/select wh-vec :all 1)
                 :*))


(defn iou-ct
  "Calculates intersection over union of boxes b = [x y w h].
  **Be sure that x,y and w,h are using the same distance units.**"
  [b1 b2]
  (let [[num-box col-count] (m/shape b1)
        b1-ul (ct/new-tensor [num-box 2])
        b1-br (ct/new-tensor [num-box 2])
        b2-ul (ct/new-tensor [num-box 2])
        b2-br (ct/new-tensor [num-box 2])
        overlap (ct/new-tensor [num-box 2])
        intersection (ct/new-tensor [num-box])
        area-b1 (ct/new-tensor [num-box])
        area-b2 (ct/new-tensor [num-box])
        union (ct/new-tensor [num-box])
        result (ct/new-tensor [num-box])
        _ (coords-ct b1 b1-ul b1-br)
        _ (coords-ct b2 b2-ul b2-br)
        UL (ct-max! b1-ul b2-ul)
        BR (ct-min! b1-br b2-br)
        overlap (-> (ct-sub! BR UL)
                    (ct-max! 0))]
    (area-ct intersection overlap)
    (area-ct area-b1 (ct/select b1 :all (range 2 4)))
    (area-ct area-b2 (ct/select b2 :all (range 2 4)))
    (ct/binary-op! union 1.0 area-b1 1.0 area-b2 :+)
    (ct/binary-op! union 1.0 union 1.0 intersection :-)
    (ct/binary-op! result 1.0 intersection 1.0 union :/)
    (ct/ternary-op! result 1.0 union 0.0 0.0 1.0 result :select)
    result))



(defn test-iou
  []
  (gpu-tm/tensor-context
   (let [src-boxes [[10 10 20 20]
                    [5 5 5 5]
                    [10 10 5 5]
                    [17 17 2 2]]
         box-pairs (mapv (fn [[x y]]
                           [(get src-boxes x)
                            (get src-boxes y)])
                         (combo/combinations (range (count src-boxes)) 2))
         b1 (ct/->tensor (map first box-pairs))
         b2 (ct/->tensor (map second box-pairs))]
     (let [correct (double-array (mapv (partial apply iou) box-pairs))
           guess (ct/to-double-array (iou-ct b1 b2))]
       (println {:correct (vec correct)
                 :guess (vec guess)})))))



(defn select-anchor
  "Given the vector iou of iou-s with the different anchors, returns a one-hot encoded vector at the max value."
  [iou]
  (let [max-iou (m/emax iou)]
    (if (< 0 max-iou)
      (m/eq iou max-iou)
      (m/zero-vector anchor-count))))


(defn ct-one-hot-encode
  "Given the vector iou of iou-s with the different anchors, returns a one-hot encoded vector at the max value.
If there are two equal max values then you will get a two-hot encoded vector."
  [data-matrix]
  (let [[n-rows n-col] (m/shape data-matrix)
        max-data (ct/new-tensor [n-rows 1])]

    (ct/unary-reduce! max-data 1.0 data-matrix :max)
    (ct/binary-op! data-matrix 1.0 data-matrix 1.0 max-data :eq)
    (ct/binary-op! max-data 1.0 max-data 0.0 0.0 :eq)
    (ct/binary-op! max-data 1.0 1.0 1.0 max-data :-)
    (ct/binary-op! data-matrix 1.0 data-matrix 1.0 max-data :*)))


(defn test-one-hot-encode
  []
  (gpu-tm/tensor-context
   (->> (ct/->tensor [[0 0 0 0]
                      [0 0 0 0]
                      [0 0 2 0]
                      [1 4 2 2]])
        ct-one-hot-encode
        ct/to-double-array
        vec)))




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

        one-hots (map select-anchor ious)
        D (-> one-hots
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
                            (m/broadcast-like formatted-prediction))
        D     (m/add
               (m/emul D scale-vec-ob)
               (m/emul (m/sub 1 D) scale-vec-noob))]
    {:ious ious
     :one-hots one-hots
     :weights D}))


(defn ->weights-ct
  [formatted-prediction truth]
  (let [formatted-pred-selector (partial ct/select formatted-prediction :all :all :all)
        truth-selector (partial ct/select truth :all :all :all)
        pred-boxes (-> (formatted-pred-selector bb)
                       (ct-reshape [(* grid-x grid-y anchor-count) (count bb)]))
        truth-boxes (-> (truth-selector bb)
                        (ct-reshape [(* grid-x grid-y anchor-count) (count bb)]))
        ious (-> (iou-ct pred-boxes truth-boxes)
                 ;; value at (+ (* i grid-x) j) is the vector of ious for each of the five anchors at cell (i,j)
                 (ct-reshape [(* grid-x grid-y) anchor-count]))


        all-d (-> (ct/new-tensor (m/shape formatted-prediction)))
        one-hots (-> (ct-one-hot-encode (m/assign! (ct/new-tensor (m/shape ious)) ious))
                     ;; value at [i j anchor-idx] is 1 if anchor-idx achieves max for cell (i,j), otherwise 0.
                     (ct-reshape [grid-x grid-y anchor-count 1]))

        D (ct/assign! all-d one-hots)
        one-minus-one-hots (ct/binary-op! (ct/new-tensor (m/shape one-hots)) 1.0 1 1.0 one-hots :-)
        one-minus-d (ct/assign! (ct/new-tensor (m/shape formatted-prediction)) one-minus-one-hots)
        scale-vec-ob (-> (ct/->tensor (concat (repeat (count bb) SCALE_COOR)
                                              (repeat (count conf) SCALE_CONF)
                                              (repeat classes-count SCALE_PROB)))
                         (ct/in-place-reshape [1 1 1 output-count]))
        scale-vec-noob (-> (ct/->tensor (concat (repeat (count bb) 0)
                                                (repeat (count conf) SCALE_NOOB)
                                                (repeat classes-count 0)))
                           (ct-reshape [1 1 1 output-count]))]
    (ct/binary-op! D 1.0 D 1.0 scale-vec-ob :*)
    (ct/binary-op! one-minus-d 1.0 one-minus-d 1.0 scale-vec-noob :*)
    (ct/binary-op! D 1.0 D 1.0 one-minus-d :+)
    {:ious ious
     :one-hots one-hots
     :weights D}))



(defn format-prediction
  [pred]
  (let [pred-selector (partial m/select pred :all :all :all)
        x-y-vals      (->> x-y pred-selector (m/emap sigmoid))
        w-h-vals      (->> w-h pred-selector (m/emul grid-ratio) (m/exp) (m/emul anchors) (m/sqrt))
        conf-vals     (->> conf pred-selector (m/emap sigmoid))
        prob-vals     (-> (pred-selector class-dims)
                          (m/reshape [(* grid-x grid-y anchor-count) classes-count])
                          ((fn [v] (map softmax v)))
                          (m/reshape [grid-x grid-y anchor-count classes-count]))]
    (m/join-along 3 x-y-vals w-h-vals conf-vals prob-vals)))


(defn format-prediction-ct!
  [pred]
  (let [ct-grid-ratio (ct/->tensor grid-ratio)
        ct-anchors (ct/->tensor anchors)
        pred-selector (partial ct/select pred :all :all :all)]
    ;;x-y-vals
    (->> x-y pred-selector ct-sigmoid!)
    ;;w-h-vals
    (->> w-h pred-selector (ct-emul! ct-grid-ratio) (ct-exp!) (ct-emul! ct-anchors) (ct-sqrt!))
    ;;conf-vals
    (->> conf pred-selector ct-sigmoid!)
    ;;prob-vals
    (-> (pred-selector class-dims)
        (ct-reshape [(* grid-x grid-y anchor-count) classes-count])
        ct-softmax!)
    pred))


(defn ->gradient
  "Calculates the gradient from a formatted prediction and a weighted error
  (weighted error is of the form 'weights*(formatted-pred - truth)'"
  [formatted-prediction weighted-error]
  (let [formatted-pred-selector (partial m/select formatted-prediction :all :all :all)
        we-selector   (partial m/select weighted-error :all :all :all)
        x-y-grad      (->> x-y formatted-pred-selector (m/emap sigmoid-gradient))
        w-h-grad      (->> w-h formatted-pred-selector (m/emul 0.5 grid-ratio))
        conf-grad     (->> conf   formatted-pred-selector (m/emap sigmoid-gradient))
        prob-grad     (-> (formatted-pred-selector class-dims)
                          (m/reshape [(* grid-x grid-y anchor-count) classes-count])
                          ((fn [v] (map softmax-gradient v))))
        prob-vals     (-> (we-selector class-dims)
                          (m/reshape [(* grid-x grid-y anchor-count) classes-count]))]
    (m/emul 2
            (m/join-along 3
                          (m/emul (we-selector x-y) x-y-grad)
                          (m/emul (we-selector w-h) w-h-grad)
                          (m/emul (we-selector conf) conf-grad)
                          (-> (map m/inner-product prob-grad prob-vals)
                              (m/reshape [grid-x grid-y anchor-count classes-count]))))))


(defn ct-sigmoid-gradient!
  [input-gradient output loss-gradient]
  (ct/activation-gradient! input-gradient loss-gradient output :logistic))


(defn ct-wh-grad!
  [input-gradient output loss-gradient]
  (let [grid-ratio (ct/->tensor grid-ratio)]
    ;;x = (a*x * b*y)
    (ct/binary-op! input-gradient 1.0 output 0.5 grid-ratio :*)
    (ct/binary-op! input-gradient 1.0 input-gradient 1.0 loss-gradient :*)))


(defn ct-softmax-grad!
  [input-gradient output loss-gradient]
  (let [output (ct/as-2d-matrix output)
        input-gradient (ct/as-2d-matrix input-gradient)
        loss-gradient (ct/as-2d-matrix loss-gradient)
        [n-rows n-cols] (m/shape output)
        output-emul-loss (ct/new-tensor (m/shape output))
        output-dot-loss (ct/new-tensor [n-rows 1])]
    ;;(v emul loss) - output emul (output dot loss)
    (ct/binary-op! output-emul-loss 1.0 output 1.0 loss-gradient :*)
    (ct/unary-reduce! output-dot-loss 1.0 output-emul-loss :sum)
    (ct/binary-op! input-gradient 1.0 output 1.0 output-dot-loss :*)
    (ct/binary-op! input-gradient 1.0 output-emul-loss 1.0 input-gradient :-)))


(defn ->gradient-ct!
  "Calculates the gradient from a formatted prediction and a weighted error
  (weighted error is of the form 'weights*(formatted-pred - truth)'"
  [formatted-prediction loss-gradient]
  (let [input-gradient (ct/new-tensor (m/shape formatted-prediction))
        formatted-pred-selector (partial ct/select formatted-prediction :all :all :all)
        lg-selector   (partial ct/select loss-gradient :all :all :all)
        ig-selector   (partial ct/select input-gradient :all :all :all)]
    ;;x-y-grad
    (ct-sigmoid-gradient! (ig-selector x-y)
                          (formatted-pred-selector x-y)
                          (lg-selector x-y))
    ;;w-h-grad
    (ct-wh-grad! (ig-selector w-h)
                 (formatted-pred-selector w-h)
                 (lg-selector w-h))
    ;;conf-grad
    (ct-sigmoid-gradient! (ig-selector conf)
                          (formatted-pred-selector conf)
                          (lg-selector conf))
    ;;prob-grad
    (ct-softmax-grad! (ig-selector class-dims)
                      (formatted-pred-selector class-dims)
                      (lg-selector class-dims))
    (ct-emul! 2.0 input-gradient)))


(defn custom-loss-gradient
  "Note that prediction has a different format than truth."
  [pred truth]
  (let [formatted-prediction (format-prediction pred)
        {:keys [weights ious one-hots]} (->weights formatted-prediction truth)
        weighted-error       (->> (m/sub formatted-prediction truth) (m/emul weights))
        gradient             (->gradient formatted-prediction weighted-error)]
    {:pred formatted-prediction
     :truth truth
     :weights weights
     :ious ious
     :one-hots one-hots
     :loss-gradient weighted-error
     :gradient gradient}))


(defn ct-custom-loss-gradient
  [pred truth]
  (let [truth           (ct/->tensor (m/array truth))
        pred            (ct/->tensor (m/array pred))
        pred            (format-prediction-ct! pred)
        {:keys [weights ious one-hots]} (->weights-ct pred truth)
        loss-gradient   (-> (ct/new-tensor (m/shape pred))
                            (ct/binary-op! 1.0 pred 1.0 truth :-)
                            ;;Produce v-hat with 2 multiplication
                            (#(ct/binary-op! % 1.0 % 1.0 weights :*)))]
    {:pred (m/assign! (ct/new-tensor (m/shape pred)) pred)
     :truth truth
     :ious ious
     :one-hots one-hots
     :weights weights
     :loss-gradient loss-gradient
     :gradient (->gradient-ct! pred loss-gradient)}))


(defmethod loss/loss :yolo9000
  [loss-term buffer-map]
  (cpu-tm/tensor-context
   (let [truth (get buffer-map :labels)
         prediction (get buffer-map :output)
         formatted-prediction (format-prediction prediction)
         ct-truth (ct/->tensor (m/array truth))
         ct-formatted-prediction (format-prediction-ct! (ct/->tensor prediction))
         weights (:weights (->weights formatted-prediction truth))
         weights-ct (:weights (->weights-ct ct-formatted-prediction ct-truth))
         corem-loss (->> (m/sub formatted-prediction truth)
                         (m/square)
                         (m/emul weights)
                         (m/esum))
         ct-loss (-> (ct/binary-op! ct-formatted-prediction 1.0 ct-formatted-prediction 1.0 ct-truth :-)
                     (ct/binary-op! 1.0 ct-formatted-prediction 1.0 ct-formatted-prediction :*)
                     (ct/binary-op! 1.0 ct-formatted-prediction 1.0 weights-ct :*)
                     ct/to-double-array
                     m/esum)]
     {:corem-loss corem-loss
      :ct-loss ct-loss})))



(defn- read-label
  []
  (let [data (util/read-nippy-file "yololoss.nippy")]
    (m/reshape (get data :data) (get data :shape))))


(defn- compare-large-vectors
  [corem-vec ct-vec]
  (let [correct (m/to-double-array corem-vec)
        guess (ct/to-double-array ct-vec)]
    (clojure.pprint/pprint
     [:equals (m/equals correct guess 1e-4)
      :results
      (->> (map vector
                (range)
                correct guess)
           (remove (fn [[idx correct guess]]
                     (= (double correct)
                        (double guess))))
           (take output-count))])))


(defn- test-loss
  []
  (cpu-tm/tensor-context
   (let [pred (m-rand/sample-uniform [grid-x grid-y anchor-count output-count])
         truth (read-label)
         loss (loss/loss {:type :yolo9000} {:labels truth
                                            :output pred})]
     loss)))


(defn- test-gradient
  []
  (gpu-tm/tensor-context
   (let [pred (m-rand/sample-uniform [grid-x grid-y anchor-count output-count])
         truth (read-label)
         corem-grad (custom-loss-gradient pred truth)
         ct-grad (ct-custom-loss-gradient pred truth)]
     (compare-large-vectors (:gradient corem-grad)
                            (:gradient ct-grad)))))
