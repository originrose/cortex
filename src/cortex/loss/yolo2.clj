(ns cortex.loss.yolo2
  (:require [clojure.core.matrix :as m]
            [cortex.loss.core :as loss]
            [cortex.util :as util]
            [clojure.core.matrix.random :as m-rand]
            [cortex.tensor :as ct]
            [cortex.compute.cpu.tensor-math :as cpu-tm]
            [clojure.math.combinatorics :as combo]
            [cortex.compute.driver :as drv]
            [cortex.tensor.allocator :as alloc]
            [mikera.vectorz.matrix-api]
            [cortex.loss.util :as loss-util]
            [cortex.compute.nn.backend :as nn-backend]
            [cortex.compute.math :as math]
            [think.datatype.core :as dtype]
            [cortex.graph :as graph]
;;            [cortex.compute.cuda.tensor-math :as gpu-tm]
            ))


(def ^:dynamic *grid-x* 13)
(def ^:dynamic *grid-y* 13)
(def ^:dynamic *scale-noob* 0.5)
(def ^:dynamic *scale-conf* 5)
(def ^:dynamic *scale-coord* 5)
(def ^:dynamic *scale-prob* 1)
(defonce default-anchors (partition 2 [1.08 1.19 3.42 4.41 6.63 11.38 9.42 5.11 16.62 10.52]))
(def ^:dynamic *anchors* default-anchors)

(defn anchor-count
  ^long []
  (long (first (m/shape *anchors*))))


(defmacro with-variables
  [grid-x grid-y scale-noob scale-conf scale-coord scale-prob anchors & body]
  `(with-bindings {#'*grid-x* (long ~grid-x)
                   #'*grid-y* (long ~grid-y)
                   #'*scale-noob* (double ~scale-noob)
                   #'*scale-conf* (double ~scale-conf)
                   #'*scale-prob* (double ~scale-prob)
                   #'*anchors* ~anchors}
     ~@body))


(defn grid-ratio
  []
  [(/ (long *grid-x*)) (/ (long *grid-y*))])


(defn output-count
  [prediction]
  (long (last (m/shape prediction))))


(defn classes-count
  [prediction]
  (- (output-count prediction) 5))


(defn class-dims
  [prediction]
  (range 5 (output-count prediction)))


(def x-y [0 1])
(def w-h [2 3])
(def bb [0 1 2 3])
(def conf [4])


(defn realize-label
  "Adds the label keyword to a datum.
  A 'truth' value to be passed into the loss function is of the form '(:label (realize-label datum))'
  A label is of the form (x y w h conf p0 ... p20), where x, y, w, h, conf are in the range 0-1, and p's form the class probability vector"
  [d {:keys [grid-x grid-y anchor-count class-count class-name->label]}]
  (let [output-count (+ class-count 5)
        label (m/mutable (m/zero-array [grid-x grid-y anchor-count output-count]))]
    (doseq [ob (:objects d)]
      (let [[w h bb] [(:width d) (:height d) (:bounding-box ob)]
            scale   [(/ grid-x w) (/ grid-y h) (/ grid-x w) (/ grid-y h)]
            [xm ym xM yM] (m/mul scale bb)
            [x-ave y-ave] [(/ (+ xm xM) 2) (/ (+ ym yM) 2)]
            [cell-x cell-y] [(int x-ave) (int y-ave)]
            [x y w h]  [(- x-ave cell-x) (- y-ave cell-y) (/ (- xM xm) grid-x) (/ (- yM ym) grid-y)]
            output (m/array (concat [x y w h] [1] (class-name->label (:class ob))))]
        (m/set-selection! label cell-x cell-y :all :all output)))
    (assoc d :label label)))




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


;;Core-m implementation for reference

(defn sigmoid [x] (/ 1 (+ 1 (Math/exp (- x)))))

(defn un-sigmoid [x] (-> (/ 1 x) (- 1) (Math/log) (-)))

(defn sigmoid-gradient [x] (* x (- 1 x)))

(defn softmax [v]
  (let [exp-v (m/exp (m/sub v (apply max v)))]
    (m/div exp-v (m/esum exp-v))))

(defn softmax-gradient
  "Softmax gradient is a matrix whose ij-entry is v_jdelta_ij - v_jv_i.
  Written globally that is Diagonal(v) minus v-tensor-v."
  [v]
  (m/sub (m/diagonal-matrix v) (m/outer-product v v)))

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


(defn to-vec
  [item]
  (mapv #(m/mget item %) (range (m/ecount item))))


(defn iou
  "Calculates intersection over union of boxes b = [x y w h].
  **Be sure that x,y and w,h are using the same distance units.**"
  [b1 b2]
  (let [UL (m/emap max (ul (to-vec b1)) (ul (to-vec b2)))
        BR (m/emap min (br (to-vec b1)) (br (to-vec b2)))
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
      (m/zero-vector (anchor-count)))))


(defn ->weights [formatted-prediction truth]
  (let [formatted-pred-selector (partial m/select formatted-prediction :all :all :all)
        output-count (output-count formatted-prediction)
        classes-count (classes-count formatted-prediction)
        truth-selector (partial m/select truth :all :all :all)
        pred-boxes (-> (formatted-pred-selector bb)
                       (m/reshape [(* *grid-x* *grid-y* (anchor-count)) (count bb)]))
        truth-boxes (-> (truth-selector bb)
                        (m/reshape [(* *grid-x* *grid-y* (anchor-count)) (count bb)]))
        ious (-> (map iou pred-boxes truth-boxes)
                 ;; value at (+ (* i grid-x) j) is the vector of ious for each of the five anchors at cell (i,j)
                 (m/reshape [(* *grid-x* *grid-y*) (anchor-count)]))

        one-hots (map select-anchor ious)
        D (-> one-hots
              ;; value at [i j anchor-idx] is 1 if anchor-idx achieves max for cell (i,j), otherwise 0.
              (m/reshape [*grid-x* *grid-y* (anchor-count)])
              ;; tensor on 1's to get shape [grid-x grid-y anchor-count output-count]
              (m/outer-product (m/assign (m/new-vector output-count) 1.0)))
        scale-vec-ob (->> (m/join
                            (m/assign (m/zero-vector (count bb)) *scale-coord*)
                            (m/assign (m/zero-vector (count conf)) *scale-conf*)
                            (m/assign (m/zero-vector classes-count) *scale-prob*))
                          (m/broadcast-like formatted-prediction))
        scale-vec-noob (->> (m/join
                              (m/zero-vector (count bb))
                              (m/assign (m/zero-vector (count conf)) *scale-noob*)
                              (m/zero-vector classes-count))
                            (m/broadcast-like formatted-prediction))
        D     (m/add
               (m/emul D scale-vec-ob)
               (m/emul (m/sub 1 D) scale-vec-noob))]
    {:ious ious
     :one-hots one-hots
     :weights D}))


(defn format-prediction
  [pred]
  (let [pred-selector (partial m/select pred :all :all :all)
        x-y-vals      (->> x-y pred-selector (m/emap sigmoid))
        w-h-vals      (->> w-h pred-selector (m/exp) (m/emul (grid-ratio)) (m/emul *anchors*) (m/sqrt))
        conf-vals     (->> conf pred-selector (m/emap sigmoid))
        prob-vals     (-> (pred-selector (class-dims pred))
                          (m/reshape [(* *grid-x* *grid-y* (anchor-count)) (classes-count pred)])
                          ((fn [v] (map softmax v)))
                          (m/reshape [*grid-x* *grid-y* (anchor-count) (classes-count pred)]))]
    (m/join-along 3 x-y-vals w-h-vals conf-vals prob-vals)))


(defn ->gradient
  "Calculates the gradient from a formatted prediction and a weighted error
  (weighted error is of the form 'weights*(formatted-pred - truth)'"
  [formatted-prediction weighted-error]
  (let [class-dims (class-dims formatted-prediction)
        classes-count (classes-count formatted-prediction)
        formatted-pred-selector (partial m/select formatted-prediction :all :all :all)
        we-selector   (partial m/select weighted-error :all :all :all)
        x-y-grad      (->> x-y formatted-pred-selector (m/emap sigmoid-gradient))
        w-h-grad      (->> w-h formatted-pred-selector (m/emul 0.5))
        conf-grad     (->> conf   formatted-pred-selector (m/emap sigmoid-gradient))
        prob-grad     (-> (formatted-pred-selector class-dims)
                          (m/reshape [(* *grid-x* *grid-y* (anchor-count)) classes-count])
                          ((fn [v] (map softmax-gradient v))))
        prob-vals     (-> (we-selector class-dims)
                          (m/reshape [(* *grid-x* *grid-y* (anchor-count)) classes-count]))]
    (m/emul 2
            (m/join-along 3
                          (m/emul (we-selector x-y) x-y-grad)
                          (m/emul (we-selector w-h) w-h-grad)
                          (m/emul (we-selector conf) conf-grad)
                          (-> (map m/inner-product prob-grad prob-vals)
                              (m/reshape [*grid-x* *grid-y* (anchor-count) classes-count]))))))


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


(defn ct-coords!
  "Return x-y coords of upper left corner of box and lower right."
  [box-vec ul-vec br-vec]
  (let [wh-vec (ct/select box-vec :all (range 2 4))
        xy-vec (ct/select box-vec :all (range 2))]
    (ct/unary-op! br-vec 0.5 wh-vec :noop)
    (ct/binary-op! ul-vec 1.0 xy-vec 1.0 br-vec :-)
    (ct/binary-op! br-vec 1.0 xy-vec 1.0 br-vec :+)))


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
        b1-ul (alloc/new-uninitialized-tensor :b1-ul [num-box 2])
        b1-br (alloc/new-uninitialized-tensor :b1-br [num-box 2])
        b2-ul (alloc/new-uninitialized-tensor :b2-ul [num-box 2])
        b2-br (alloc/new-uninitialized-tensor :b2-br [num-box 2])
        overlap (alloc/new-uninitialized-tensor :overlap [num-box 2])
        intersection (alloc/new-uninitialized-tensor :intersection [num-box])
        area-b1 (alloc/new-uninitialized-tensor :area-b1 [num-box])
        area-b2 (alloc/new-uninitialized-tensor :area-b2 [num-box])
        union (alloc/new-uninitialized-tensor :union [num-box])
        result (alloc/new-uninitialized-tensor :result [num-box])
        _ (ct-coords! b1 b1-ul b1-br)
        _ (ct-coords! b2 b2-ul b2-br)
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



;; (defn test-iou
;;   []
;;   (gpu-tm/tensor-context
;;    (alloc/with-allocator (alloc/passthrough-allocator)
;;      (let [src-boxes [[10 10 20 20]
;;                       [5 5 5 5]
;;                       [10 10 5 5]
;;                       [17 17 2 2]]
;;            box-pairs (mapv (fn [[x y]]
;;                              [(get src-boxes x)
;;                               (get src-boxes y)])
;;                            (combo/combinations (range (count src-boxes)) 2))
;;            b1 (ct/->tensor (map first box-pairs))
;;            b2 (ct/->tensor (map second box-pairs))]
;;        (let [correct (double-array (mapv (partial apply iou) box-pairs))
;;              guess (ct/to-double-array (iou-ct b1 b2))]
;;          (println {:correct (vec correct)
;;                    :guess (vec guess)}))))))



(defn ct-one-hot-encode
  "Given the vector iou of iou-s with the different anchors, returns a one-hot encoded vector at the max value.
If there are two equal max values then you will get a two-hot encoded vector."
  [data-matrix]
  (let [[n-rows n-col] (m/shape data-matrix)
        max-data (alloc/new-uninitialized-tensor :max-data [n-rows 1])]

    (ct/unary-reduce! max-data 1.0 data-matrix :max)
    (ct/binary-op! data-matrix 1.0 data-matrix 1.0 max-data :eq)
    (ct/binary-op! max-data 1.0 max-data 0.0 0.0 :eq)
    (ct/binary-op! max-data 1.0 1.0 1.0 max-data :-)
    (ct/binary-op! data-matrix 1.0 data-matrix 1.0 max-data :*)))


;; (defn test-one-hot-encode
;;   []
;;   (gpu-tm/tensor-context
;;    (alloc/with-allocator (alloc/passthrough-allocator)
;;     (->> (ct/->tensor [[0 0 0 0]
;;                        [0 0 0 0]
;;                        [0 0 2 0]
;;                        [1 4 2 2]])
;;          ct-one-hot-encode
;;          ct/to-double-array
;;          vec))))


(defn apply-selector
  [item shp]
  (let [shp-count (count (m/shape item))]
    (-> (apply ct/select item (concat (repeat (- shp-count 1) :all)
                                      [shp]))
        ct/as-2d-matrix)))


(defn make-selector
  [item]
  (fn [shp]
    (apply-selector item shp)))


(defn ->weights-ct
  [formatted-prediction truth]
  (let [formatted-pred-selector (make-selector formatted-prediction)
        pred-shape (m/shape formatted-prediction)
        truth-selector (make-selector truth)
        pred-boxes (formatted-pred-selector bb)
        truth-boxes (truth-selector bb)
        num-ious (first (m/shape pred-boxes))
        ious (-> (iou-ct pred-boxes truth-boxes)
                 ;; value at (+ (* i grid-x) j) is the vector of ious for each of the five anchors at cell (i,j)
                 (ct-reshape [(quot (long num-ious) (anchor-count)) (anchor-count)]))


        all-d (-> (alloc/new-uninitialized-tensor :all-d (m/shape formatted-prediction))
                  ct/as-2d-matrix)
        one-hots (-> (ct-one-hot-encode ious)
                     ;;Flatten out so broadcasting will apply correctly
                     (ct/in-place-reshape [num-ious 1]))

        D (ct/assign! all-d one-hots)
        one-minus-one-hots (ct/binary-op! one-hots 1.0 1 1.0 one-hots :-)
        one-minus-d (ct/assign! (-> (alloc/new-uninitialized-tensor :one-minus-d (m/shape formatted-prediction))
                                    ct/as-2d-matrix)
                                one-minus-one-hots)
        scale-vec-ob (-> (alloc/->const-tensor :scale-vec-ob
                                               (concat (repeat (count bb) *scale-coord*)
                                                       (repeat (count conf) *scale-conf*)
                                                       (repeat (classes-count formatted-prediction) *scale-prob*))))
        scale-vec-noob (-> (alloc/->const-tensor :scale-vec-noob
                                                 (concat (repeat (count bb) 0)
                                                         (repeat (count conf) *scale-noob*)
                                                         (repeat (classes-count formatted-prediction) 0))))]
    (ct/binary-op! D 1.0 D 1.0 scale-vec-ob :*)
    (ct/binary-op! one-minus-d 1.0 one-minus-d 1.0 scale-vec-noob :*)
    (ct/binary-op! D 1.0 D 1.0 one-minus-d :+)
    {:ious ious
     :one-hots one-hots
     :weights (ct/in-place-reshape D (m/shape formatted-prediction))}))


(defn format-prediction-ct!
  [pred]
  (let [ct-grid-ratio (alloc/->const-tensor :ct-grid-ratio (grid-ratio))
        ct-anchors (alloc/->const-tensor :ct-anchors *anchors*)
        pred-selector (make-selector pred)]
    ;;x-y-vals
    (->> x-y pred-selector ct-sigmoid!)
    ;;w-h-vals
    (->> w-h pred-selector (ct-exp!) (ct-emul! ct-grid-ratio) (ct-emul! ct-anchors) (ct-sqrt!))
    ;;conf-vals
    (->> conf pred-selector ct-sigmoid!)
    ;;prob-vals
    (->> (pred-selector (class-dims pred)) ct-softmax!)
    pred))


(defn ct-sigmoid-gradient!
  [input-gradient output loss-gradient]
  (ct/activation-gradient! input-gradient loss-gradient output :logistic))


(defn ct-wh-grad!
  [input-gradient output loss-gradient]
  (ct/binary-op! input-gradient 1.0 output 0.5 loss-gradient :*))


(defn- ct-softmax-grad!
  [input-gradient output loss-gradient]
  (let [output (ct/as-2d-matrix output)
        input-gradient (ct/as-2d-matrix input-gradient)
        loss-gradient (ct/as-2d-matrix loss-gradient)
        [n-rows n-cols] (m/shape output)
        output-emul-loss (alloc/new-uninitialized-tensor :output-emul-loss (m/shape output))
        output-dot-loss (alloc/new-uninitialized-tensor :output-dot-loss [n-rows 1])]
    ;;(v emul loss) - output emul (output dot loss)
    (ct/binary-op! output-emul-loss 1.0 output 1.0 loss-gradient :*)
    (ct/unary-reduce! output-dot-loss 1.0 output-emul-loss :sum)
    (ct/binary-op! input-gradient 1.0 output 1.0 output-dot-loss :*)
    (ct/binary-op! input-gradient 1.0 output-emul-loss 1.0 input-gradient :-)))


(defn- ->gradient-ct!
  "Calculates the gradient from a formatted prediction and a weighted error
  (weighted error is of the form 'weights*(formatted-pred - truth)'"
  [input-gradient formatted-prediction loss-gradient]
  (let [formatted-pred-selector (make-selector formatted-prediction)
        lg-selector   (make-selector loss-gradient)
        ig-selector   (make-selector input-gradient)
        c-dims (class-dims formatted-prediction)]
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
    (ct-softmax-grad! (ig-selector c-dims)
                      (formatted-pred-selector c-dims)
                      (lg-selector c-dims))
    (ct-emul! 2.0 input-gradient)))


(defn- ct-custom-loss-gradient!
  [input-gradient pred truth]
  (let [pred            (format-prediction-ct! pred)
        {:keys [weights ious one-hots]} (->weights-ct pred truth)
        loss-gradient   (-> (alloc/new-uninitialized-tensor :loss-gradient (m/shape pred))
                            (ct/binary-op! 1.0 pred 1.0 truth :-)
                            ;;Produce v-hat with 2 multiplication
                            (#(ct/binary-op! % 1.0 % 1.0 weights :*)))]
    {:truth truth
     :ious ious
     :one-hots one-hots
     :weights weights
     :loss-gradient loss-gradient
     :gradient (->gradient-ct! input-gradient pred loss-gradient)}))


(defn- custom-loss
  [prediction truth]
  (let [formatted-prediction (format-prediction prediction)
        weights (:weights (->weights formatted-prediction truth))]
    {:pred formatted-prediction
     :weights weights
     :value
     (->> (m/sub formatted-prediction truth)
          (m/square)
          (m/emul weights)
          (m/esum))}))

(defn ct-clone
  [item]
  (ct/assign! (ct/new-tensor (m/shape item)) item))

(defn ct-sub!
  [lhs rhs]
  (ct/binary-op! lhs 1.0 lhs 1.0 rhs :-))

(defn ct-sqr!
  [lhs]
  (ct/binary-op! lhs 1.0 lhs 1.0 lhs :*))


(defn- ct-custom-loss
  [prediction truth]
  (alloc/with-allocator (alloc/passthrough-allocator)
    (let [output-count (/ (ct/ecount prediction)
                          (* (long *grid-x*)
                             (long *grid-y*)
                             (long (anchor-count))))
          data-shape [1 *grid-x* *grid-y* (anchor-count) output-count]
          prediction (-> (ct/->tensor [prediction])
                         (ct/in-place-reshape data-shape))
          ct-truth (-> (ct/->tensor [truth])
                       (ct/in-place-reshape data-shape))
          ct-formatted-prediction (format-prediction-ct! prediction)
          ct-weights (:weights (->weights-ct ct-formatted-prediction ct-truth))]
      {:pred ct-formatted-prediction
       :weights ct-weights
       :value
       (-> (ct-sub! ct-formatted-prediction ct-truth)
           (ct-sqr!)
           (ct-emul! ct-weights)
           (ct/to-double-array)
           m/esum)})))


(defmethod loss/loss :yolo2
  [loss-term buffer-map]
  (let [truth (get buffer-map :labels)
        prediction (get buffer-map :output)
        {:keys [grid-x grid-y
                scale-noob scale-conf scale-coord scale-prob anchors]} loss-term]
    (with-variables grid-x grid-y scale-noob scale-conf scale-coord scale-prob anchors
      (cpu-tm/tensor-context
       (:value (ct-custom-loss prediction truth))))))


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
      (->> (map vector (range) correct guess)
           (remove (fn [[idx correct guess]]
                     (= (double correct)
                        (double guess))))
           (take 20))])))


(defn- test-loss
  []
  (cpu-tm/tensor-context
   (let [truth (m/array (read-label))
         pred (m-rand/sample-uniform [*grid-x* *grid-y* (anchor-count) (output-count truth)])
         corem-data (custom-loss pred truth)
         ct-data (ct-custom-loss pred truth)]
     (println (:value corem-data) (:value ct-data)))))


(defn- repeatv
  [batch-size item]
  (-> (repeat batch-size item)
      vec))


;; (defn- test-gradient
;;   []
;;   (gpu-tm/tensor-context
;;    (alloc/with-allocator (alloc/atom-allocator)
;;      (let [batch-size 10
;;            truth (repeatv 10 (read-label))
;;            pred (repeatv 10 (m-rand/sample-uniform [*grid-x* *grid-y* (anchor-count) (output-count truth)]))
;;            corem-grad (custom-loss-gradient (first pred) (first truth))
;;            input-gradient (ct/new-tensor (m/shape pred))
;;            truth-tens    (ct/->tensor (m/array truth))
;;            ct-grad (ct-custom-loss-gradient! input-gradient
;;                                              (ct/->tensor pred)
;;                                              truth-tens)
;;            _ (m/assign! input-gradient 0)
;;            ct-grad (ct-custom-loss-gradient! input-gradient
;;                                              (ct/->tensor pred)
;;                                              truth-tens)]
;;        (compare-large-vectors (:gradient corem-grad)
;;                               (ct/select (:gradient ct-grad) 1 :all :all :all :all))))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Compute implementation
(defrecord Yolo2Loss [loss-term backend allocator]
  loss-util/PComputeLoss
  (compute-loss-gradient [this buffer-map]
    (let [v (get-in buffer-map [:output :buffer])
          gradient (get-in buffer-map [:output :gradient])
          target (get-in buffer-map [:labels :buffer])
          stream (nn-backend/get-stream)
          [batch-size output-size] (math/batch-shape v)
          {:keys [grid-x grid-y
                  scale-noob scale-conf scale-coord scale-prob
                  anchors]} loss-term]
      (with-variables grid-x grid-y scale-noob scale-conf scale-coord scale-prob anchors
       (ct/with-stream stream
         (ct/with-datatype (dtype/get-datatype v)
           (alloc/with-allocator allocator
             (let [output-count (/ (m/ecount v)
                                   (* (long batch-size)
                                      *grid-x*
                                      *grid-y*
                                      (anchor-count)))
                   data-shape [batch-size *grid-x* *grid-y* (anchor-count) output-count]
                   ->tensor #(-> (math/array->cortex-tensor %)
                                 (ct/in-place-reshape data-shape))
                   pred (->tensor v)
                   input-gradient (->tensor gradient)
                   truth (->tensor target)]
               (ct-custom-loss-gradient! input-gradient pred truth)))))))))


(defmethod loss-util/create-compute-loss-term :yolo2
  [backend network loss-term batch-size]
  (->Yolo2Loss loss-term backend (alloc/atom-allocator)))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Graph implementation
(defmethod graph/get-node-metadata :yolo2
  [loss-term]
  {:arguments {:output {:gradients? true}
               :labels {}}
   :passes [:loss]})

(defn yolo2-loss
  [{:keys [grid-x grid-y scale-noob scale-conf scale-coord scale-prob anchors]
    :or {grid-x 13 grid-y 13 scale-noob 0.5 scale-conf 5 scale-coord 5 scale-prob 1
         anchors (partition 2 default-anchors)}
    :as arg-map}]
  (merge {:type :yolo2
          :grid-x grid-x
          :grid-y grid-y
          :scale-noob scale-noob
          :scale-conf scale-conf
          :scale-coord scale-coord
          :scale-prob scale-prob
          :anchors anchors}
         arg-map))


(defmethod loss/generate-loss-term :yolo2
  [item-key]
  (assoc (yolo2-loss {}) :lambda (loss/get-loss-lambda {:type :yolo2})))


(defmethod graph/generate-stream-definitions :yolo2
  [graph loss-term]
  (loss-util/generate-loss-term-stream-definitions graph loss-term))


;; Code to handle the predictions once they are made

(defn prediction->boxes
  [pred prob-threshold]
  (for [col (range *grid-y*)
        row (range *grid-x*)
        b (range (anchor-count))]
    (let [[x y w h c] (take 5 (m/select pred col row b :all))
          box-x (/ (+ col (sigmoid x)) *grid-x*)
          box-y (/ (+ row (sigmoid y)) *grid-y*)
          [box-w box-h] (->> [w h] (m/exp)
                             (m/emul (get *anchors* b))
                             (m/emul (grid-ratio))
                             (m/sqrt))
          box-c (sigmoid c)
          class-probs (-> (drop 5 (m/select pred row col b :all))
                          softmax)
          ;; replace all probs < threshold with 0's
          box-probs (->> (m/emul box-c class-probs)
                         (mapv #(if (> % prob-threshold) % 0.0)))]
      (concat [box-x box-y box-w box-h] box-probs [{:box-probability box-c
                                                    :class-probabilities class-probs}]))))


(defn- center-and-size->corners [[x y w h]]
  [(- x (/ w 2)) (- y (/ h 2)) (+ x (/ w 2)) (+ y (/ h 2))])


;; potential problem here: actual is 3, prediction A is part 3 part 7.
;; prediction B is more confident it is a 3 and has high iou thresh.
;; result is that A has prob(3) zeroed out, but does not remove the box completely.
;; So after NMS prediction A gets a label of 7 even though it was more confident in 3.
;; This problem is more specific to objects which are guaranteed to be non-overlapping.
;; A solution could be to not zero out its prob, but instead just remove it from the
;; list of possible 3's all together.
(defn yolo-nms
  "Take a yolo network output and perform a specialized non-maximal-suppression that takes
  into account the class probabilities *and* the bounding box probability."
  [prediction grid-x grid-y anchors
   prob-threshold iou-threshold
   classes-vec]
  (when (or (nil? grid-x)
            (nil? grid-y)
            (nil? anchors))
    (throw (ex-info "The same grid-x, grid-y and anchors must be used as were used to train the network"
                    {:grid-x grid-x
                     :grid-y grid-y
                     :anchors anchors})))
  (with-bindings {#'*grid-x* grid-x
                  #'*grid-y* grid-y
                  #'*anchors* anchors}
    (let [pred (m/reshape prediction [*grid-x* *grid-y* (anchor-count) (+ 5 (count classes-vec))])
          ;; boxes returns a matrix, for which each row is an x-y-w-h + class prob vector,
          ;; in which x-y-w-h are all scaled to [0,1] and the class prob vector is
          ;; the product of obj conf and cond prob and has small values zeroed out.
          ;; to check if that is happening we can print out the before and after with label.
          boxes (prediction->boxes pred prob-threshold)
          NMS-boxes (reduce
                     (fn [boxes class-index]
                       (let [sorted-boxes (sort-by #(nth % class-index) > boxes)]
                         ;; perform  NMS for this particular class
                         (reduce
                           (fn [boxes i]
                             (let [current-item (nth boxes i)
                                   current-box (take 4 current-item)
                                   current-prob (nth current-item class-index)]
                               (if (= 0.0 current-prob)
                                 boxes
                                 ;; suppress boxes if lower prob confidence (i.e. ranked lower in list) AND high iou
                                 (concat (take (+ i 1) boxes)
                                         (map (fn [compare-item]
                                                (let [compare-box (take 4 compare-item)
                                                      compare-prob (nth compare-item class-index)]
                                                  ;; my addition: if compare-item has prob of 0, don't even calculate iou
                                                  ;; (println " Compare prob: " compare-prob)
                                                  (if (= 0.0 compare-prob)
                                                    compare-item
                                                    ;; else, if high iou with current, set prob to 0
                                                    (if (>= (iou current-box compare-box) iou-threshold)
                                                      (assoc (vec compare-item) class-index 0.0)
                                                      compare-item))))
                                              (drop (+ i 1) boxes))))))
                           sorted-boxes (range (count sorted-boxes)))))
                     boxes
                     (range 4 (+ 4 (count classes-vec))))]

      (->> NMS-boxes
           (filter #(< prob-threshold (apply max (drop 4 (drop-last %)))))
           (map (fn [data]
                  (let [probs (drop 4 (drop-last data))
                        prob-max-index (util/max-index probs)]
                   (merge (last data)
                          {:bounding-box (center-and-size->corners data)
                           :class (get classes-vec prob-max-index)
                           :class-probability (nth probs prob-max-index)}))))))))


(defn- prediction->objects
  "Convert yolo net output to human-readable form.  More precisely, returns:
  1. bounding-box-xywh = bounding box in x-y-w-h coords,
     on a scale such that w = 1 corresponds to the network image input width and same for height.
  2. bounding-box = bounding box on same scale as above, but in upper left corner - lower right corner coords.
  3. class = the string class prediction for the object.
  4. class-probability = (confidence there is an object at all) * (probability such an object has the given class)
  Note: w-h (and also upper left, lower right corner coords) may be outside of the range [0,1].
        This means the network is predicting a bounding box which does not fit within the image,
        which could be the correct answer if it is predicting a box for an object which is halfway out of the picture."
  [prediction classes-vec]
  (let [pred (m/reshape prediction [*grid-x* *grid-y* (anchor-count) (+ 5 (count classes-vec))])]
    (for [col (range *grid-y*)
         row (range *grid-x*)
         b   (range (anchor-count))]
     (let [[x y w h c] (take 5 (m/select pred col row b :all))
           box-x          (/ (+ col (sigmoid x)) *grid-x*)
           box-y          (/ (+ row (sigmoid y)) *grid-y*)
           [box-w box-h] (->> [w h] (m/exp) (m/emul (get *anchors* b)) (m/emul (grid-ratio)) (m/sqrt))
           box-confidence (sigmoid c)
           class-probs    (-> (drop 5 (m/select pred row col b :all)) (softmax))
           class-index    (util/max-index class-probs)]
       {:bounding-box-xywh [box-x box-y box-w box-h]
        :bounding-box (center-and-size->corners [box-x box-y box-w box-h])
        :class             (nth classes-vec class-index)
        :class-probability (* box-confidence (nth class-probs class-index))}))))

(defn nms-filter
  "Remove each object that overlaps with another object and has lower class probability."
  [boxes iou-threshold]
  (reduce (fn [sorted-boxes i]
            (concat (take (inc i) sorted-boxes)
                    (filter #(> iou-threshold (iou (:bounding-box-xywh (nth sorted-boxes i)) (:bounding-box-xywh %)))
                            (drop (inc i) sorted-boxes))))
          (sort-by :class-probability > boxes) (range (count boxes))))

(defn yolo-non-overlapping-nms
  "NMS for image sets in which items from different classes are not expected to overlap
  (for example a car and truck would not have high iou in overhead imagery).  "
  [prediction grid-x grid-y anchors prob-threshold iou-threshold classes-vec]
  (with-bindings {#'*grid-x* grid-x
                  #'*grid-y* grid-y
                  #'*anchors* anchors}
    (let [boxes     (->> (prediction->objects prediction classes-vec)
                         (filter #(< prob-threshold (:class-probability %))))
          NMS-boxes (nms-filter boxes iou-threshold)]
      {:certainty-threshold prob-threshold
       :iou-suppression-threshold iou-threshold
       :all-boxes        boxes
       :suppressed-boxes NMS-boxes})))

(defn yolo-overlapping-nms
  "Take a yolo network output and perform a non-maximal-suppression that takes
  into account the class probabilities *and* the bounding box probability.  Mopre precisely,
  an object that overlaps with another object of the *same class* and has lower class probability is removed."
  [prediction grid-x grid-y anchors prob-threshold iou-threshold classes-vec]
  (with-bindings {#'*grid-x* grid-x
                  #'*grid-y* grid-y
                  #'*anchors* anchors}
    (let [boxes     (->> (prediction->objects prediction classes-vec)
                         (filter #(< prob-threshold (:class-probability %))))
          NMS-boxes-overlapping (->> (group-by :class boxes)
                                     (vals)
                                     (map #(nms-filter % iou-threshold))
                                     (apply concat))]

      {:certainty-threshold prob-threshold
       :iou-suppression-threshold iou-threshold
       :all-boxes        boxes
       :suppressed-boxes NMS-boxes-overlapping})))