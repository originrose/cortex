(ns neurosis.core
  (:use [nuroko.lab core charts])
  (:use [nuroko.gui visual])
  (:use [clojure.core.matrix])
  (:require [task.core :as task])
  (:require [mikera.cljutils.error :refer [error]])
  (:require [mikera.vectorz.core])
  ;; (:require [nuroko.data mnist])
  (:import [mikera.vectorz Op Ops])
  (:import [mikera.vectorz.ops ScaledLogistic Logistic Tanh])
  (:import [nuroko.coders CharCoder])
  (:import [mikera.vectorz AVector Vectorz]))

;; (ns nuroko.demo.conj)

;; some utility functions
(defn feature-img [vector]
  ((image-generator :width 8 :height 28 :colour-function weight-colour-mono) vector))

(defn demo []

  ;; ============================================================
  ;; SCRABBLE score task

  (defn ^nuroko.coders.FixedStringCoder string-coder
    ([& {:keys [length]
         :or {}
         :as options}]
     (if length
       (nuroko.coders.FixedStringCoder. length)
       (error "Invaid input: " options))))

  (defn get-all-words-shuffled [file]
    (shuffle (take 3000 (filter #(< (count %) 20)
                                (clojure.string/split (slurp file) #"[\n\r]+")))))

  (def all-english-words (get-all-words-shuffled "resources/english.txt"))
  ;;TODO refactor the following functions - DRY
  (def training-english-words (zipmap (take (/ (count all-english-words) 2) all-english-words) (repeat 0)))
  (def test-english-words (zipmap (drop (/ (count all-english-words) 2) all-english-words) (repeat 0)))

  (def all-spanish-words (get-all-words-shuffled "resources/spanish.txt"))
  (def training-spanish-words (zipmap (take (/ (count all-spanish-words) 2) all-spanish-words) (repeat 1)))
  (def test-spanish-words (zipmap (drop (/ (count all-spanish-words) 2) all-spanish-words) (repeat 1)))


  (def scores
    (merge training-english-words training-spanish-words))


  (first scores)

  (def test-scores
    (merge test-english-words test-spanish-words))

  (def score-coder (int-coder :bits 1))
  (encode score-coder 1)
  (decode score-coder (encode score-coder 0))

  (def word-coder (string-coder :length 20))

  (def task
    (mapping-task scores
                  :input-coder word-coder
                  :output-coder score-coder))

  (def net
    (neural-network :inputs 160
                    :outputs 1
                    :hidden-op Ops/LOGISTIC
                    :output-op Ops/LOGISTIC
                    :hidden-sizes [400]))

  ;; (show (network-graph net :line-width 2)
  ;;       :title "Neural Net : Scrabble")

  (defn language-score [net word]
    (->> word
         (encode word-coder)
         (think net)
         (decode score-coder)))

  ;; evaluation function
  (defn evaluate-scores [net]
    (let [net (.clone net)
          words (keys test-scores)]
      (count (filter #(not (nil? %)) (for [w words]
                                       (if (= (language-score net w) (test-scores w))
                                         w
                                         (do
                                           (print w)
                                           nil)))))))

  (show (time-chart
          [#(evaluate-scores net)]
          :y-min (/ (+ (count all-english-words) (count all-spanish-words)) 4)
          :y-max (/ (+ (count all-english-words) (count all-spanish-words)) 2))
        :title "Correct letters")

  (evaluate-scores net)
  (+ (count test-spanish-words) (count test-english-words))
  ;; training algorithm
  (def trainer (supervised-trainer net task :batch-size 100))

  (task/run
    ;    {:sleep 1 :repeat 1000} ;; sleep used to slow it down, otherwise trains instantly.....
    {:repeat 1000000}
    (trainer net))

  ;; Test it out with some spanish and english words.
  ;; Should return 0 for English words and 1 for Spanish
  (language-score net "mujer")
  (language-score net "enjoy")
  (language-score net "revel")

  ;; end of SCRABBLE DEMO



  ;; ============================================================
;; MNIST digit recognistion task

;; ;; training data - 60,000 cases
;; (def data @nuroko.data.mnist/data-store)
;; (def labels @nuroko.data.mnist/label-store)
;; (def INNER_SIZE 300)

;; (count data)

;; ;; some visualisation
;; ;; image display function


;; (show (map img (take 100 data))
;;       :title "First 100 digits")

;; ;; we also have some labels
;; (count labels)
;; (take 10 labels)

;; ;; ok so let's compress these images

;; (def compress-task (identity-task data))

;; (def compressor
;;         (stack
;;     ;;(offset :length 784 :delta -0.5)
;;     (neural-network :inputs 784
;;                           :outputs INNER_SIZE
;;                     :layers 1
;;                    ;; :max-weight-length 4.0
;;                     :output-op Ops/LOGISTIC
;;                    ;; :dropout 0.5
;;                     )
;;     (sparsifier :length INNER_SIZE)))

;; (def decompressor
;;         (stack
;;     (offset :length INNER_SIZE :delta -0.5)
;;     (neural-network :inputs INNER_SIZE
;;                           :outputs 784
;;                    ;; :max-weight-length 4.0
;;                     :output-op Ops/LOGISTIC
;;                     :layers 1)))

;; (def reconstructor
;;   (connect compressor decompressor))

;; (defn show-reconstructions []
;;   (let [reconstructor (.clone reconstructor)]
;;     (show
;;       (->> (take 100 data)
;;         (map (partial think reconstructor))
;;         (map img))
;;       :title "100 digits reconstructed")))
;; (show-reconstructions)

;; (def trainer (supervised-trainer reconstructor compress-task))

;;       (task/run
;;   {:sleep 1 :repeat true}
;;   (do (trainer reconstructor) (show-reconstructions)))

;; (task/stop-all)

;; ;; look at feature maps for 150 hidden units
;; (show (map feature-img (feature-maps compressor :scale 2)) :title "Feature maps")


;; ;; now for the digit recognition
;; (def num-coder (class-coder
;;                  :values (range 10)))
;;       (encode num-coder 3)

;;       (def recognition-task
;;         (mapping-task
;;     (apply hash-map
;;            (interleave data labels))
;;           :output-coder num-coder))

;; (def recogniser
;;   (stack
;;     (offset :length INNER_SIZE :delta -0.5)
;;     (neural-network :inputs INNER_SIZE
;;                   :output-op Ops/LOGISTIC
;;                         :outputs 10
;;                   :layers 2)))

;; (def recognition-network
;;   (connect compressor recogniser))

;; (def trainer2 (supervised-trainer recognition-network
;;                                   recognition-task
;;                                   ;;:loss-function nuroko.module.loss.CrossEntropyLoss/INSTANCE
;;                                  ))

;; ;; test data and task - 10,000 cases
;; (def test-data @nuroko.data.mnist/test-data-store)
;; (def test-labels @nuroko.data.mnist/test-label-store)

;; (def recognition-test-task
;;         (mapping-task (apply hash-map
;;                       (interleave test-data test-labels))
;;                       :output-coder num-coder))

;; ;; show chart of training error (blue) and test error (red)
;; (show (time-chart [#(evaluate-classifier
;;                       recognition-task recognition-network )
;;                    #(evaluate-classifier
;;                       recognition-test-task recognition-network )]
;;                   :y-max 1.0)
;;       :title "Error rate")

;; (task/run
;;   {:sleep 1 :repeat true}
;;   (trainer2 recognition-network :learn-rate 0.1))
;;    ;; can tune learn-rate, lower => fine tuning => able to hit better overall accuracy

;; (task/stop-all)

;; (defn recognise [image-data]
;;   (->> image-data
;;     (think recognition-network)
;;     (decode num-coder)))

;; (recognise (data 0))

;; ;; show results, errors are starred
;; (show (map (fn [l r] (if (= l r) l (str r "*")))
;;            (take 100 labels)
;;            (map recognise (take 100 data)))
;;       :title "Recognition results")

;; (let [rnet (.clone recognition-network)]
;;   (reduce
;;   (fn [acc i] (if (= (test-labels i) (->> (test-data i) (think rnet) (decode num-coder)))
;;                 (inc acc) acc))
;;   0 (range (count test-data))))

;; (show (class-separation-chart recognition-network (take 1000 test-data) (take 1000 test-labels)))

;; ===============================
;; END of DEMO



;; final view of feature maps
(task/stop-all)

;; (show (map feature-img (feature-maps recognition-network :scale 10)) :title "Recognition maps")
;; (show (map feature-img (feature-maps reconstructor :scale 10)) :title "Round trip maps")
)
