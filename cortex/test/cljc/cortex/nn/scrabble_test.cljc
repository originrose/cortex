(ns cortex.nn.scrabble-test
  (:require
    [cortex.nn.core :refer [calc output forward backward input-gradient parameters gradient parameter-count]]
    #?(:cljs
        [cljs.test :refer-macros [deftest is are testing]]
        :clj
        [clojure.test :refer [deftest is are testing]])
    [cortex.optimise :as opt]
    [clojure.core.matrix :as mat]
    #?(:cljs [thi.ng.ndarray.core :as nd])
    [cortex.nn.network :as net]
    [cortex.nn.core :as core]
    [cortex.nn.layers :as layers]))

(def scrabble-values {\a 1 \b 3 \c 3 \d 2 \e 1 \f 4 \g 2 \h 4 \i 1 \j 8 \k 5
                      \l 1 \m 3 \n 1 \o 1 \p 3 \q 10 \r 1 \s 1 \t 1 \u 1 \v 4
                      \w 4 \x 8 \y 4 \z 10})

#?(:cljs (enable-console-print!))

(defn- parse-int2 [string]
  #?(:clj (Integer/parseInt string 2)
     :cljs (js/parseInt string 2)))

(defn- charval [character]
  (.indexOf "abcdefghijklmnopqrstuvwxyz" #?(:clj (str character) :cljs character)))

#?(:clj (do

(defn num->bit-array [input & {:keys [bits]
                               :or {bits 4}}]
 (->> (clojure.pprint/cl-format nil (str "~" bits ",'0',B") input)
      (mapv #(new Double (str %)))))

(defn char->bit-array [character]
  (let [offset (charval character)
        begin (take offset (repeatedly #(new Double 0.0)))
        end (take (- 26 offset 1) (repeatedly #(new Double 0.0)))]
  (into [] (concat begin [1.0] end)))))

:cljs (do

(defn num->bit-array [input & {:keys [bits]
                               :or {bits 4}}]
 (->> (cljs.pprint/cl-format nil (str "~" bits ",'0',B") input)
      (mapv #(js/parseFloat (str %)))))

(defn char->bit-array [character]
  (let [offset (charval character)
        begin (take offset (repeatedly #(js/parseFloat 0.0)))
        end (take (- 26 offset 1) (repeatedly #(js/parseFloat 0.0)))]
  (into [] (concat begin [1.0] end))))))

(defn bit-array->num [input]
  (-> (map int input)
      (clojure.string/join)
      (parse-int2)))

(defn classify [network character]
  (->> (net/run network [(char->bit-array character)])
       (first)
       (map #(if (> % 0.5) 1 0))
       (clojure.string/join)
       (parse-int2)))

(deftest scrabble-test
  []
  (let [training-data (into [] (for [k (keys scrabble-values)] (char->bit-array k)))
        training-labels (into [] (for [v (vals scrabble-values)] (num->bit-array v)))
        input-width (last (mat/shape training-data))
        output-width (last (mat/shape training-labels))
        hidden-layer-size 4
        n-epochs 400
        batch-size 1
        loss-fn (opt/mse-loss)
        network (core/stack-module [(layers/linear-layer input-width hidden-layer-size)
                         (layers/logistic [hidden-layer-size])
                         (layers/linear-layer hidden-layer-size output-width)])
        optimizer (opt/sgd-optimiser (core/parameter-count network))
        trained-network (net/train network optimizer loss-fn training-data training-labels batch-size n-epochs)]

      (is (= (classify trained-network \j) 8))
      (is (= (classify trained-network \k) 5))))
