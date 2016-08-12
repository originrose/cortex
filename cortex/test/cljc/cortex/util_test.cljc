(ns cortex.util-test
  (:refer-clojure :exclude [defonce])
  (:require
    #?(:cljs [cljs.test :refer-macros [deftest is testing]]
       :clj [clojure.test :refer [deftest is testing]])
    [clojure.core.matrix :as m]
    [clojure.string :as str]
    [cortex.util :refer :all]))

;;;; Mathematics

(deftest clamp-test
  (is (= (clamp 1 0 3) 1.0))
  (is (= (clamp 1 2 3) 2.0))
  (is (= (clamp 1 4 3) 3.0)))

(deftest relative-error-test
  (is (= (relative-error 10 10) 0.0))
  (is (= (relative-error 9 10) 0.1))
  (is (= (relative-error 10 9) 0.1)))

(deftest avg-test
  (is (approx= 1e-12
               (avg 1 3 7)
               (double 11/3)))
  (is (approx= 1e-12
               (apply avg (range 10))
               4.5))
  (is (approx= 1e-12
               (avg 2 -3)
               -0.5)))

;;;; Sequences

(deftest seq-like?-test
  (is (false? (seq-like? nil)))
  (is (false? (seq-like? "Hello")))
  (is (false? (seq-like? 5)))
  (is (true? (seq-like? (range 5))))
  (is (true? (seq-like? [:a :b :c])))
  (is (true? (seq-like? {:a 1 :b 2})))
  (is (true? (seq-like? (m/array :vectorz [1 2 3]))))
  (is (true? (seq-like? (m/array :vectorz [[1 2] [3 4]])))))

(deftest interleave-all-test
  (is (= (interleave-all) []))
  (is (= (interleave-all (range 5)) (range 5)))
  (is (= (interleave-all [:a :b :c] [:d :e] [:f :g :h :i])
         [:a :d :f :b :e :g :c :h :i])))

(deftest pad-test
  (is (= (pad 5 (range 5) [:a :b :c])
         [:a :b :c 0 1]))
  (is (= (pad 3 (range 5) [:a :b :c])
         [:a :b :c]))
  (is (= (pad 2 (range 5) [:a :b :c])
         [:a :b :c])))

;;;; Collections

(deftest map-keys-test
  (is (= (map-keys name {:a 1 :b 2})
         {"a" 1 "b" 2}))
  (is (= (map-keys inc {1 2 3 4})
         {2 2 4 4}))
  (is (= (map-keys #(error "key fn should not be called") {})
         {})))

(deftest map-vals-test
  (is (= (map-vals dec {:a 1 :b 2})
         {:a 0 :b 1}))
  (is (= (map-vals #(cons 0 %) {[1 2] [3 4] [5 6] [7 8]})
         {[1 2] [0 3 4] [5 6] [0 7 8]})))

(deftest map-entry-test
  (is (= (key (map-entry :a :b)) :a))
  (is (= (val (map-entry :a :b)) :b))
  (is (= (vec (map-entry nil nil)) [nil nil])))

(deftest approx=-test
  (testing "comparing floats"
    (is (false? (approx= 0.1 10 11)))
    (is (false? (approx= 0.1 10 11.0)))
    (is (true? (approx= 0.1 10.0 11.0)))
    (is (false? (approx= 0.09 10.0 11.0))))
  (testing "comparing more than two floats"
    (is (false? (approx= 0.1 10.0 10.75 11.5)))
    (is (true? (approx= 0.1 10.0 10.5 11.0))))
  (testing "comparing vectors"
    (is (false? (approx= 0.1
                         (m/array :vectorz [10.0 11.0])
                         [11.5 10.0])))
    (is (true? (approx= 0.1
                        (m/array :vectorz [10.0 11.0])
                        [11.0 10.0]))))
  (testing "comparing nested data structures"
    (is (false? (approx= 0.1
                         {:a [10.0 11.0] :b 10.0}
                         {:a [11.0 10.0] :c 10.0})))
    (is (false? (approx= 0.1
                         {:a [10.0 11.0] :b 10.0}
                         {:a [12.0 10.0] :b 10.0})))
    (is (true? (approx= 0.1
                        {:a [10.0 11.0] :b 10.0}
                        {:a [11.0 10.0] :b 10.0}))))
  (testing "lists of different lengths"
    (is (false? (approx= 0.1
                         [1.0 2.0 3.0]
                         [1.0 2.0])))))

;;;; Lazy maps

(defn make-lazy-map
  []
  (->LazyMap
    {:a (delay (error "value :a should not be realized"))
     :b 50
     :c (delay (error "value :c should not be realized"))}))

(deftest lazy-map-test
  (testing "lazy maps are lazy"
    (is (= (:b (make-lazy-map))
           50))
    (is (= (with-out-str
             (:c (->LazyMap
                   {:a (delay (error "value :a should not be realized"))
                    :b 50
                    :c (delay (print "value :c was realized"))})))
           "value :c was realized")))
  (testing "lazy maps can be called as fns"
    (is (= (let [m (->LazyMap
                     {:a 1
                      :b 2})]
             (m :b))
           2)))
  (testing "keys and vals work on lazy maps"
    (is (= (set
             (keys (make-lazy-map)))
           #{:a :b :c}))
    (is (= (->> (->LazyMap
                  {:a (delay (println "value :a was realized"))
                   :b (delay (println "value :b was realized"))
                   :c (delay (println "value :c was realized"))})
             (vals)
             (take 2)
             (dorun)
             (with-out-str)
             (re-seq #"value :[a-c] was realized")
             (count))
           2)))
  (testing "assoc and dissoc work with lazy maps"
    (is (= (-> (make-lazy-map)
             (assoc :d (delay (error "value :d should not be realized")))
             (keys)
             (set))
           #{:a :b :c :d}))
    (is (= (-> (make-lazy-map)
             (dissoc :a :b)
             (keys)
             (set))
           #{:c})))
  (testing "lazy maps support default value for lookup"
    (is (= (:d (make-lazy-map) :default)
           :default))
    (is (= (get (make-lazy-map) :d :default)
           :default)))
  (testing "seqs for lazy maps do not contain delays"
    (is (= (set (->LazyMap
                  {:a 1
                   :b (delay 2)}))
           #{[:a 1] [:b 2]})))
  (testing "equality for lazy maps"
    (is (= (->LazyMap
             {:a 1
              :b (delay 2)})
           {:a 1
            :b 2})))
  (testing "reduce and kv-reduce for lazy maps"
    (is (= (reduce (fn [m [k v]]
                     (assoc m k v))
                   {}
                   (->LazyMap
                     {:a 1
                      :b (delay 2)}))
           {:a 1 :b 2}))
    (is (= (reduce-kv (fn [m k v]
                        (assoc m k v))
                      {}
                      (->LazyMap
                        {:a 1
                         :b (delay 2)}))
           {:a 1 :b 2})))
  (testing "string representation of lazy maps"
    (is (= (pr-str (->LazyMap {:a 1}))
           "{:a 1}"))
    (is (= (pr-str (->LazyMap {:a (delay 1)}))
           "{:a <unrealized>}")))
  (testing "string representation of lazy map entries"
    (is (= (pr-str (lazy-map-entry :a 1))
           "[:a 1]"))
    (is (= (pr-str (lazy-map-entry :a (delay 1)))
           "[:a <unrealized>]")))
  (testing "lazy-map macro"
    (is (= (with-out-str
             (let [m (lazy-map
                       {:a (println "value :a was realized")
                        :b (println "value :b was realized")
                        :c (println "value :c was realized")})]
               (doseq [k [:b :c]]
                 (k m))))
           (format "value :b was realized%nvalue :c was realized%n"))))
  (testing "forcing a lazy map"
    (is (= (->> (lazy-map
                  {:a (println "value :a was realized")
                   :b (println "value :b was realized")})
             (force-map)
             (with-out-str)
             (re-seq #"value :[ab] was realized")
             (set))
           #{"value :a was realized" "value :b was realized"}))))

;;;; Confusion matrices

(deftest confusion-test
  (let [cf (confusion-matrix ["cat" "dog" "rabbit"])
        cf (-> cf
             (add-prediction "dog" "cat")
             (add-prediction "dog" "cat")
             (add-prediction "cat" "cat")
             (add-prediction "cat" "cat")
             (add-prediction "rabbit" "cat")
             (add-prediction "dog" "dog")
             (add-prediction "cat" "dog")
             (add-prediction "rabbit" "rabbit")
             (add-prediction "cat" "rabbit")
             )]
    ;; (print-confusion-matrix cf)
    (is (= 2 (get-in cf ["cat" "dog"])))))

(deftest test-convergence
  (is (converges? (range 10) 5))
  (is (not (converges? [] 5)))
  (is (not (converges? (range 10) 5 {:hits-needed 2})))
  (is (converges? (range 10) 5 {:hits-needed 2 :tolerance 1}))
  (is (not (converges? (range 10) 15)))
  (is (converges? (range 10) 15 {:tolerance 10})))

(deftest test-mse-gradient
  (is (m/equals [-2 0 2] (mse-gradient-fn [10 11 12] [11 11 11]))))

;;;; Formatting

(deftest parse-long-test
  (is (= (parse-long "5") 5))
  (is (= (parse-long "-47") -47)))

(deftest parse-double-test
  (is (= (parse-double "5") 5.0))
  (is (= (parse-double "-7.77e-7") -7.77e-7)))

(deftest fmt-string-regex-test
  (is (= (rest (re-matches fmt-string-regex "%3$2s"))
         ["3" "2s"]))
  (is (= (rest (re-matches fmt-string-regex "%(,.2f"))
         [nil "(,.2f"]))
  (is (= (map rest (re-seq fmt-string-regex "%1$tm %1$te,%1$tY"))
         [["1" "tm"]
          ["1" "te"]
          ["1" "tY"]]))
  (is (= (map rest (re-seq fmt-string-regex "Format %1$s as %1$.2f using the %%.2f formatter%n."))
         [["1" "s"]
          ["1" ".2f"]
          [nil "%"]
          [nil "n"]])))

(deftest sformat-test
  (testing "basic formatting"
    (is (= (sformat "Hello, %s!" "world")
           "Hello, world!")))
  (testing "numeric formatting and %% specifier"
    (is (= (sformat "Sales tax of %d%% on $%.2f item" 37 1.25)
           "Sales tax of 37% on $1.25 item")))
  (testing "autocasting"
    (is (= (sformat "Long as double %.2f and double as long %d" 3141 2.718)
           "Long as double 3141.00 and double as long 2")))
  (testing "positional arguments"
    (is (= (sformat "%s %% %1$s %s %3$s %3$s %% %s %2$s %s" 1 2 3 4)
           "1 % 1 2 3 3 % 3 2 4")))
  (testing "map format specifier over collections"
    (is (= (sformat "%1$.2f %1$.4f" [Math/PI Math/E 5])
           "[3.14 2.72 5.00] [3.1416 2.7183 5.0000]")))
  (testing "formatting core.matrix arrays as nested collections"
    (is (= (sformat "%.1f" (m/identity-matrix :vectorz 2))
           "[[1.0 0.0] [0.0 1.0]]"))))

(deftest sprintf-test
  (is (= (with-out-str (sprintf "%(d%n" [[1 -3 5] [-7 3]]))
         (format "[[1 (3) 5] [(7) 3]]%n"))))
