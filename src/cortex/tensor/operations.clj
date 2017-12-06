(ns cortex.tensor.operations
  "tensor operations with syntatic sugar"
  (:refer-clojure :exclude [max min * - + > >= < <= bit-and bit-xor])
  (:require [clojure.core.matrix :as m]
            [cortex.tensor :as tensor]
            [think.datatype.core :as dtype]))

(defn max
  "Takes an input tensor and returns the max of x or the value given, mutates the output tensor and returns it"
  ([output max-value]
   (max output output max-value))
  ([output input max-value]
   (tensor/binary-op! output 1.0 input 1.0 max-value :max)))

(defn min
  "Takes an input tensor and returns the min of x or the value given, mutates the output tensor and returns it"
  ([output min-value]
   (min output output min-value))
  ([output input min-value]
   (tensor/binary-op! output 1.0 input 1.0 min-value :min)))

(defn ceil
  "Takes an tensor returns the mutated tensor with the value with the ceiling function applied"
  ([output]
   (ceil output output))
  ([output input]
   (tensor/unary-op! output 1.0 input :ceil)))

(defn floor
  "Takes an tensor returns the mutated tensor with the value with the floor function applied"
  ([output]
   (floor output output))
  ([output input]
   (tensor/unary-op! output 1.0 input :floor)))

(defn logistic
  "Takes an tensor returns the mutated tensor with the value with the logistic function applied"
  ([output]
   (logistic output output))
  ([output input]
   (tensor/unary-op! output 1.0 input :logistic)))

(defn tanh
  "Takes an tensor returns the mutated tensor with the value with the tanh function applied"
  ([output]
   (tanh output output))
  ([output input]
   (tensor/unary-op! output 1.0 input :tanh)))

(defn exp
  "Takes an tensor returns the mutated tensor with the value with the exp function applied"
  ([output]
   (exp output output))
  ([output input]
   (tensor/unary-op! output 1.0 input :exp)))

(defn *
  "Takes and x1 and x2 multiples them together and puts the result in the mutated output"
  ([output x1]
   (* output output x1))
  ([output x1 x2]
   (tensor/binary-op! output 1.0 x1 1.0 x2 :*)))

(defn -
  "Takes and x1 and x2 subtracts them and puts the result in the mutated output"
  ([output x1]
   (- output output x1))
  ([output x1 x2]
   (tensor/binary-op! output 1.0 x1 1.0 x2 :-)))

(defn +
  "Takes and x1 and x2 adds them and puts the result in the mutated output"
  ([output x1]
   (+ output output x1))
  ([output x1 x2]
   (tensor/binary-op! output 1.0 x1 1.0 x2 :+)))

(defn >
  "Takes and x1 and x2 returns 1 if x1 > x2 and 0 otherwise them and puts the result in the mutated output"
  ([output x1]
   (> output output x1))
  ([output x1 x2]
   (tensor/binary-op! output 1.0 x1 1.0 x2 :>)))

(defn >=
  "Takes and x1 and x2 returns 1 if x1 >= x2 and 0 otherwise them and puts the result in the mutated output"
  ([output x1]
   (>= output output x1))
  ([output x1 x2]
   (tensor/binary-op! output 1.0 x1 1.0 x2 :>=)))

(defn <
  "Takes and x1 and x2 returns 1 if x1 < x2 and 0 otherwise them and puts the result in the mutated output"
  ([output x1]
   (< output output x1))
  ([output x1 x2]
   (tensor/binary-op! output 1.0 x1 1.0 x2 :<)))

(defn <=
  "Takes and x1 and x2 returns 1 if x1 <= x2 and 0 otherwise them and puts the result in the mutated output"
  ([output x1]
   (<= output output x1))
  ([output x1 x2]
   (tensor/binary-op! output 1.0 x1 1.0 x2 :<=)))

(defn bit-xor
  "Takes and x1 and x2 returns xor x1 and x2 them and puts the result in the mutated output"
  ([output x1]
   (bit-xor output output x1))
  ([output x1 x2]
   (tensor/binary-op! output 1.0 x1 1.0 x2 :bit-xor)))

(defn bit-and
  "Takes and x1 and x2 returns bit and of x1 and x2 and them and puts the result in the mutated output"
  ([output x1]
   (bit-and output output x1))
  ([output x1 x2]
   (tensor/binary-op! output 1.0 x1 1.0 x2 :bit-and)))

(defn new-tensor
  "Returns a new tensor of the same shape and type of the given output tensor"
  [output]
  (tensor/new-tensor (m/shape output) :datatype (dtype/get-datatype output)))

(defn where
  "Takes a tests tensor of 1s and 0s and a tensor of then and a tensor of else.
  It will multiply the test and the complement of the test
  to the then and else and then add the result to the output.
  The condition tensor acts as a mask that chooses, based on the value at each element,
  whether the corresponding element / row in the output should be taken from x1 (if true) or x2 (if false)."
  [output test then else]
  (let [compl (bit-xor (new-tensor test) test 1)
        x1 (* then test)
        x2 (* else compl)]

      ;; add the two conditional branches together
    (+ output x1 x2)))
