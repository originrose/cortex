(ns cortex.impl.vectorz-blas
  (:require [mikera.vectorz.core]
            [core.blas.protocols :as blas]
            [clojure.core.matrix.protocols :as cp]
            [clojure.core.matrix :as m]
            [cortex.backends :as b])
  (:import [com.github.fommil.netlib BLAS]
           [mikera.arrayz INDArray]))



(extend-protocol blas/PBLASBase
  INDArray
  (supports-blas? [c] (cp/as-double-array c))
  (gemm! [c trans-a? trans-b? alpha a b beta]
    (let [alpha (double alpha)
          beta (double beta)
          a-shape (m/shape a)
          b-shape (m/shape b)
          c-shape (m/shape c)
          a-col-count (second a-shape)
          b-col-count (second b-shape)
          c-col-count (second c-shape)
          a-shape (if trans-a?
                    [(a-shape 1) (a-shape 0)]
                    a-shape)
          b-shape (if trans-b?
                    [(b-shape 1) (b-shape 0)]
                    b-shape)
          M (long (first a-shape))
          N (long (second b-shape))
          K (long (first b-shape))
          a-data (cp/as-double-array a)
          b-data (cp/as-double-array b)
          c-data (cp/as-double-array c)
          trans-command-a (if trans-a? "t" "n")
          trans-command-b (if trans-b? "t" "n")]
      (when-not (and (= K (second a-shape))
                     (= M (first c-shape))
                     (= N (second c-shape)))
        (throw (Exception. (format "Incompatible matrix sizes: a %s b %s m %s"
                                   (str a-shape)
                                   (str b-shape)
                                   (str c-shape)))))
      (.dgemm (BLAS/getInstance) trans-command-b trans-command-a N M K alpha
              b-data b-col-count
              a-data a-col-count
              beta
              c-data c-col-count)
      c))
  (gemv! [c trans-a? alpha a b beta]
    (let [alpha (double alpha)
          beta (double beta)
          a-shape (m/shape a)
          a-row-count (first a-shape)
          a-col-count (second a-shape)
          a-shape (if trans-a?
                    [(a-shape 1) (a-shape 0)]
                    a-shape)
          M (first a-shape)
          N (second a-shape)
          a-data (cp/as-double-array a)
          b-data (cp/as-double-array b)
          c-data (cp/as-double-array c)
          _ (when-not (or (= N (count b-data))
                          (= N (count c-data)))
              (throw (Exception. "GEMV mismatch")))
          ;;The matrix is already transposed according to blas
          ;;so we reverse this here
          trans-command-a (if trans-a? "n" "t")]
      (.dgemv (BLAS/getInstance) trans-command-a a-col-count a-row-count
              alpha
              a-data a-col-count
              b-data 1
              beta
              c-data 1)
      c)))
