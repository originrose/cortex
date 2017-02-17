(ns cortex.compute.math-util)



;;A (a x b)
;;B (b x c)
;;C (a x c)
;; (a x b)(b x c) = (a x c)
;;transposed is:
;; A (b x a)
;; B (c x b)
;; C (c x a)
;; (c x b)(b x a) = (c x a)
;; a = a-row-count
;; b = a-col-count = b-row-count
;; c = b-col-count
(defn col->row-gemm
  "Perform a column major gemm using a library that is row-major."
  [blas-fn trans-a? trans-b? a-row-count a-col-count b-col-count
   alpha A a-colstride
   B b-colstride
   beta C c-colstride]
  (blas-fn trans-b? trans-a?
           b-col-count a-col-count a-row-count
           alpha B b-colstride
           A a-colstride
           beta C c-colstride))


(defn bool->blas-trans
  ^String [trans?]
  (if trans? "t" "n"))


(defn col->row-gemv
  "Perform a column-major gemv using a library that is row-major."
  [blas-fn trans-a a-row-count a-col-count alpha a a-colstride x inc-x beta y inc-y]
  (blas-fn (not trans-a) a-col-count a-row-count
           alpha a a-colstride
           x inc-x
           beta y inc-y))
