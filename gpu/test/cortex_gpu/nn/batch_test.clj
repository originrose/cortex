(ns cortex-gpu.nn.batch-test
  (:require [clojure.test :refer :all]
            [cortex-gpu.test-framework :as framework]
            [cortex-gpu.nn.batch :as batch]
            [cortex-gpu.nn.cudnn :as cudnn]
            [cortex-gpu.cuda :as cuda]
            [clojure.core.matrix :as m])
  (:import [org.bytedeco.javacpp DoublePointer IntPointer]
           [java.nio DoubleBuffer]))

(use-fixtures :once framework/with-contexts)
(use-fixtures :each framework/with-resource-context)

(deftest indexed-copy
  []
  (let [stride 3
        batch-count 10
        num-batches 2
        item-count (* stride batch-count num-batches)
        src-data-seq (range item-count)
        src-data (cudnn/array src-data-seq)
        ^ints index-data (int-array (range (quot item-count stride)))
        src-data-input (vec (partition stride src-data-seq))
        host-ptr (IntPointer. index-data)
        index-byte-size (* (.capacity host-ptr) Integer/BYTES)
        ^IntPointer device-ptr (cuda/mem-alloc index-byte-size (IntPointer.))
        _ (cuda/mem-copy-host->device host-ptr device-ptr index-byte-size)
        network (batch/setup-batch-buffer { :batch-indexes { :device-ptr device-ptr }}
                                          batch-count stride :batch-buffer)
        batch-buffer (:batch-buffer network)]
    (batch/load-batch-buffer network src-data 0 batch-buffer)
    (is (= (map double (take (* stride batch-count) src-data-seq))
           (seq (cudnn/to-double-array batch-buffer))))
    (batch/load-batch-buffer network src-data 1 batch-buffer)
    (is (= (map double (drop (* stride batch-count) src-data-seq ))
           (seq (cudnn/to-double-array batch-buffer))))))
