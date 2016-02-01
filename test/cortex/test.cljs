(ns cortex.test
  (:require [doo.runner :refer-macros [doo-tests]]
            [cortex.wiring-test]
            [cortex.function-test]
            [cortex.network-test]
            [cortex.normaliser-test]))

(enable-console-print!)

(doo-tests 'cortex.wiring-test
           'cortex.function-test
           'cortex.network-test
           'cortex.normaliser-test)

