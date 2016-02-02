(ns cortex.test
  (:require [doo.runner :refer-macros [doo-tests]]
            [cortex.wiring-test]
            [cortex.function-test]
            [cortex.network-test]
            [cortex.normaliser-test]
            [cortex.optimise-test]
            [cortex.util-test]))

(enable-console-print!)

(doo-tests 'cortex.wiring-test
           'cortex.function-test
           'cortex.util-test
           'cortex.network-test
           'cortex.normaliser-test
           'cortex.optimise-test)

