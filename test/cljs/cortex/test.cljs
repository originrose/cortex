(ns cortex.nn.test
  (:require [doo.runner :refer-macros [doo-tests]]
            [cortex.nn.wiring-test]
            [cortex.nn.function-test]
            [cortex.nn.network-test]
            [cortex.nn.normaliser-test]
            [cortex.nn.scrabble-test]
            [cortex.optimise-test]
            [cortex.util-test]))

(enable-console-print!)

(doo-tests 'cortex.nn.wiring-test
           'cortex.nn.function-test
           'cortex.util-test
           'cortex.nn.scrabble-test
           'cortex.nn.network-test
           'cortex.nn.normaliser-test
           'cortex.optimise-test)

