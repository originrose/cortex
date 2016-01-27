(ns cortex.test
  (:require [doo.runner :refer-macros [doo-tests]]
            [cortex.wiring-test]))

(enable-console-print!)

(doo-tests 'cortex.wiring-test)

