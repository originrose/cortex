(set-env! 
  :dependencies '[[org.clojure/clojure "1.8.0"]
                 [thinktopic/experiment "0.9.9"]
                 [org.clojure/tools.cli "0.3.5"]

                 ; to manipulate images
                 [thinktopic/think.image "0.4.8"]
                 ; [net.mikera/imagez "0.12.0"]
                 ;;If you need cuda 8...
                 [org.bytedeco.javacpp-presets/cuda "8.0-1.2"]
                 ;;If you need cuda 7.5...
                 ;;[org.bytedeco.javacpp-presets/cuda "7.5-1.2"]

                 [proto-repl "0.3.1" :scope "test"]
                 [proto-repl-charts "0.3.1" :scope "test"]
                 ])


(deftask dev
  "Profile setup for development.
    Starting the repl with the dev profile...
    boot dev repl "
  []
  (println "Dev profile running")
  (set-env!
   :init-ns 'user
   :source-paths #(into % ["dev" "test"])
   :dependencies #(into % '[[org.clojure/tools.namespace "0.2.11"]]))

  ;; Makes clojure.tools.namespace.repl work per https://github.com/boot-clj/boot/wiki/Repl-reloading
  (require 'clojure.tools.namespace.repl)
  (eval '(apply clojure.tools.namespace.repl/set-refresh-dirs
                (get-env :directories)))

identity)