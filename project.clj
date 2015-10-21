(defproject thinktopic/cortex "0.1.0-SNAPSHOT"
  :description "A neural network toolkit for ThinkTopic projects."
  :dependencies [[ org.clojure/clojure "1.7.0"]
                 [com.taoensso/timbre "4.1.4"]
                 [net.mikera/vectorz-clj "0.35.0"]
                 [thinktopic/datasets "0.1.1"]
                 ;[com.nuroko/nurokore "0.0.6"]
                 ;[com.nuroko/nurokit "0.0.3"]
                 ;[net.mikera/imagez "0.5.0"]
                 ;[incanter/incanter-core "1.9.0"]
                 ;[incanter/incanter-charts "1.9.0"]
                 ]

  :jvm-opts  ["-Xmx8g"
              "-XX:+UseConcMarkSweepGC"
              "-XX:-OmitStackTraceInFastThrow"])
