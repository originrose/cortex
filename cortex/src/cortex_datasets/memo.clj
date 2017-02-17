(ns cortex-datasets.memo
  (:import [com.github.benmanes.caffeine.cache Caffeine CacheLoader LoadingCache]))


;;This overlaps functionality with the clojure cache system but tests with the caffeine backend
;;indicate that it is far more memory efficient (generates far less garbage under heave usage scenarios)
;;and it is much faster.


(defn create-cache-loader
  ^CacheLoader [loader-fn]
  (reify CacheLoader
    (load [this key]
      (loader-fn key))))


(defn make-caffeine-lru-cache
  [loader-fn & {:keys [maximum-size]
                :or {maximum-size 32}}]
  (let [builder (Caffeine/newBuilder)]
    (doto builder
      (.maximumSize maximum-size))
    (.build builder (create-cache-loader loader-fn))))


(defn lookup-value
  [^LoadingCache cache, value]
  (.get cache value))


(defn build-caffeine-memo
  [data-fn & {:keys [maximum-size]
              :or {maximum-size 32}}]
  (let [new-cache (make-caffeine-lru-cache (fn [key] (apply data-fn key)) :maximum-size maximum-size)]
    (fn [& args]
      (lookup-value new-cache args))))
