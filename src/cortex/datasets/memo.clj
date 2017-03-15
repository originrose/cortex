(ns cortex.datasets.memo
  (:import [com.github.benmanes.caffeine.cache Caffeine CacheLoader LoadingCache]))


;;This overlaps functionality with the clojure cache system but tests with the caffeine backend
;;indicate that it is far more memory efficient (generates far less garbage under heave usage scenarios)
;;and it is much faster.


(defn cache-loader
  ^CacheLoader [loader-fn]
  (reify CacheLoader
    (load [this key]
      (apply loader-fn key))))


(defn caffeine-lru-cache
  [loader & {:keys [maximum-size]
                :or {maximum-size 32}}]
  (let [builder (Caffeine/newBuilder)]
    (doto builder
      (.maximumSize maximum-size))
    (.build builder loader)))


(defn lookup-value
  [^LoadingCache cache, value]
  (.get cache value))


(defn lru-cache
  [data-fn & {:keys [maximum-size]
              :or {maximum-size 1024}}]
  (let [new-cache (caffeine-lru-cache
                    (cache-loader #(apply data-fn %))
                    :maximum-size maximum-size)]
    (fn [& args]
      (lookup-value new-cache args))))

