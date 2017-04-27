(ns cortex.dataset
  "A Cortex dataset is a sequence of maps. The sequence is the dataset
  and the maps are the individual observations. Keys in the maps are keywords
  that indicate variables of the obeservation (e.g., :x :y :data :labels etc.)
  Values in the observation maps are core.matrix compatible vectors/matrices;
  notably, this includes persistent Clojure vectors of doubles.

  A number of related, but separable, concerns are handled elsewhere:
  - NN specific concerns, e.g. batching, are handled by NN specific code.
  Details about epochs and testing are specified at training time and are
  not inherent to datasets themselves.

  - Data augmentation, e.g. image processing, is handled by the dataset
  creator. Cortex expects that the dataset sequence is pre-augmented.
  Helpful utility functions are provided in Cortex, think.image, and
  elsewhere.

  - Testing, e.g validation, cross-validation, holdout, etc... are typically
  handled by simply having multiple datasets (again, read 'sequeces of
  maps'). For example, a common pattern is to have one dataset for
  training and another for test.

  Serialization to various formats is simple:
  For each format, two functions are necessary.
  (1) A function to take a sequence of maps and serialize it.
  (2) A function that takes serialized data and returns a sequence of maps.")
