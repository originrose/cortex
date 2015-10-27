(ns thinktopic.cortex.autoencoder)

;; K-sparse autoencoder
;0) setup
;       initialize weights with gaussian and standard deviation of 0.01
;       momentum = 0.9
;       learning_rate = 0.01
;       learning_rate_decay = 0.0001 (maybe?) (decay rate to 0.001 over first
;       few hundred epochs, then stick at 0.001)
;1) forward phase
;       z = W*x + b
;2) k-sparsify
;       zero out all but top-K activations
;3) reconstruction (uses tied weights)
;       x~ = Wt*z + b'
;4) error
;       e = sqrt((x - x~)^2)
;5) backpropagate error through top-K units using momentum
;       v_(k+1) = m_(k)*v_(k) - n_(k)df(x_(k))
;       x_(k+1) = x_(k) + v_(k)

;       (v = velocity, m = momentum, n=learning rate)

