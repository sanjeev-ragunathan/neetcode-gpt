# Notes
> Learning notes - built as I go through each topic.

## Foundations

### Gradient Descent
Used to update parameters. `w = w - lr * dL/dw`.
`w` - parameter, `lr` - learning rate, `dL/dw` - total gradient of loss with respect to weight, computed through chain rule.

Backpropagating loss - `dL/dw = dL/da * da/dz * dz/dw` (Single layer). Chain Rule.
```
dL/dw = dL/da * da/dz * dz/dw
         ↑ loss    ↑ activation  ↑ input
         gradient  derivative    to layer
```

`da/dz`, `a` - activation function.

A bit more on gradients and how parameters and loss are calculated.

- Single Layer - three term chain rule
  ```
  Input x → Linear (z = w*x + b) → Activation (a = sigmoid(z)) → Loss L
  
  dL/dw = dL/da * da/dz * dz/dw
           ↑ 3 terms only
  ```

- Two Layers - the chain rule gets longer
  ```
  x → z1 = w1*x + b1 → a1 = relu(z1) → z2 = w2*a1 + b2 → a2 = sigmoid(z2) → L
  
  dL/dw1 = dL/da2 * da2/dz2 * dz2/da1 * da1/dz1 * dz1/dw1
           ↑ from loss  ↑ sigmoid'  ↑ w2 passes  ↑ relu'   ↑ input to layer 1
           
           = (loss term) * (0.25) * (w2) * (1 or 0) * (x)
  ```

- 10 Layers - it EXPLODES!
  ```
  x → [Layer 1] → [Layer 2] → ... → [Layer 10] → Loss
  
  dL/dw1 = dL/da10 * da10/dz10 * dz10/da9 * da9/dz9 * dz9/da8 * ... * da1/dz1 * dz1/dw1
           ↑ All the way back!
           
           ≈ (something) * (0.25) * (w10) * (0.25) * (w9) * (0.25) * ... * (0.25) * (0.25)
           # Let's assume activation is sigmoid, therefore 0.25
  ```

### Sigmoid & ReLU - Activation Functions
What happened to step function? - derivative is 0, useless during backpropagation.  
> In gradient Descent - we compute gradient using chain rule. `dL/dw = dL/da * da/dz * dz/dw`.  
A step function breaks the chain, `d(step)/dw = 0`. Sigmoid keeps the chain alaive, `d(sigmoid)/dw ≠ 0`. 

**Sigmoid** - Squish between 0 and 1. Probability.  
> Problem with Sigmoid? - Vanishing gradients. Sigmoids maximum derivative value is 0.25. In deep networks -> gradients shrink exponentially.  
`d(sigmoid)/dz = sigmoid(z) * (1 - sigmoid(z)) — max 0.25`

**ReLU** - max(0, x). Kill negative values. No Squishing.
> NO squishing - no vanishing gradients.
> Why kill negatives? - Makes some neurons inactive, increasing randomness, brings disconinuity in slope, can't convert to a linear transformation - breaks linearity - Gives Non-linearity. Also Sparsity - less neurons active - less compute - side benefit.  
> `d(ReLU)/dz = 1 if z > 0, else 0` Always 1 - so no vanishing or exploding gradients.
