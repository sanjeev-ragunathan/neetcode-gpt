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

> The derivative value tells us - the direction and the magnitude of error.  
> +ve gradient - loss is increasing as parameter value increases  
> -ve gradient - loss is decreasing as parameter value decreases  
> 0 gradient - algorithm has converged  
> size of the gradient - tells us the size of the adjustments that needs to be made. Larger size - larger adjustment.  

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

### Softmax
Converts logits (raw vector) -> probabilities (probability distribution).  
This is how GPT decides which token comes next.  

Temperature - softmax(z/T). T - temperature. This tells how the distribution is supposed to be.  
- Higher T value - stronger distribution (Exploitation) - forces the right one to correct with very high prob.
- Lower T value - flatter distribution (Exploration) - forces the right one to be not so high so the model can explore other tokens - randomness.

### Cross-Entropy Loss
Entropy - uncertainity, surprise.  
How uncertain is the model in it's prediction of the next token? High Entropy - Highly uncertain(unsure). Low Entropy - confident.

Cross-Entropy Loss - How wrong is the model?
- High when confidently wrong, and low when it's right.
- Penalizes strongly - confident wrong predictions. Done using log.
- Perfect Prediction - 0 loss.

Binary Cross-Entropy

$$
L = -\frac{1}{n} \sum_{i=1}^{n} \left[y_i \ln(p_i) + (1-y_i)\ln(1-p_i)\right]
$$

$$
\begin{aligned}
n \text{: no.of samples} \\
y_i \ln(p_i) \text{: this is for when y is true i.e. 1} \\ 
(1-y_i)\ln(1-p_i) \text{: this is for when y is false i.e. 0}
\end{aligned}
$$

Categorical Cross-Entropy

$$
L = -\frac{1}{n} \sum_{i=1}^{n} \sum_{c=1}^{C} y_{i,c}\ln(p_{i,c})
$$

$$
\begin{aligned}
n \text{: no.of samples} \\ 
C \text{: no.of categories} \\
y_{i,c}\ln(p_{i,c}) \text{: for only when y is true i.e. 1}
\end{aligned}
$$

> Loss is calculated for every sample, then average is calculated.  
> If low - model is confident and correct, if high - model is confident and wrong.

### Linear Regression (forward pass)
Get model prediction
> Y = X(inputs) . W(weights); np.dot(x, w).
> dot product is basically just matrix multiplication between just two vectors - resulting in a sigle number
> matrix multiplication is just a combination of multiple dot products. between matrices - resulting in a matrix.

Get error - MSE (Mean Squared Error)

$$
MSE = -\frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_{i})^{2}
$$

### Linear Regression (Training)

