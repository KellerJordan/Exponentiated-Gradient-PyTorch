# Exponentiated-Gradient-PyTorch

Configurable PyTorch implementation of the exponentiated gradient (EG) algorithm and plus-minus variant. For a detailed description of the algorithm, see [1].

To use, import and then initialize the optimizer:

    from egpm import EGPM
    
    optim = EGPM(
        model.parameters(),
        lr,
        u_scaling=100,
        norm_per='neuron',
        gradient_clipping=False,
        weight_regularization=None,
        plus_minus=True,
        init='log_normal',
    )

It can now be used the same way as any other optimizer in PyTorch. The init arguments are as follows.

`lr` : learning rate. A good heuristic is to start `lr` at `4 / u_scaling` and then decrease exponentially during training.

`u_scaling` : constant value $u$ such that we maintain $\forall S: \sum_{i \in S} (w_i^+ + w_i^-) = u$. The set $S$ can be a neuron, each connection, or each variable, depending on the setting of `norm_per`. In other words, for partitions of the weights specified by `norm_per`, the sum of weights over each such partition is always rescaled to equal `u_scaling`.

`norm_per` : whether to do per-neuron u-rescaling. If set to `None`, then the weights will not be rescaled. Otherwise, must be set to `neuron`, in which case the weights are partitioned into sets based on which neuron they are inputs to, and rescaled with respect to the sums across each set.

`gradient_clipping` : if set to `None`, then no gradient clipping will be performed. Otherwise, set this parameter to some value in order to clip all gradients to this value.

`weight_regularization` : if set to `None`, then no weight regularization will be performed. Otherwise, must be of the form (`reg_type`, `alpha`) where `reg_type` is one of {`entropy`, `l1`} and `alpha` is the regularization strength.

`plus_minus` : whether to use negative weights. If set to `True`, then $w_i = w_i^+ - w_i^-$, if `False` just $w_i = w_i^+$.

`init` : distribution from which to initialize the parameters of the model. Currently, the four options are `log_normal`, `gamma`, `uniform`, and `bootstrap`. The `bootstrap` initialization simply sets each positive and negative weight such that the resulting difference is the same as the pre-existing model parameters, and their sum is equal to $u$ divided by the number of parameters for that neuron.

### Paper

[1] Kivinen, J & Warmuth, M (1997). Additive versus exponentiated gradient updates for linear
prediction. Journal of Tnformation and Computation 132, 1-64. 
