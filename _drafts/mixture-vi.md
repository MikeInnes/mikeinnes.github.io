---
layout: post
title: Variational Inference
---

The goal of statistical inference is to tell us things that we didn't know based on things we do. The things want to learn about could be anything – [phylogenetic trees](https://www.beast2.org), for example – but usually we'll talk about numerical summaries, like the GDP of France, or the reproductive rate of an infectious disease. In mathematical notation we'll label the set of numbers we don't know [[\theta \in \mathbb R^n]] (as in, [[\theta]] is a list of [[n]] unknown numbers) and the numbers we do know [[X \in \mathbb R^m]].

However, we don't usually expect a single right answer, like "[[\theta]] is [[5]]". Instead we want a range of plausible answers, like "[[\theta]] is probably [[5 \pm 3]]". To do this we use a function [[P(\theta)]] which takes a candidate for [[\theta]] and tells us how plausible it is – a probability distribution. When we see new data [[X]], we want to update what we currently think is plausible [[P(\theta)]] (aka the prior) to a new distribution [[Q(\theta)]] (the posterior). [[Q]] becomes our new [[P]] and we can rinse and repeat as new information comes in.

Bayes' rule makes the process of getting [[Q]] from [[P]] look simple:

$[[Q(\theta) = \frac{P(X,\theta)}{P(X)}]]

[[P(X, \theta)]] is very easy to get, in most cases. It's the function we write down when we use a probabilistic programming language (PPL) to describe how [[X]] and [[\theta]], the knowns and unknowns, are related. But [[P(X)]], the marginal likelihood of the data, turns out to be much harder. To get it from [[P(X, \theta)]] we'd have to integrate over all [[\theta]]:

$[[P(X) = \int P(X,\theta)\,d\theta]]

We often have thousands of [[\theta]] parameters, which makes this integral impossible. That means we need a workaround. One option is to use MCMC, which lets us sample from [[Q(\theta)]] without knowing [[P(X)]]. Another is variational inference, or VI, in which we try to turn this tricky integral into an optimisation problem instead.

## Neat Little Rows

To frame inference as optimisation, we can make up a candidate distribution [[\hat Q(\theta)]] which has parameters we can tune, and then try to minimise the difference [[D(\hat Q, Q)]] between the candidate [[\hat Q]] and the true posterior [[Q]]. (For example, we might assume all the [[\theta_i]] are independent Gaussians, and we can tweak each mean and variance to improve the fit.)

There are a few ways to measure distance between distributions, but one particularly useful one is the KL divergence, defined as

<div>
    $[[\begin{aligned}
        D_{KL}(\hat Q, Q) &= \int \hat Q(\theta) \log(\frac{\hat Q(\theta)}{Q(\theta)}) \,d\theta \\
        &= E_{\hat Q}[ \log(\frac{\hat Q(\theta)}{Q(\theta)}) ]
    \end{aligned}]]
</div>

where [[E_{\hat Q}[f(\theta)]]] means the expected value of [[f(\theta)]], given [[\theta \sim \hat Q]]. The divergence is always positive, and if [[D_{KL} = 0]] then [[\hat Q = Q]].

At first this must look circular; we can find [[Q]] if we minimise a value that we need [[Q]] to calculate. But some magic happens if we replace [[Q]] with its definition per Bayes' rule.

<div>
    $[[\begin{aligned}
        D_{KL}(\hat Q, Q) &= E_{\hat Q}[ \log(\frac{\hat Q(\theta) P(X)}{P(X, \theta)}) ] \\
        &= E_{\hat Q}[ \log(\hat Q(\theta)) - \log(P(X, \theta)) + \log(P(X)) ] \\
        &= E_{\hat Q}[ \log(\hat Q(\theta)) - \log(P(X, \theta)) ] + \log(P(X))
    \end{aligned}]]
</div>

The trick here is that we can separate out [[\log(P(X))]], the part that's hard to calculate. Because that term is independent of [[\theta]], minimising [[D_{KL}]] is the same as minimising

<div>
    $[[\begin{aligned}
        D_{KL}(\hat Q, Q) - \log(P(X)) &= E_{\hat Q}[ \log(\hat Q(\theta)) - \log(P(X, \theta)) ]
    \end{aligned}]]
</div>

which is easy to work out. Another nice feature of this approach is that it gives us a lower bound on the model evidence [[P(X)]], because [[D_{KL}]] must be positive.

$[[E_{\hat Q}[ \log(P(X, \theta)) - \log(\hat Q(\theta))] \le \log(P(X))]]

(Note the sign flip; we'll maximise this quantity to minimise [[D_{KL}]].)

For this reason, the left hand side is often called the "evidence lower bound" or ELBO. This is a particularly useful feature of VI, since [[P(X)]] gives a measure of model fit that can be used for model comparison and hypothesis testing. Maximising the ELBO will both improve our model and give an idea of how good the fit is.

Calculating the ELBO still involves an integral, but because it's an expectation we can approximate it by taking samples from [[\hat Q]].

$[[E\_{\hat Q}[f(\theta)] \approx \frac{1}{N}\sum_{\theta_i \sim \hat Q}^N f(\theta_i)]]

This means that our objective function – what we minimise by gradient descent – will be noisy rather than deterministic. But that's OK, because deep learning has given us excellent noisy optimisers, like `Adam`. In fact, these optimisers are so good that we can set [[N = 1]] and estimate the ELBO as we improve it.[^batch] We just have to maximise the objective

```julia
function objective(Q̂)
  θ = rand(Q̂)
  model(θ) - logpdf(Q̂, θ)
end
```

where `model(θ)` represents the user-defined log likelihood function (closing over the data [[X]]).

[^batch]: [[N = 1]] usually needs the fewest samples to converge, too. However, it can still be useful to set [[N > 1]]. Extra samples in a batch are often cheap to compute.

## Grounds for Divorce

It's worth breaking down this objective a bit, and interesting to compare it to neural networks and deep learning.

$[[\log(P(X, \theta)) - \log(\hat Q(\theta))]]

The first term implies that we want to maximise how likely we think the data is, which seems reasonable enough. This term is often identical in neural networks. For example, the negative log likelihood of a Gaussian increases with [[\|\|y - \hat y\|\|^2]] (comparing our data [[y]] and model output [[\hat y]]), which you might recognise as the mean squared error. Likewise the cross entropy, used for discrete [[y]], is the log likelihood of the multinomial distribution. So typical deep learning is secretly maximum likelihood estimation on a statistical model, with implicit distributional assumptions.

The second term is more surprising. We want [[\theta]] to be as *unlikely* as possible, with respect to [[\hat Q]]. Unlike deep learning, which would find a single point estimate for [[\theta]], we get a distribution of plausible [[\theta]] values, and this term forces that distribution to be as spread out as possible. So it's effectively a kind of regularisation, preventing us from overfitting. The two terms of this objective are like attractive and repulsive forces, pulling our estimates towards the posterior mode but also repelling them from each other, making model fitting an energy-minimisation problem.[^forces] That lets us estimate the posterior mean, which is less likely to be an [artifact of noise in the data](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation#Limitations) than the mode.

[^forces]: [This demo](https://chi-feng.github.io/mcmc-demo/app.html?algorithm=SVGD&target=banana&delay=0) gives a striking visualisation of this, using a different but related VI-based inference method.

Here's a more information-theoretical way to view the ELBO. [[E_{\hat Q}[ \log(P(X, \theta))]]] is the negentropy of our data with respect to the model, or in other words, the number of bits of information in the data not explained by the model. [[E_{\hat Q}[\log(\hat Q(\theta))]]] is the negentropy of [[\hat Q]], or in other words the amount of information stored by our model. Our cost function implies that the model will only add complexity if it is [more than offset](https://en.wikipedia.org/wiki/Occam%27s_razor) by better explanation of the data. So the model compresses information in the data and changes in the ELBO measure how many bits are saved; information that can be compressed is likely to be pattern rather than noise.

## Little Fictions

That's all well and good in theory, but we need to know how to set up [[\hat Q]]. The most common option is to assume each parameter in [[\theta]] is distributed as an independent Gaussian (normal distribution).

$[[\theta_i \sim \mathcal{N}(\mu_i,\sigma_i)]]

We can also write this as a single multivariate Gaussian

$[[\boldsymbol \theta \sim \mathcal N(\boldsymbol \mu, \boldsymbol \sigma I)]]

where [[\boldsymbol \sigma I]] is the [[n]]-dimensional covariance matrix (with only diagonal elements).[^correlation]

[^correlation]: We could, of course, fit the full correlation matrix. But since the size of that matrix increases with [[n^2]] it's not feasible for large numbers of parameters.

More explicitly, [[\hat Q(\boldsymbol \theta)]] will be given by

$[[\hat Q(\boldsymbol \theta) = p(\boldsymbol \theta \| \boldsymbol \theta \sim \mathcal N(\boldsymbol \mu, \boldsymbol \sigma I))]]

where [[p(x \| x \sim D)]] gives the probility density of the distribution [[D]]. It's easy to sample from this distribution (just sample from each normal independently), it's easy to calculate the ELBO, and it's easy to update each [[\mu]] and [[\sigma]] with gradient descent.

This works great in many cases, but remember the assumption it depends on: that all parameters are independent of each other in the posterior. It's useful to see what happens when we break this assumption. Consider what happens if we write this in Stan:

<div>
    $[[\begin{aligned}
        x &\sim \mathcal N(0, 1) \\
        y &\sim \mathcal N(0, 1) \\
        x + y &\sim \mathcal N(0, 1)
    \end{aligned}]]
</div>

The posteriors of [[x]] and [[y]] are tightly correlated. Here's what the posterior looks like, compared to what we calculate using VI.

[vi plot]

Notice that because [[x]] and [[y]] are treated as independent, the posterior plot can only be a horizontal or vertical elipse (or be circular). We simply can't represent the slanted shape of the true posterior here. Instead we end up with a very small circle, because making the disc any bigger would include more unlikely values of [[x]] and [[y]] (eg where [[x + y = -1]]) than likely ones.

The result is that when our parameters are correlated, we really underestimate their marginal uncertainty – fatal for many statistical tasks. With lots of correlated parameters the method reduces to MAP estimation (finding the [[\theta]] that maximises the likelihood of the data, while igoring the entropy term). In these cases, our model will both overfit and falsely claim very high confidence in its results.

If our representation of the posterior distribution [[\hat Q]] is to restrictive, perhaps we can make it more expressive.

## The Fix

One simple way to do so is to use the idea of a mixture model, in which a more complex distribution is made up of a combination of simpler ones. Consider the two peaks in the distribution of adult human heights, for example. The overall chance of any given height is the average of the (Gaussian) probability a man has that height, and that a woman does.

[graphic?]

In our case, [[\hat Q]] will be a mixture of [[N]] distributions of the kind we described above: a set of [[n]] independent Gaussians. The [[i]]th component has a set of means [[\boldsymbol \mu^i]] and deviations [[\boldsymbol \sigma^i]].

$[[\hat Q(\boldsymbol \theta) = \frac{1}{N}\sum_{i=1}^N p(\boldsymbol \theta \| \boldsymbol \theta \sim \mathcal N(\boldsymbol \mu^i, \boldsymbol \sigma^i I))]]

I think this idea is much easier to understand graphically than through mathematical symbols. Here's how our posterior for the [[x + y]] example above looks as [[N]] increases; we can see that [[N=1]] is equivalent to what we did earlier, and we can see that it starts to look much more like the true posterior.

[plot]

One way to look at the mean-field approximation is that we represent a distribution with a single representative [[n]]-dimensional point at [[\mu]], with some uncertainty over the exact location. The mixture approach generalises this by working with [[N]] representative points (not entirely unlike a sample) along with some fuzz around them. Components are repelled from each other by the ELBO: those that are too close will generate samples that have a high log-likelihood with respect to their neighbours, which gradient descent will try to avoid.

In principle the mixture model can approximate any other distribution, given enough components. Whether it works well in practice will still, of course, depend on the details of the problem.

## Bitten by the Tailfly

Unfortunately, [some time series models]() will defeat most inference methods, including the one detailed above. Very high dimensionality and correlation between parameters seem to combine to make the posterior very sparse, so points that look close together (for example, two criss-crossing lines) register as being far apart. The result is again very tight marginals around the MAP estimate.

One last trick is to take the idea of a set of samples repelling each other more literally. To work backwards, imagine we already had a good set of samples via MCMC. MCMC doesn't give us a distribution or a PDF, but with the samples we can estimate that underlying function (using a kernel density estimator, for example). Doing this will take sandpaper to the fine peaks and troughs of the true posterior, but that's fine if we're primarily interested in the marginal uncertainty of individual parameters.

So instead of starting with a guess for the posterior distribution [[\hat Q]], start with a guess for a set of samples from the posterior; estimate the distribution implied by those samples; use both to calculate the ELBO; then improve the ELBO by tweaking the samples. The net result should be a representative[^random] set of points that are good enough for getting marginals.

[^random]: Representative, but not random. Compared to a random sample, the points will look unnaturally spread out. The upside is that fewer points are needed to cover the posterior space well.

We'll need a way to calculate the probability density function, or PDF. The word "density" is not coincidental; it's the expected number of particles in a unit volume. Given two particles at a Euclidean distance of [[d]] from each other, we can carve around each point a hypersphere[^shape] (with volume proportional to [[d^n]]) and the density is [[1/d^n]]. We can average the density over all pairs of points[^harmonic] to get he density at a point [[\theta_i]]:

$[[\rho(\theta_i) = \frac{1}{N d^n} \sum_{j \ne i}^N \|\|\theta_i-\theta_j\|\|]]

And so the ELBO, estimated using the samples that we have, is:

$[[\frac{1}{N} \sum_i^N \log(P(X,\theta_i)) - \log(\rho(\theta_i))]]

[^shape]: The choice of size and shape may seem pretty arbitrary. It doesn't matter: because we want _log_ density, factors like [[\frac{3}{4}\pi]] get absorbed into an additive constant that doesn't affect how we optimise the ELBO. Only the way that volume scales with distance matters.

[^harmonic]: Equivalently, we take the harmonic mean of all pairwise distances and then compute volume and density. The (harmonic) average distance will be close to the smallest pairwise distance.

In this modeller's testing so far, this seems to work quite well. Here's what the resulting samples look like on our motivating example; notice that at [[N=1]] it handily reduces to MAP again.

[plot]

Even on tricky high-dimensional examples, we seem to either get the right marginals or overestimate them by a small factor – far better than underestimating them. When all else fails, variational inference is an extremely useful trick to have up your sleeve.

An [implementation]() of the particle based method is available, along with an implementation of the [time series model it powered]().

## Notes
