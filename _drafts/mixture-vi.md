---
layout: post
title: Mixture Variational Inference
---

## Introducing VI

The main concern of statistical inference is to tell us things that we didn't know based on things we do know. The things want to learn about could be anything – phylogenetic trees, for example – but usually we'll talk about numerical summaries, like the GDP of France, the average heights of males and females, or the reproductive rate of an infectious disease. We'll label the set of numbers we don't know [[\theta \in \mathbb R^n]], and the numbers we do know [[X \in \mathbb R^m]].

However, we don't usually expect a single right answer, like "[[\theta]] is [[5]]". Instead we want a range of plausible answers, like "[[\theta]] is probably [[5 \pm 3]]". To do this we use a function [[P(\theta)]] which takes a candidate for [[\theta]] and tells us how plausible it is – a probability distribution. When we see new data [[X]], we want to update what we currently think is plausible, [[P(\theta)]], to a new distribution [[Q(\theta)]]. [[Q]] becomes our new [[P]] and we can rinse and repeat.

Bayes' rule makes the process of getting [[Q]] from [[P]] look deceptively simple:

$[[Q(\theta) = \frac{P(X,\theta)}{P(X)}]]

[[P(X, \theta)]] is very easy to get, in most cases. It's the function we write down when we use a PPL. But [[P(X)]], the marginal likelihood of the data, turns out to be much harder. To get it from [[P(X, \theta)]] we'd have to integrate over all [[\theta]]:

$[[P(X) = \int P(X,\theta)\,d\theta]]

We often have thousands of [[\theta]] parameters, which makes this integral unfeasible. That means we need a workaround. One option is to use MCMC, which lets us sample from [[Q(\theta)]] without knowing [[P(X)]]. Another is variational inference, or VI, in which we try to turn this tricky integral into an optimisation problem instead.

## KL-Pax

To frame this problem as optimisation, we can make up a candidate distribution [[\hat Q(\theta)]] which has parameters we can tune, and then try to minimise the difference [[D(\hat Q, Q)]] between the candidate [[\hat Q]] and the true distribution [[Q]]. (For example, we might assume all the [[\theta]] are independent Gaussians, and we can tweak each mean and variance to reduce the difference.)

There are a few ways to measure distance between distributions, but one useful one is the KL divergence, defined as

<div>
    $[[\begin{aligned}
        D_{KL}(\hat Q, Q) &= \int \hat Q(\theta) \log(\frac{\hat Q(\theta)}{Q(\theta)}) \,d\theta \\
        &= E_{\hat Q}[ \log(\frac{\hat Q(\theta)}{Q(\theta)}) ]
    \end{aligned}]]
</div>

where [[E_{\hat Q}[f(\theta)]]] means the expected value of [[f(\theta)]], given [[\theta \sim \hat Q]]. The divergence is always positive, and if [[D_{KL} = 0]] then [[\hat Q = Q]].

At first this must look like begging the question; we can find [[Q]] if we minimise a value that we need [[Q]] to calculate. But some magic happens if we replace [[Q]] with its definition per Bayes' rule.

<div>
    $[[\begin{aligned}
        D_{KL}(\hat Q, Q) &= E_{\hat Q}[ \log(\frac{\hat Q(\theta) P(X)}{P(X, \theta)}) ] \\
        &= E_{\hat Q}[ \log(\hat Q(\theta)) - \log(P(X, \theta)) + \log(P(X)) ] \\
        &= E_{\hat Q}[ \log(\hat Q(\theta)) - \log(P(X, \theta)) ] + \log(P(X))
    \end{aligned}]]
</div>

The trick here is that we can separate out [[\log(P(X))]], the part that's hard to calculate. And because it's independent of [[\theta]], we can ignore it entirely. To minimise [[D_{KL}]] we just need to minimise the expectation of [[\log(\hat Q(\theta)) - \log(P(X, \theta))]], which is easy to work out. As a bonus, because [[D_{KL} \ge 0]], we also get a lower bound on [[P(X)]]. For this reason the left quantity is often called the "evidence lower bound" or ELBO.[^elbo]

[^elbo]: It's common to think of the ELBO as approximating what the model evidence would be if there were no restrictions on the posterior. Another interpetation is that VI's constraints are *assumptions* of the model, which [[Q]] must satisfy. Then [[D_{KL}]] can tend to [[0]] and the ELBO to the true model evidence. The upshot is that model comparison is still valid when using VI; it's just that a similar model might be better when inferred using MCMC than VI, because those assumptions get implicitly relaxed.

$[[E_{\hat Q}[ \log(P(X, \theta)) - \log(\hat Q(\theta))] \le \log(P(X))]]

The expectation still involves an integral, but we can easily approximate it by taking samples from [[\hat Q]].

$[[E\_{\hat Q}[f(\theta)] \approx \frac{1}{N}\sum_{\theta_i \sim \hat Q}^N f(\theta_i)]]

This means that our objective function – what we minimise by gradient descent – will be noisy. But that's OK, because deep learning has given us excellent noisy optimisers, like `Adam`. In fact, these optimisers are so good that we can set [[N = 1]] and estimate the ELBO as we improve it.[^batch] We just have to maximise the objective

[^batch]: [[N = 1]] usually needs the fewest samples to converge, too. However, it can still be useful to set [[N > 1]]. Extra samples in a batch can be cheap to compute, especially on GPUs, where they can be more-or-less free.

```julia
function objective(Q̂)
  θ = rand(Q̂)
  model(θ) - logpdf(Q̂, θ)
end
```

where `model(θ)` represents the user-defined log likelihood function (closing over the data [[X]]).

## At What Cost?

It's worth breaking down this objective a bit, and interesting to compare it to neural networks and deep learning.

$[[\log(P(X, \theta)) - \log(\hat Q(\theta))]]

The first term implies that we want to maximise how likely we think the data is, which seems reasonable enough. This term is often identical in neural networks. For example, the negative log likelihood of a Gaussian increases with [[\|\|y - \hat y\|\|^2]] (comparing our data and model output), which you might recognise as the mean squared error. Likewise the cross entropy, used for discrete [[y]], is the log likelihood of the multinomial distribution. So typical deep learning is secretly MAP estimation on a statistical model, with implicit distributional assumptions.

The second term is a bit less obvious. We want [[\theta]] to be as *unlikely* as possible, with respect to [[\hat Q]]. Unlike deep learning, which would find a point estimate for [[\theta]] directly, we get a distribution of plausible [[\theta]] values, and this term forces this distribution to be as spread out as possible. So it's effectively a kind of regularisation, preventing us from overfitting. The two terms of this objective are like attractive and repulsive forces, pulling our estimates towards the distribution mode but also repelling them from each other, and causing them to settle in the peripheries.[^forces] That lets us estimate the posterior mean, which is less likely to be an artifact of noise in the data than the mode.

[^forces]: [This demo](https://chi-feng.github.io/mcmc-demo/app.html?algorithm=SVGD&target=banana&delay=0) gives a striking visualisation of this, using a different but related VI-based inference method.

Here's a more information-theoretical way to view the ELBO. [[E_{\hat Q}[ \log(P(X, \theta))]]] is the negentropy of our data with respect to the model, or in other words, the number of bits of information in the data not explained by the model. [[E_{\hat Q}[\log(\hat Q(\theta))]]] is the negentropy of [[\hat Q]], or in other words the amount of information stored by our model. Our cost function implies that the model will only add complexity if it is more than offset by better explanation of the data. So the model compresses information in the data and changes in the ELBO measure how many bits are saved; information that can be compressed is likely to be pattern rather than noise.

## Pleasant pastures, mean fields

<div>
    $[[\begin{aligned}
        \theta_i &\sim \mathcal{N}(\mu_i,\sigma_i) \\
        \boldsymbol \theta &\sim \mathcal N(\boldsymbol \mu, \boldsymbol \sigma I) \\
        \hat Q(\boldsymbol \theta) &= p(\boldsymbol \theta | \boldsymbol \theta \sim \mathcal N(\boldsymbol \mu, \boldsymbol \sigma I))
    \end{aligned}]]
</div>

$[[\hat Q(\boldsymbol \theta) = \frac{1}{N}\sum_{i=1}^N p(\boldsymbol \theta \| \boldsymbol \theta \sim \mathcal N(\boldsymbol \mu_i, \boldsymbol \sigma_i I))]]

## Notes
