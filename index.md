---
layout: page
title: home
---

# Mike J Innes

## Some things you should know

* My name is Mike
* I know very little about anything (but I'm working on it)
* As a programmer, I spend most of my time writing pieces of text which solve problems
  * Although the problems are often caused by other pieces of text, which were often also written by me

## Some things I've worked on

* [Julia](https://julialang.org/), a programming language
* [Flux](https://en.wikipedia.org/wiki/Flux_(machine-learning_framework)), a machine learning system
* [Zygote](https://github.com/FluxML/Zygote.jl), an algorithmic differentiation engine
* [Juno](http://junolab.org), a live-coding IDE
* [Various miscellaneous packages](https://github.com/MikeInnes/)

## Some things you can click on

[GitHub](https://github.com/MikeInnes), [Twitter](https://twitter.com/MikeJInnes), [Publications](https://scholar.google.co.uk/citations?user=zffDj88AAAAJ&hl=en), [Blog]({{site.url}}), [Email](mailto:mike.j.innes@gmail.com)

[Addresses, keys etc.](keys.html)

## Some things you can read

<ul>
{% for post in site.posts %}
<li>
  <a href="{{post.url}}">{{post.title}}</a> ({{post.date | date: '%B %Y' }})
</li>
{% endfor %}
</ul>

[RSS feed](/feed.xml)
