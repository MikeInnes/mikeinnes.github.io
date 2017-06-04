---
layout: page
title: home
---

# Mike J Innes

## Some things you should know about me

* My name is Mike
* I know very little about anything (but I'm working on it)
* I'm a semi-effective code monkey, currently monkeying around for [Julia Computing](https://juliacomputing.com)
* As a programmer, I spend most of my time writing pieces of text which solve problems
  * Although the problems are often themselves caused by other pieces of text, which were often also written by me

## Some things I've worked on

* [The Julia Language](https://julialang.org/)
* [Juno](http://junolab.org), an IDE for Julia
* [Various miscellaneous packages](https://github.com/MikeInnes/)

## Some things you can click on

[GitHub](https://github.com/MikeInnes), [Twitter](https://twitter.com/one_more_minute), [Here]({{site.url}})

## Some things you can read

<ul>
{% for post in site.posts %}
<li>
  <a href="{{post.url}}">{{post.title}}</a>
</li>
{% endfor %}
</ul>
