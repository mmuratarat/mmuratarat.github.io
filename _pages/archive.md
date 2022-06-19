---
layout: post
title: "Archive"
author: MMA
social: true
comments: false
permalink: /archive/
---
{% assign current_year = 'now' | date: "%Y" %}
current_year {{ current_year }}
{% for year in (2000..current_year) reversed %}
  {% assign y1 = year | plus: 0 %}
  {% assign posts_url_array = "" | split: ',' %}
  {% assign posts_date_array = "" | split: ',' %}
  {% assign posts_title_array = "" | split: ',' %}
  {% for post in site.posts %}
    {% assign y2 = post.date | date: '%Y' | plus: 0 %}
    {% if y1 == y2 %}
      {% assign a = post.url %}
      {% assign b = post.date | date:"%d %b" %}
      {% assign c = post.title %}
      {% assign posts_url_array = posts_url_array | push: a %}
      {% assign posts_date_array = posts_date_array | push: b %}
      {% assign posts_title_array = posts_title_array | push: c %}
    {% endif %}
  {% endfor %}
  {% if posts_url_array.size > 0 %}
# {{ y1 }}
<ul>
    {% assign till = posts_url_array.size | minus: 1 %}
    {% for i in (0..till) %}
<li style="line-height:1.5em"> {{ posts_date_array[i] }} &middot; <a href="{{ posts_url_array[i] }}" target="_blank">{{ posts_title_array[i] }}</a></li>
    {% endfor %}
</ul>
  {% endif %}
{% endfor %}
