---
layout: post
title: "Archive"
author: MMA
social: true
comments: false
permalink: /archive/
years:
- 2020
- 2019
- 2018
---

{% for year in page.years %}
{% assign y1 = year | plus: 0 %}
# {{ y1 }}
<ul>
{% for post in site.posts %}
{% assign y2 = post.date | date: '%Y' | plus: 0 %}
{% if y1 == y2 %}
<li style="line-height:1.5em">{{ post.date | date:"%d %b" }} &middot; <a href="{{ post.url }}" target="_blank">{{ post.title }}</a></li>
{% endif %}
{% endfor %}
</ul>
{% endfor %}
