---
layout: default
title: Workshops
---

# Workshops

<ul>
  {% assign files = site.static_files | where_exp: "f", "f.path contains '/workshops/'" %}
  {% for f in files %}
    {% unless f.name == "index.md" or f.name == "index.html" %}
      <li><a href="{{ f.path | relative_url }}">{{ f.name }}</a></li>
    {% endunless %}
  {% endfor %}
</ul>
