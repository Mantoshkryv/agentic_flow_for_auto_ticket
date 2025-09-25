# Create: tg/templatetags/__init__.py (empty file)
# Create: tg/templatetags/url_extras.py
from django import template
from urllib.parse import urlencode

register = template.Library()

@register.simple_tag
def url_replace(request, field, value):
    query_params = request.GET.copy()
    query_params[field] = value
    return query_params.urlencode()