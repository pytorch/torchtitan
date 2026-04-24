{{ fullname | smart_fullname | escape | underline}}

.. *Based on custom template* `class.rst`

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:
   :exclude-members: __init__, __new__

   {# :no-index: 
      with this we can get that the page with the class (named after the class),
      gets directly a link to the start of the class definition in the 
      navigation bar. But this also breaks auto-generated links to this page #}

   {% block attributes %}
   {% if all_attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in all_attributes %}
      {% if not item.startswith('__') %}
         ~{{ name }}.{{ item }}
      {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block methods %}
   {% if all_methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :nosignatures:
   {% for item in all_methods %}
      {% if not item.startswith('__') %}
         ~{{ name }}.{{ item }}
      {% endif %}
   {%- endfor %}
   {% endif %}

   .. automethod:: __init__

   {% for item in all_methods %}
      {% if not item.startswith('__') and item != '__init__' %}
   .. automethod:: {{ name }}.{{ item }}
      {% endif %}
   {% endfor %}

   {% endblock %}