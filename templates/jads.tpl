{%- extends 'slides_reveal.tpl' -%}

{% block input_group -%}
<div class="input_hidden">
{{ super() }}
</div>
{% endblock input_group %}

{%- block header -%}
{{ super() }}

<script src="//ajax.googleapis.com/ajax/libs/jquery/1.10.2/jquery.min.js"></script>

<style type="text/css">
//div.output_wrapper {
//  margin-top: 0px;
//}
.input_hidden {
  display: hidden;
//  margin-top: 5px;
}
.input {
  display: none !important;
//  margin-top: 5px;
}
.reveal.slide .slides{
  width: 1000px !important;
  height: 100%;
}
.reveal.slide .slides > section, .reveal.slide .slides > section > section {
  display: flex !important;
  flex-direction: column !important;
  justify-content: center !important;
  position: absolute !important;
  top: 0 !important;
  font-size: 110% !important;
}
.rendered_html > h1, .rendered_html > h2, .rendered_html > h3{
//  position: absolute !important;
  top: 0 !important;
  left: 0 !important;
  right: 0 !important;
  padding-bottom: 20px;
  margin-top: 0px !important;
}
.rendered{
  flex-direction: inherit !important;
}
.print-pdf .reveal.slide .slides > section, .print-pdf .reveal.slide .slides > section > section {
  min-height: 770px !important;
  position: relative !important;
}
</style>

<script>
$(document).ready(function(){
  $(".output_wrapper").click(function(){
      $(this).prev('.input_hidden').slideToggle();
  });
  $(".output_subarea").css('max-width','100%');
})
</script>
{%- endblock header -%}
