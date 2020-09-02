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
  display: none;
//  margin-top: 5px;
}
.reveal.slide .slides{
  width: 1200px !important;
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
  min-height: 727px !important;
  position: relative !important;
  width: 1200px !important;
  height: 100%;
}
.pdf-page {
  height: 727px !important;
}
.output_subarea {
  padding: 0px !important;
}
.reveal section img {
  margin: 0 !important;
}
.print-pdf section {
  left: 0px;
}
.print-pdf .prompt{
  min-width: 0px !important;
}
.rendered_html ul:not(.list-inline){
  padding-left: 0px !important;
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
