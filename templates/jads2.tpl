{%- extends 'slides_reveal.tpl' -%}

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
  width: 1000px !important;
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
})
</script>
{%- endblock header -%}
