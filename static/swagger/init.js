// Bootstrap Swagger UI against our OpenAPI spec. Kept in a separate file so
// the page needs no inline scripts and Content-Security-Policy stays locked
// to 'self'.
window.addEventListener('DOMContentLoaded', () => {
  window.SwaggerUIBundle({
    url: '/openapi.json',
    dom_id: '#swagger-ui',
    deepLinking: true,
    showExtensions: true,
    showCommonExtensions: true,
    presets: [
      window.SwaggerUIBundle.presets.apis,
      window.SwaggerUIBundle.SwaggerUIStandalonePreset,
    ],
    layout: 'BaseLayout',
  });
});
