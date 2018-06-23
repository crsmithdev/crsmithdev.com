FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY public /usr/share/nginx/html
RUN mkdir -p /etc/nginx/ssl/blog
VOLUME /etc/nginx/ssl/blog
EXPOSE 443
