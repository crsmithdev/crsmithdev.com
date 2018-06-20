# Build

```
sudo docker build -t crsmithdev/blog .
```

# Upload

```
sudo docker login
sudo docker push crsmithdev/blog
```

# Run

```
sudo docker run --rm -v <SSL_DIR>:/etc/nginx/ssl/blog:ro -p 80:80 -p 443:443 crsmithdev/blog
```

# Inspect

```
sudo docker run --rm -it -v <SSL_DIR>:/etc/nginx/ssl/blog:ro -p 80:80 -p 443:443 crsmithdev/blog /bin/sh
```
