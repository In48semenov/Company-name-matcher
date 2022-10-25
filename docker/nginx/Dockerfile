FROM nginx:1.19.2-alpine

ARG MLFLOW_USER
ARG MLFLOW_PASSWORD

RUN apk update && apk add apache2-utils
RUN htpasswd -bc /usr/.htpasswd ${MLFLOW_USER} ${MLFLOW_PASSWORD}