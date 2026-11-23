FROM node:20-alpine AS builder
WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

ARG VITE_API_BASE_URL=http://localhost:8000
ARG VITE_STREAM_URL=http://localhost:8001
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
ENV VITE_STREAM_URL=${VITE_STREAM_URL}

RUN npm run build

FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

