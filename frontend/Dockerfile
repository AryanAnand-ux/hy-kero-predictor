# Stage 1: Build React Application
FROM node:20-alpine AS build

WORKDIR /app

# Copy dependency files
COPY frontend/package*.json ./
RUN npm ci

# Copy source files
COPY frontend/ ./

# Build production bundle
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

# Stage 2: Serve using Nginx
FROM nginx:stable-alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
