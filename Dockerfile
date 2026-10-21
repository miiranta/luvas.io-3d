# Stage 1: Build Angular app
FROM node:22-alpine AS build
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --legacy-peer-deps
COPY frontend/ .
RUN npm run build

# Stage 2: Run Node API, serve compiled Angular
FROM node:22-alpine
WORKDIR /app
COPY api/package.json .
RUN npm install
COPY api/ .
COPY --from=build /build/dist/v1/browser ./dist/v1/browser

EXPOSE 7130
CMD ["node", "app.js"]
