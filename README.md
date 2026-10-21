# Scan 3D — luvas.io

Phone 3D scanner: record a slow walk around an object or a room and the phone builds a textured
3D model on the device. The phone provides the camera frames and its own metric, gravity-aligned
pose (WebXR / ARCore); the vision pipeline does the rest — ORB features, epipolar-guided
matching, triangulation, bundle adjustment with the odometry as priors, PatchMatch multi-view
stereo and TSDF fusion on WebGPU, surface nets meshing and a seam-levelled texture atlas. The app
shows every pipeline stage live with its timing, a measuring tool and diagnostics, and exports
GLB / PLY / JSON. Full write-up (in Portuguese) in [frontend/README.md](frontend/README.md).

Runs on Chrome for Android on ARCore phones with WebGPU.

## Structure

```
api/            Express server that serves the compiled app (deploy stage)
frontend/       Angular 22 application
```

## Development

```sh
cd frontend
npm install --legacy-peer-deps
npm start             # dev server on http://localhost:4200
npm run start:https   # HTTPS on the LAN, needed to open it on the phone
npm run build         # production build to dist/v1/browser
npm test -- --watch=false && npm run verify && npm run e2e
```

## Deploy

Same standard as the other luvas.io apps (multi-stage Docker build, Express serving the compiled
bundle):

```sh
docker compose up --build -d   # serves on port 7130
```

WebXR and the camera only work in a secure context, so in production the container must sit
behind the HTTPS reverse proxy.
