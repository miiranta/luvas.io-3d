// End-to-end run of the production build in headless Chrome with a fake WebXR runtime
// (tools/fake-xr.ts): start AR, record two orbits of the synthetic scene, build the model at the
// default (High) quality, look at every layer, measure, export and score the exported GLB against
// the true scene; then rebuild at Medium and Low from the pipeline card and score those too.
import { build } from 'esbuild';
import { createServer } from 'node:http';
import {
    existsSync,
    mkdirSync,
    readdirSync,
    readFileSync,
    rmSync,
    statSync,
    writeFileSync,
} from 'node:fs';
import { extname, join, resolve } from 'node:path';
import { execSync } from 'node:child_process';
import { openChrome, watchConsole } from './devtools.mjs';

const DIST = 'dist/v1/browser';
const OUT = resolve(process.env.E2E_DIR ?? '.e2e-results');
const PORT = Number(process.env.E2E_PORT ?? 4392);
const RECORD_SECONDS = Number(process.env.RECORD_SECONDS ?? 33);
const BUILD_TIMEOUT_MS = 6 * 60_000;

if (!existsSync(join(DIST, 'index.html')) || process.env.E2E_BUILD)
    execSync('npx ng build', { stdio: 'inherit' });
rmSync(OUT, { recursive: true, force: true });
mkdirSync(join(OUT, 'downloads'), { recursive: true });

const fake = await build({
    entryPoints: ['tools/fake-xr.ts'],
    bundle: true,
    format: 'iife',
    platform: 'browser',
    target: 'es2022',
    write: false,
});
await build({
    entryPoints: ['tools/synthetic-scene.ts'],
    bundle: true,
    format: 'esm',
    platform: 'node',
    outfile: join(OUT, 'scene.mjs'),
    logLevel: 'warning',
});
const scene = await import(join(OUT, 'scene.mjs'));

const types = {
    '.js': 'text/javascript',
    '.css': 'text/css',
    '.html': 'text/html',
    '.ico': 'image/x-icon',
};
const server = createServer((request, response) => {
    const path = decodeURIComponent(new URL(request.url, 'http://localhost').pathname).slice(1);
    const file =
        path && existsSync(join(DIST, path)) && statSync(join(DIST, path)).isFile()
            ? path
            : 'index.html';
    response.writeHead(200, { 'content-type': types[extname(file)] ?? 'application/octet-stream' });
    response.end(readFileSync(join(DIST, file)));
}).listen(PORT);

const wait = (ms) => new Promise((done) => setTimeout(done, ms));
const results = [];
function check(name, pass, detail) {
    results.push({ name, pass });
    console.log(`${pass ? 'PASS' : 'FAIL'}  ${name} — ${detail}`);
}

const chrome = await openChrome({ name: 'e2e', port: 9445, windowSize: '412,915' });
const { call, evaluate, socket } = chrome;
const page = watchConsole(socket);

async function screenshot(name) {
    const { data } = await call('Page.captureScreenshot', { format: 'png' });
    writeFileSync(join(OUT, name), Buffer.from(data, 'base64'));
}

async function until(expression, timeoutMs, intervalMs = 250) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        const value = await evaluate(expression);
        if (value) return value;
        await wait(intervalMs);
    }
    return null;
}

const click = (selector) =>
    evaluate(
        `(() => { const el = document.querySelector(${JSON.stringify(selector)}); if (!el || el.disabled) return false; el.click(); return true; })()`,
    );
const text = (selector) =>
    evaluate(`document.querySelector(${JSON.stringify(selector)})?.textContent?.trim() ?? null`);

async function tap(x, y) {
    for (const type of ['mousePressed', 'mouseReleased']) {
        await call('Input.dispatchMouseEvent', {
            type,
            x,
            y,
            button: 'left',
            buttons: type === 'mousePressed' ? 1 : 0,
            clickCount: 1,
        });
    }
}

async function downloaded(extension, timeoutMs = 60_000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        const file = readdirSync(join(OUT, 'downloads')).find((name) => name.endsWith(extension));
        if (file) {
            const path = join(OUT, 'downloads', file);
            const size = statSync(path).size;
            await wait(400);
            if (statSync(path).size === size && size > 0) return path;
        }
        await wait(300);
    }
    return null;
}

/** Reads the triangle mesh, its UVs or vertex colours and the embedded JPEG from a GLB. */
function parseGlb(path) {
    const bytes = readFileSync(path);
    const jsonLength = bytes.readUInt32LE(12);
    const json = JSON.parse(bytes.subarray(20, 20 + jsonLength).toString('utf8'));
    const bin = bytes.subarray(20 + jsonLength + 8);
    const accessor = (index, Type, components) => {
        const a = json.accessors[index];
        const view = json.bufferViews[a.bufferView];
        const slice = bin.subarray(view.byteOffset, view.byteOffset + view.byteLength);
        return new Type(
            slice.buffer.slice(
                slice.byteOffset,
                slice.byteOffset + a.count * components * Type.BYTES_PER_ELEMENT,
            ),
        );
    };
    const mesh = json.meshes.find((m) => m.name === 'mesh').primitives[0];
    const image = json.images?.[0];
    const imageView = image && json.bufferViews[image.bufferView];
    return {
        json,
        positions: accessor(mesh.attributes.POSITION, Float32Array, 3),
        uvs:
            mesh.attributes.TEXCOORD_0 === undefined
                ? null
                : accessor(mesh.attributes.TEXCOORD_0, Float32Array, 2),
        colors:
            mesh.attributes.COLOR_0 === undefined
                ? null
                : accessor(mesh.attributes.COLOR_0, Uint8Array, 4),
        indices: accessor(mesh.indices, Uint32Array, 1),
        jpegBytes: imageView ? imageView.byteLength : 0,
    };
}

const percentile = (values, q) =>
    [...values].sort((a, b) => a - b)[Math.floor(q * (values.length - 1))];

/** Surface error of every third vertex of an exported mesh (m). */
function meshErrors(glb) {
    const errors = [];
    for (let i = 0; i < glb.positions.length; i += 3 * 3) {
        errors.push(
            scene.surfaceDistance([glb.positions[i], glb.positions[i + 1], glb.positions[i + 2]]),
        );
    }
    return errors;
}

/** Stage name, method line and time of every row of the (opened) pipeline card. */
const pipelineRows = () =>
    evaluate(`[...document.querySelectorAll('app-pipeline-track li')].map((li) => ({
        status: li.className,
        name: li.querySelector('.name').firstChild.textContent.trim(),
        method: li.querySelector('.name small')?.textContent.replace(/\\s+/g, ' ').trim() ?? '',
        time: li.querySelector('.time').textContent.trim(),
    }))`);

async function openTrack() {
    if (!(await evaluate(`!!document.querySelector('app-pipeline-track li')`)))
        await click('app-pipeline-track .head');
    await wait(300);
}

async function exportGlb() {
    for (const file of readdirSync(join(OUT, 'downloads')))
        rmSync(join(OUT, 'downloads', file), { force: true });
    await click('app-model-viewer button.save');
    await wait(300);
    await evaluate(
        `[...document.querySelectorAll('app-export-sheet .choice')].find((b) => b.querySelector('b').textContent.trim() === 'Model · GLB')?.click()`,
    );
    const path = await downloaded('.glb');
    await until(
        `![...document.querySelectorAll('app-export-sheet .choice')].some((b) => b.disabled)`,
        30_000,
    );
    await click('app-export-sheet .round-action.close');
    await wait(300);
    return path;
}

try {
    await call('Page.addScriptToEvaluateOnNewDocument', { source: fake.outputFiles[0].text });
    await call('Browser.setDownloadBehavior', {
        behavior: 'allow',
        downloadPath: join(OUT, 'downloads'),
    });
    await call('Emulation.setDeviceMetricsOverride', {
        width: 412,
        height: 915,
        deviceScaleFactor: 2,
        mobile: true,
    });
    await call('Page.navigate', { url: `http://localhost:${PORT}/` });

    const ready = await until(
        `(() => { const b = document.querySelector('app-capture-stage button.open'); return b && !b.disabled; })()`,
        20_000,
    );
    check(
        'app loads and the reconstruction worker gets WebGPU',
        !!ready,
        ready
            ? 'start button enabled'
            : `start button disabled: ${await text('app-capture-stage .toast')}`,
    );
    await screenshot('01-start.png');
    const chosen = await text('app-capture-stage .quality button.on');
    check(
        'quality selector defaults to High',
        chosen === 'High',
        `${chosen} · ${await text('app-capture-stage .quality-note')}`,
    );

    await click('app-capture-stage button.open');
    const overlay = await until(
        `document.querySelector('app-capture-stage .overlay')?.classList.contains('shown')`,
        10_000,
    );
    check(
        'AR session starts with the DOM overlay',
        !!overlay,
        overlay ? 'overlay shown' : 'overlay hidden',
    );

    await until(`!document.querySelector('app-capture-stage .shutter')?.disabled`, 5000);
    await click('app-capture-stage .shutter');
    const recording = await until(
        `document.querySelector('app-capture-stage .shutter')?.classList.contains('recording')`,
        3000,
    );
    check('recording starts', !!recording, (await text('app-capture-stage .hud')) ?? '');
    await wait((RECORD_SECONDS / 2) * 1000);
    await screenshot('02-recording.png');
    await wait((RECORD_SECONDS / 2) * 1000);
    const hud = await text('app-capture-stage .hud');
    const keyframes = Number(/(\d+)\s*keyframes/.exec(hud ?? '')?.[1] ?? 0);
    const points = Number(/(\d+)\s*points/.exec(hud ?? '')?.[1] ?? 0);
    check(
        'keyframes and live sparse points accumulate',
        keyframes >= 50 && points > 2000,
        hud?.replace(/\s+/g, ' ') ?? '',
    );

    await click('app-capture-stage .shutter');
    const viewer = await until(`!!document.querySelector('app-model-viewer canvas')`, 15_000);
    check(
        'stopping opens the model viewer',
        !!viewer,
        viewer ? 'viewer shown while the model builds' : 'no viewer',
    );
    const seen = new Set();
    const started = Date.now();
    let done = false;
    let shot = false;
    while (!done && Date.now() - started < BUILD_TIMEOUT_MS) {
        const running = await evaluate(
            `document.querySelector('app-pipeline-track li.running .name')?.firstChild.textContent.trim() ?? null`,
        );
        if (running) seen.add(running);
        if (!shot && seen.size >= 3) {
            await screenshot('03-building.png');
            shot = true;
        }
        done = await evaluate(`!document.querySelector('app-model-viewer button.save')?.disabled`);
        if (!done) await wait(300);
    }
    const buildSeconds = (Date.now() - started) / 1000;
    check(
        'reconstruction finishes',
        done,
        `${buildSeconds.toFixed(0)} s; seen running: ${[...seen].join(' → ')}`,
    );
    await openTrack();
    const track = await pipelineRows();
    await screenshot('03b-pipeline.png');
    check(
        'pipeline track lists every stage as done with its time',
        track.length === 8 &&
            track.every((row) => row.status === 'done' && /\d s$|\d:\d\d$/.test(row.time)),
        track.map((row) => `${row.name} ${row.time}`).join(' · ') +
            ` · total ${await text('app-pipeline-track .total')}`,
    );
    const method = (stage) => track.find((row) => row.name === stage)?.method ?? '';
    check(
        'High shows its methods: SIFT bundle, 1.5× stereo, confidence-weighted TSDF',
        /SIFT/.test(method('Bundle adjustment')) &&
            /1\.5×/.test(method('Dense stereo')) &&
            /confidence/.test(method('Meshing')),
        track.map((row) => `${row.name}: ${row.method}`).join(' | '),
    );
    await click('app-pipeline-track .head');
    const errorToast = await text('app-model-viewer .toast');
    if (errorToast) console.log(`  toast: ${errorToast}`);
    await wait(800);
    await screenshot('04-mesh.png');
    const summary = await text('app-model-viewer .hud .chip');
    check('textured mesh is shown', /triangles/.test(summary ?? ''), summary ?? '');

    for (const [label, file] of [
        ['Dense', '05-dense.png'],
        ['Sparse', '06-sparse.png'],
        ['Mesh', null],
    ]) {
        await evaluate(
            `[...document.querySelectorAll('app-model-viewer .segmented button')].find((b) => b.textContent.trim() === '${label}')?.click()`,
        );
        await wait(500);
        if (file) await screenshot(file);
    }

    await click('app-model-viewer button.icon[title="Metrics"]');
    await wait(400);
    const insights = await text('app-insights-sheet .body');
    await screenshot('07-metrics.png');
    const adjusted = /reprojection, adjusted\s*([\d.]+)\s*\/\s*([\d.]+)\s*px/.exec(insights ?? '');
    check(
        'diagnostics report sub-pixel reprojection',
        !!adjusted && Number(adjusted[2]) < 1,
        adjusted
            ? `mean ${adjusted[1]} / median ${adjusted[2]} px`
            : (insights ?? '').slice(0, 200),
    );
    await click('app-insights-sheet .round-action.close');

    // Measure: tap two points on the mesh.
    await click('app-model-viewer button.icon[title="Measure"]');
    await tap(160, 430);
    await wait(250);
    await tap(260, 470);
    const measured = await until(
        `document.querySelector('app-model-viewer .measure')?.textContent?.trim()`,
        3000,
    );
    await screenshot('08-measure.png');
    check('measuring tool reports a distance', !!measured, measured ?? 'no label');
    await click('app-model-viewer button.icon[title="Measure"]');

    await click('app-model-viewer button.save');
    await wait(300);
    const exports = {};
    for (const [title, extension] of [
        ['Model · GLB', '.glb'],
        ['Dense cloud · PLY', '.ply'],
        ['Metrics · JSON', '.json'],
    ]) {
        await evaluate(
            `[...document.querySelectorAll('app-export-sheet .choice')].find((b) => b.querySelector('b').textContent.trim() === '${title}')?.click()`,
        );
        exports[extension] = await downloaded(extension);
        await until(
            `![...document.querySelectorAll('app-export-sheet .choice')].some((b) => b.disabled)`,
            30_000,
        );
    }
    check(
        'exports download',
        Object.values(exports).every(Boolean),
        Object.values(exports)
            .map((p) => p && `${p.split('/').pop()} ${(statSync(p).size / 1e6).toFixed(1)} MB`)
            .join(', '),
    );

    if (exports['.glb']) {
        const glb = parseGlb(exports['.glb']);
        const errors = meshErrors(glb);
        const uvOk = !!glb.uvs && glb.uvs.every((v) => v >= 0 && v <= 1);
        check(
            'exported mesh matches the true scene',
            percentile(errors, 0.5) < 0.002 &&
                percentile(errors, 0.9) < 0.006 &&
                uvOk &&
                glb.jpegBytes > 10_000,
            `${glb.indices.length / 3} triangles, error median ${(percentile(errors, 0.5) * 1000).toFixed(2)} mm / p90 ${(percentile(errors, 0.9) * 1000).toFixed(2)} mm, texture ${(glb.jpegBytes / 1e6).toFixed(1)} MB, nodes ${glb.json.nodes.map((n) => n.name).join('+')}`,
        );
    }
    if (exports['.json']) {
        const metrics = JSON.parse(readFileSync(exports['.json'], 'utf8'));
        check(
            'metrics JSON is complete',
            metrics.keyframes > 0 && metrics.densePoints > 0 && metrics.timings.length >= 5,
            `${metrics.keyframes} keyframes, ${metrics.sparsePoints} sparse, ${metrics.densePoints} dense, voxel ${(metrics.voxel * 1000).toFixed(2)} mm, timings ${metrics.timings.map((t) => `${t.name} ${(t.ms / 1000).toFixed(1)}s`).join(', ')}`,
        );
    }

    // Rebuild the same capture at the other levels from the pipeline card.
    await click('app-export-sheet .round-action.close');
    for (const [label, limit, textured] of [
        ['Medium', 0.002, true],
        ['Low', 0.003, false],
    ]) {
        await openTrack();
        await evaluate(
            `[...document.querySelectorAll('app-pipeline-track .rebuild button')].find((b) => b.textContent.trim() === '${label}')?.click()`,
        );
        const rebuildStart = Date.now();
        const rebuilt = await until(
            `document.querySelector('app-pipeline-track .head .quality')?.textContent.trim() === '${label}' && !document.querySelector('app-pipeline-track .head .spinner') && !document.querySelector('app-model-viewer button.save')?.disabled`,
            BUILD_TIMEOUT_MS,
            500,
        );
        await openTrack();
        const rows = await pipelineRows();
        await screenshot(`09-rebuild-${label.toLowerCase()}.png`);
        const texture = rows.find((row) => row.name === 'Texture')?.method ?? '';
        check(
            `rebuild at ${label} finishes with its own methods`,
            !!rebuilt &&
                rows.every((row) => row.status === 'done') &&
                (textured ? /atlas/.test(texture) : /vertex colours/.test(texture)),
            `${((Date.now() - rebuildStart) / 1000).toFixed(0)} s; ${rows.map((row) => `${row.name} ${row.time}`).join(' · ')}; texture: ${texture}`,
        );
        await click('app-pipeline-track .head');
        const path = await exportGlb();
        if (!path) {
            check(`${label} GLB downloads`, false, 'no file');
            continue;
        }
        const glb = parseGlb(path);
        const errors = meshErrors(glb);
        const appearance = textured
            ? !!glb.uvs && glb.jpegBytes > 10_000 && !glb.colors
            : !!glb.colors && !glb.uvs && glb.jpegBytes === 0;
        check(
            `${label} GLB matches the true scene`,
            percentile(errors, 0.5) < limit && appearance,
            `${glb.indices.length / 3} triangles, error median ${(percentile(errors, 0.5) * 1000).toFixed(2)} mm / p90 ${(percentile(errors, 0.9) * 1000).toFixed(2)} mm, ${textured ? `atlas ${(glb.jpegBytes / 1e6).toFixed(1)} MB` : 'vertex colours'}`,
        );
    }

    check(
        'no console errors',
        page.errors.length === 0,
        page.errors.slice(0, 3).join(' | ') || 'clean',
    );
} catch (error) {
    console.error(error);
    results.push({ name: 'run', pass: false });
} finally {
    chrome.close();
    server.close();
}
const failed = results.filter((r) => !r.pass).length;
console.log(`\n${results.length - failed}/${results.length} checks passed · screenshots in ${OUT}`);
process.exit(failed ? 1 : 0);
