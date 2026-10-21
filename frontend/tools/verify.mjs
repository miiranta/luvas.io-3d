// Bundles tools/verify.ts for the browser, serves it on localhost (WebGPU needs a secure
// context) and runs it in headless Chrome. Exit code 1 when any check fails.
import { build } from 'esbuild';
import { createServer } from 'node:http';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { openChrome, watchConsole } from './devtools.mjs';

const OUT = '.verify';
const PORT = Number(process.env.VERIFY_PORT ?? 4391);
const TIMEOUT_MS = Number(process.env.VERIFY_TIMEOUT ?? 900_000);

mkdirSync(OUT, { recursive: true });
await build({
    entryPoints: ['tools/verify.ts'],
    bundle: true,
    format: 'esm',
    platform: 'browser',
    target: 'es2022',
    outfile: join(OUT, 'verify.js'),
    logLevel: 'warning',
});
writeFileSync(
    join(OUT, 'index.html'),
    '<!doctype html><meta charset="utf-8"><script type="module" src="verify.js"></script>',
);

const server = createServer((request, response) => {
    const file = request.url === '/' ? 'index.html' : request.url.slice(1);
    try {
        const body = readFileSync(join(OUT, file));
        response.writeHead(200, {
            'content-type': file.endsWith('.js') ? 'text/javascript' : 'text/html',
        });
        response.end(body);
    } catch {
        response.writeHead(404).end();
    }
}).listen(PORT);

const chrome = await openChrome({ name: 'verify', port: 9444 });
const console_ = watchConsole(chrome.socket);
let failed = true;
try {
    await chrome.call('Page.navigate', { url: `http://localhost:${PORT}/` });
    const started = Date.now();
    let result = null;
    while (!result && Date.now() - started < TIMEOUT_MS) {
        await new Promise((resolve) => setTimeout(resolve, 2000));
        result = await chrome.evaluate('window.__verify ?? null');
    }
    if (!result) throw new Error('verification timed out');
    for (const line of result.log) console.log(`  · ${line}`);
    for (const check of result.checks)
        console.log(`${check.pass ? 'PASS' : 'FAIL'}  ${check.name} — ${check.detail}`);
    if (result.error) throw new Error(result.error);
    failed = result.checks.some((check) => !check.pass);
    console.log(
        `\n${result.checks.filter((c) => c.pass).length}/${result.checks.length} checks passed in ${((Date.now() - started) / 1000).toFixed(0)} s`,
    );
} catch (error) {
    console.error(error.message);
    for (const line of console_.errors) console.error(`  console: ${line}`);
} finally {
    chrome.close();
    server.close();
}
process.exit(failed ? 1 : 0);
