import { spawn } from 'node:child_process';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const CONNECT_ATTEMPTS = 80;
const CONNECT_INTERVAL_MS = 250;

/**
 * Hardware WebGPU + WebGL in headless Chrome on Linux. The first adapter request may restart the
 * GPU process (dropping live WebGL contexts), so pages must obtain WebGPU before using WebGL.
 */
export const GPU_FLAGS = [
    '--enable-unsafe-webgpu',
    '--ignore-gpu-blocklist',
    '--enable-gpu',
    '--use-gl=angle',
    '--use-angle=gl',
    '--enable-features=Vulkan',
    '--use-vulkan=native',
];

async function debuggerUrl(port, log) {
    for (let attempt = 0; attempt < CONNECT_ATTEMPTS; attempt++) {
        try {
            const response = await fetch(`http://127.0.0.1:${port}/json/version`);
            return (await response.json()).webSocketDebuggerUrl;
        } catch {
            await new Promise((resolve) => setTimeout(resolve, CONNECT_INTERVAL_MS));
        }
    }
    throw new Error(`devtools unavailable\n${log.join('')}`);
}

export async function openChrome({ name, port, flags = [], windowSize = '1280,900' }) {
    const profile = mkdtempSync(join(tmpdir(), `chrome-${name}-`));
    const chrome = spawn(
        'google-chrome',
        [
            '--headless=new',
            `--remote-debugging-port=${port}`,
            `--user-data-dir=${profile}`,
            '--no-first-run',
            '--no-default-browser-check',
            `--window-size=${windowSize}`,
            ...GPU_FLAGS,
            ...flags,
            'about:blank',
        ],
        { stdio: ['ignore', 'pipe', 'pipe'] },
    );
    const log = [];
    chrome.stderr.on('data', (chunk) => log.push(chunk.toString()));

    const socket = new WebSocket(await debuggerUrl(port, log));
    await new Promise((resolve, reject) => {
        socket.addEventListener('open', resolve, { once: true });
        socket.addEventListener('error', reject, { once: true });
    });

    let nextId = 1;
    const pending = new Map();
    socket.addEventListener('message', (event) => {
        const message = JSON.parse(event.data);
        if (!message.id || !pending.has(message.id)) return;
        const { resolve, reject } = pending.get(message.id);
        pending.delete(message.id);
        if (message.error) reject(new Error(JSON.stringify(message.error)));
        else resolve(message.result);
    });

    const send = (method, params = {}, sessionId) =>
        new Promise((resolve, reject) => {
            const id = nextId++;
            pending.set(id, { resolve, reject });
            socket.send(JSON.stringify({ id, method, params, sessionId }));
        });

    const { targetId } = await send('Target.createTarget', { url: 'about:blank' });
    const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true });
    const call = (method, params) => send(method, params, sessionId);
    await call('Page.enable');
    await call('Runtime.enable');

    const evaluate = async (expression) => {
        const result = await call('Runtime.evaluate', {
            expression,
            awaitPromise: true,
            returnByValue: true,
        });
        if (result.exceptionDetails) {
            const detail =
                result.exceptionDetails.exception?.description ?? result.exceptionDetails.text;
            throw new Error(`evaluation error: ${detail}`);
        }
        return result.result.value;
    };

    const close = () => {
        socket.close();
        chrome.kill('SIGKILL');
    };

    return { socket, send, call, evaluate, close, log };
}

/** Collects console errors and uncaught exceptions of the page. */
export function watchConsole(socket) {
    const errors = [];
    const lines = [];
    socket.addEventListener('message', (event) => {
        const message = JSON.parse(event.data);
        if (message.method === 'Runtime.consoleAPICalled') {
            const text = message.params.args
                .map((arg) => arg.value ?? arg.description ?? '')
                .join(' ');
            lines.push(text);
            if (message.params.type === 'error') errors.push(text);
        }
        if (message.method === 'Runtime.exceptionThrown') {
            errors.push(
                message.params.exceptionDetails.exception?.description ??
                    message.params.exceptionDetails.text,
            );
        }
    });
    return { errors, lines };
}
