import { spawn } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

function freePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      server.close(() => resolve(address.port));
    });
  });
}

async function waitForJson(url, timeoutMs = 15000) {
  const deadline = Date.now() + timeoutMs;
  let lastError;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) return await response.json();
    } catch (error) {
      lastError = error;
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw lastError || new Error(`Timed out waiting for ${url}`);
}

async function stopBrowser(browser, timeoutMs = 3000) {
  if (browser.exitCode !== null || browser.signalCode !== null) return;
  const exited = new Promise((resolve) => browser.once("exit", resolve));
  browser.kill();
  await Promise.race([
    exited,
    new Promise((resolve) => setTimeout(resolve, timeoutMs)),
  ]);
}

async function closeBrowserViaCdp(client, timeoutMs = 3000) {
  if (!client) return;
  try {
    await Promise.race([
      client.send("Browser.close"),
      new Promise((resolve) => setTimeout(resolve, timeoutMs)),
    ]);
  } catch {
    // The browser may close its WebSocket before acknowledging Browser.close.
  }
  try {
    client.close();
  } catch {
    // Closing an already-closed WebSocket is harmless during final cleanup.
  }
}

class CdpClient {
  constructor(url) {
    this.socket = new WebSocket(url);
    this.nextId = 1;
    this.pending = new Map();
    this.events = new Map();
  }

  async open() {
    await new Promise((resolve, reject) => {
      this.socket.addEventListener("open", resolve, { once: true });
      this.socket.addEventListener("error", reject, { once: true });
    });
    this.socket.addEventListener("message", (event) => {
      const message = JSON.parse(event.data);
      if (message.id) {
        const pending = this.pending.get(message.id);
        if (!pending) return;
        this.pending.delete(message.id);
        if (message.error) pending.reject(new Error(message.error.message));
        else pending.resolve(message.result || {});
        return;
      }
      const listeners = this.events.get(message.method) || [];
      this.events.delete(message.method);
      for (const listener of listeners) listener(message.params || {});
    });
  }

  send(method, params = {}) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.socket.send(JSON.stringify({ id, method, params }));
    });
  }

  once(method) {
    return new Promise((resolve) => {
      const listeners = this.events.get(method) || [];
      listeners.push(resolve);
      this.events.set(method, listeners);
    });
  }

  close() {
    this.socket.close();
  }
}

async function main() {
  const [browserPath, htmlPath, outputPath] = process.argv.slice(2);
  if (!browserPath || !htmlPath || !outputPath) {
    throw new Error("usage: node render_html_with_edge.mjs <browser> <html> <png>");
  }
  const resolvedHtml = path.resolve(htmlPath);
  const resolvedOutput = path.resolve(outputPath);
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "advisor-edge-"));
  const port = await freePort();
  const browser = spawn(
    browserPath,
    [
      "--headless=new",
      "--disable-gpu",
      "--hide-scrollbars",
      "--no-first-run",
      "--no-default-browser-check",
      "--edge-skip-compat-layer-relaunch",
      `--remote-debugging-port=${port}`,
      `--user-data-dir=${tempRoot}`,
      "about:blank",
    ],
    { stdio: "ignore", windowsHide: true },
  );
  let browserClient;
  try {
    const version = await waitForJson(`http://127.0.0.1:${port}/json/version`);
    browserClient = new CdpClient(version.webSocketDebuggerUrl);
    await browserClient.open();
    const response = await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, {
      method: "PUT",
    });
    if (!response.ok) throw new Error(`Unable to create browser target: ${response.status}`);
    const target = await response.json();
    const client = new CdpClient(target.webSocketDebuggerUrl);
    await client.open();
    try {
      await client.send("Page.enable");
      await client.send("Runtime.enable");
      await client.send("Emulation.setDeviceMetricsOverride", {
        width: 1080,
        height: 800,
        deviceScaleFactor: 1,
        mobile: false,
      });
      const loaded = client.once("Page.loadEventFired");
      await client.send("Page.navigate", { url: pathToFileURL(resolvedHtml).href });
      await loaded;
      await client.send("Runtime.evaluate", {
        expression: "document.fonts && document.fonts.ready",
        awaitPromise: true,
        returnByValue: true,
      });
      const metrics = await client.send("Runtime.evaluate", {
        expression: "({width: document.documentElement.scrollWidth, height: document.documentElement.scrollHeight, text: document.body.innerText})",
        returnByValue: true,
      });
      const value = metrics.result.value;
      const width = Math.max(1080, Math.ceil(value.width));
      const height = Math.max(1, Math.min(20000, Math.ceil(value.height)));
      await client.send("Emulation.setDeviceMetricsOverride", {
        width,
        height,
        deviceScaleFactor: 1,
        mobile: false,
      });
      const screenshot = await client.send("Page.captureScreenshot", {
        format: "png",
        fromSurface: true,
        captureBeyondViewport: true,
      });
      fs.mkdirSync(path.dirname(resolvedOutput), { recursive: true });
      fs.writeFileSync(resolvedOutput, Buffer.from(screenshot.data, "base64"));
      process.stdout.write(JSON.stringify({ width, height, textLength: value.text.length }));
    } finally {
      client.close();
    }
  } finally {
    await closeBrowserViaCdp(browserClient);
    await stopBrowser(browser);
    fs.rmSync(tempRoot, {
      recursive: true,
      force: true,
      maxRetries: 20,
      retryDelay: 100,
    });
  }
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
