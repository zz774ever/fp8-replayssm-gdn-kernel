// Thin CLI bridge to the local Codex MCP entry "mcp-ssh-apply-patch".
//
// The MCP server is normally driven by the Codex app. Until that MCP is
// reloaded, this script talks to the very same server process / command line
// recorded in ~/.codex/config.toml, so remote reads, uploads and apply_patch
// edits all keep using the configured SSH host and its credential handling.
//
// Usage:
//   node tools/mcp_ssh_cli.mjs run <hostAlias> <command...>
//   node tools/mcp_ssh_cli.mjs put <hostAlias> <localPath> <remotePath>
//   node tools/mcp_ssh_cli.mjs get <hostAlias> <remotePath> <localPath>
//   node tools/mcp_ssh_cli.mjs hosts
import { spawn } from 'node:child_process';
import { homedir } from 'node:os';
import { join } from 'node:path';

const ENTRY = join(
  homedir(),
  '.codex',
  'mcp-ssh-apply-patch',
  'node_modules',
  '@aiondadotcom',
  'mcp-ssh',
  'bin',
  'mcp-ssh.js',
);

const [action, ...rest] = process.argv.slice(2);
if (!action) {
  console.error('usage: run|put|get|hosts ...');
  process.exit(2);
}

const child = spawn('cmd', ['/c', 'node', ENTRY], {
  stdio: ['pipe', 'pipe', 'pipe'],
  windowsHide: true,
  env: { ...process.env, MCP_SILENT: 'true', ProgramData: 'C:\\ProgramData' },
});

let buffer = '';
const pending = new Map();
child.stdout.on('data', (chunk) => {
  buffer += chunk.toString();
  let index;
  while ((index = buffer.indexOf('\n')) >= 0) {
    const line = buffer.slice(0, index).trim();
    buffer = buffer.slice(index + 1);
    if (!line) continue;
    let message;
    try {
      message = JSON.parse(line);
    } catch {
      continue;
    }
    if (message.id !== undefined && pending.has(message.id)) {
      pending.get(message.id)(message);
      pending.delete(message.id);
    }
  }
});
child.stderr.on('data', (chunk) => {
  const text = chunk.toString().trim();
  if (text) console.error('[mcp]', text.slice(0, 500));
});

let nextId = 1;
function request(method, params, timeoutMs = 300000) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`timeout: ${method}`)), timeoutMs);
    pending.set(id, (message) => {
      clearTimeout(timer);
      resolve(message);
    });
    child.stdin.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n');
  });
}

function call(name, args) {
  return request('tools/call', { name, arguments: args }).then((message) => {
    const text = message.result?.content?.[0]?.text;
    if (text === undefined) throw new Error(JSON.stringify(message).slice(0, 400));
    return JSON.parse(text);
  });
}

await request('initialize', {
  protocolVersion: '2024-11-05',
  capabilities: {},
  clientInfo: { name: 'mcp-ssh-cli', version: '1' },
});
child.stdin.write(JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' }) + '\n');

let exitCode = 0;
try {
  if (action === 'hosts') {
    console.log(JSON.stringify(await call('listKnownHosts', {}), null, 2));
  } else if (action === 'run') {
    const [alias, ...command] = rest;
    const result = await call('runRemoteCommand', {
      hostAlias: alias,
      command: command.join(' '),
    });
    process.stdout.write(result.stdout ?? '');
    if (result.stderr) process.stderr.write(result.stderr);
    exitCode = result.code ?? 0;
    console.error(`[exit ${result.code}]`);
  } else if (action === 'put' || action === 'get') {
    const [alias, from, to] = rest;
    const args = { hostAlias: alias };
    if (action === 'put') {
      args.localPath = from;
      args.remotePath = to;
    } else {
      args.remotePath = from;
      args.localPath = to;
    }
    const result = await call(action === 'put' ? 'uploadFile' : 'downloadFile', args);
    console.log(JSON.stringify(result));
    if (!result.success) exitCode = 1;
  } else {
    console.error(`unknown action: ${action}`);
    exitCode = 2;
  }
} catch (error) {
  console.error(String(error?.message ?? error));
  exitCode = 1;
}

child.kill();
process.exit(exitCode);
