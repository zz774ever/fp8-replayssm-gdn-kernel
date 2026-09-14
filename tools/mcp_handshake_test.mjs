// Temporary verification harness for the local Codex MCP entry "mcp-ssh-apply-patch".
// It spawns exactly the command/args recorded in ~/.codex/config.toml and speaks
// MCP JSON-RPC over stdio, so it validates the configured entry point end to end.
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

const child = spawn('cmd', ['/c', 'node', ENTRY], {
  stdio: ['pipe', 'pipe', 'pipe'],
  windowsHide: true,
  env: { ...process.env, MCP_SILENT: 'true', ProgramData: 'C:\\ProgramData' },
});

let buf = '';
const waiters = new Map();

child.stdout.on('data', (d) => {
  buf += d.toString();
  let idx;
  while ((idx = buf.indexOf('\n')) >= 0) {
    const line = buf.slice(0, idx).trim();
    buf = buf.slice(idx + 1);
    if (!line) continue;
    let msg;
    try {
      msg = JSON.parse(line);
    } catch {
      console.log('[non-json]', line.slice(0, 200));
      continue;
    }
    if (msg.id !== undefined && waiters.has(msg.id)) {
      waiters.get(msg.id)(msg);
      waiters.delete(msg.id);
    } else {
      console.log('[notification]', JSON.stringify(msg).slice(0, 300));
    }
  }
});

child.stderr.on('data', (d) => console.log('[stderr]', d.toString().trim().slice(0, 400)));

function send(obj) {
  child.stdin.write(JSON.stringify(obj) + '\n');
}

function request(obj, timeoutMs = 120000) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('timeout: ' + obj.method)), timeoutMs);
    waiters.set(obj.id, (m) => {
      clearTimeout(timer);
      resolve(m);
    });
    send(obj);
  });
}

const textOf = (resp) => resp?.result?.content?.[0]?.text;

const init = await request({
  jsonrpc: '2.0',
  id: 1,
  method: 'initialize',
  params: {
    protocolVersion: '2024-11-05',
    capabilities: {},
    clientInfo: { name: 'verify', version: '1' },
  },
});
console.log('serverInfo:', JSON.stringify(init.result?.serverInfo));
send({ jsonrpc: '2.0', method: 'notifications/initialized' });

const tools = await request({ jsonrpc: '2.0', id: 2, method: 'tools/list', params: {} });
const toolList = tools.result?.tools ?? [];
console.log('tools:', toolList.map((t) => t.name).join(', '));
const desc = toolList.find((t) => t.name === 'runRemoteCommand')?.description ?? '';
console.log('apply_patch description present:', desc.includes('prefer it for code and file edits'));

const hosts = await request({ jsonrpc: '2.0', id: 3, method: 'tools/call', params: { name: 'listKnownHosts', arguments: {} } });
const hostList = JSON.parse(textOf(hosts));
console.log(
  'known hosts:',
  hostList
    .map((h) => `${h.alias}(${h.user ?? '?'}@${h.hostname}:${h.port ?? 22}${h.passwordAuth ? ',password' : ',key'})`)
    .join('  '),
);

const run = await request({
  jsonrpc: '2.0',
  id: 4,
  method: 'tools/call',
  params: {
    name: 'runRemoteCommand',
    arguments: {
      hostAlias: 'gdn-remote',
      command: 'hostname; whoami; ls -l ~/.local/bin/apply_patch ~/.local/bin/codex',
    },
  },
});
console.log('remote command result:', textOf(run));

// Phase 5 of the guide: test the remote apply_patch wrapper under a clean PATH
// (simulating the SSH MCP's non-login shell), both stdin mode and argument mode.
const cleanPathTest = [
  'set -e',
  'wd=$(mktemp -d)',
  'cd "$wd"',
  `printf '%s\\n' 'print("old")' > test.py`,
  `patch=$(printf '%s\\n' '*** Begin Patch' '*** Update File: test.py' '@@' '-print("old")' '+print("new")' '*** End Patch')`,
  `printf '%s\\n' "$patch" | env -i HOME="$HOME" PATH=/usr/bin:/bin "$HOME/.local/bin/apply_patch"`,
  'echo "--- file after stdin-mode patch ---"',
  'cat test.py',
  `env -i HOME="$HOME" PATH=/usr/bin:/bin "$HOME/.local/bin/apply_patch" "$(printf '%s\\n' '*** Begin Patch' '*** Update File: test.py' '@@' '-print("new")' '+print("arg-mode")' '*** End Patch')"`,
  'echo "--- file after argument-mode patch ---"',
  'cat test.py',
  'cd /',
  'rm -rf "$wd"',
].join('\n');

const patchTest = await request({
  jsonrpc: '2.0',
  id: 5,
  method: 'tools/call',
  params: {
    name: 'runRemoteCommand',
    arguments: { hostAlias: 'gdn-remote', command: cleanPathTest },
  },
});
console.log('remote clean-PATH apply_patch result:', textOf(patchTest));

child.kill();
process.exit(0);
