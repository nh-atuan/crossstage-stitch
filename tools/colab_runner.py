"""Colab Remote Debugging Bridge (CDP over port 9222).
Automates syncing the working branch to Google Colab Tesla T4 and running builds/tests.
"""
import asyncio
import base64
import io
import json
import os
import tarfile
import sys
import time
import websockets

def get_devtools_active_port():
    port_file = r'C:\Users\ADMIN\AppData\Local\Google\Chrome\User Data\DevToolsActivePort'
    if not os.path.exists(port_file):
        raise FileNotFoundError(f"DevToolsActivePort not found at {port_file}")
    with open(port_file, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    return lines[0], lines[1]

class ColabBridge:
    def __init__(self):
        self.ws = None
        self.session_id = None
        self.msg_id = 0

    async def connect(self):
        port, path = get_devtools_active_port()
        browser_uri = f'ws://127.0.0.1:{port}{path}'
        self.ws = await websockets.connect(browser_uri, max_size=200_000_000)
        
        # 1. Discover Colab target
        self.msg_id += 1
        await self.ws.send(json.dumps({'id': self.msg_id, 'method': 'Target.getTargets'}))
        targets_res = json.loads(await self.ws.recv())
        
        colab_id = None
        for t in targets_res['result']['targetInfos']:
            if t['type'] == 'page' and 'colab.research.google.com' in t.get('url', ''):
                colab_id = t['targetId']
                break
        
        if not colab_id:
            raise RuntimeError("Colab page target not found in Chrome!")
        
        # 2. Attach to Colab target
        self.msg_id += 1
        await self.ws.send(json.dumps({
            'id': self.msg_id,
            'method': 'Target.attachToTarget',
            'params': {'targetId': colab_id, 'flatten': True}
        }))
        
        while not self.session_id:
            msg = json.loads(await self.ws.recv())
            if msg.get('id') == self.msg_id and 'result' in msg and 'sessionId' in msg['result']:
                self.session_id = msg['result']['sessionId']
            elif msg.get('method') == 'Target.attachedToTarget' and 'params' in msg:
                self.session_id = msg['params']['sessionId']

    async def close(self):
        if self.ws:
            await self.ws.close()

    async def evaluate_js(self, expression: str, return_by_value: bool = True):
        self.msg_id += 1
        call_id = self.msg_id
        await self.ws.send(json.dumps({
            'id': call_id,
            'sessionId': self.session_id,
            'method': 'Runtime.evaluate',
            'params': {'expression': expression, 'returnByValue': return_by_value}
        }))
        
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get('id') == call_id:
                if 'error' in msg:
                    raise RuntimeError(f"CDP Evaluation error: {msg['error']}")
                result = msg.get('result', {}).get('result', {})
                return result.get('value')

    async def execute_in_colab(self, py_code: str, timeout_sec: int = 600):
        # Set code into Monaco editor and click run
        escaped_code = json.dumps(py_code)
        setup_js = f"""(() => {{
            const ed = window.monaco.editor.getEditors()[0];
            ed.setValue({escaped_code});
            const btn = document.querySelector('colab-run-button').shadowRoot.querySelector('#run-button');
            btn.click();
            return 'launched';
        }})()"""
        
        await self.evaluate_js(setup_js)
        
        check_js = """(() => {
            const btn = document.querySelector('colab-run-button');
            const shadow = btn ? btn.shadowRoot : null;
            const execDiv = shadow ? shadow.querySelector('.cell-execution') : null;
            const isRunning = execDiv ? (execDiv.classList.contains('running') || execDiv.classList.contains('pending')) : false;
            
            const cell = document.querySelector('.cell');
            const output = cell ? (cell.querySelector('.output') || cell.querySelector('colab-output-area')) : null;
            const text = output ? (output.innerText || output.textContent) : '';
            
            return { isRunning, text };
        })()"""
        
        start_time = time.time()
        await asyncio.sleep(0.5)
        
        last_text = ""
        while time.time() - start_time < timeout_sec:
            res = await self.evaluate_js(check_js)
            if res and isinstance(res, dict):
                is_running = res.get('isRunning', False)
                text = res.get('text', '')
                if text != last_text:
                    last_text = text
                if not is_running:
                    return text
            await asyncio.sleep(1.0)
            
        raise TimeoutError(f"Colab cell execution timed out after {timeout_sec}s! Output so far:\n{last_text}")

async def sync_repo_to_colab(bridge: ColabBridge):
    print("Preparing repository tarball from local branch...")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for root in ['src', 'harness', 'tests', 'tools']:
            for dirpath, _, filenames in os.walk(root):
                if '__pycache__' in dirpath or '.pytest_cache' in dirpath:
                    continue
                for f in filenames:
                    full = os.path.join(dirpath, f)
                    tar.add(full)
        tar.add('CMakeLists.txt')
        if os.path.exists('data'):
            for dirpath, _, filenames in os.walk('data'):
                for f in filenames:
                    tar.add(os.path.join(dirpath, f))

    gz_bytes = buf.getvalue()
    b64_str = base64.b64encode(gz_bytes).decode('ascii')
    print(f"Archive compressed: {len(gz_bytes) / 1024:.1f} KB (b64 length: {len(b64_str)})")
    
    # Send extraction code to Colab
    code = f"""
import base64, io, os, tarfile, shutil
shutil.rmtree('/content/crossstage-stitch', ignore_errors=True)
os.makedirs('/content/crossstage-stitch', exist_ok=True)
raw = base64.b64decode('{b64_str}')
with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as tar:
    tar.extractall('/content/crossstage-stitch')
print("Successfully synced repository to /content/crossstage-stitch!")
import subprocess
print(subprocess.check_output(['ls', '-la', '/content/crossstage-stitch']).decode())
"""
    print("Syncing files to Colab...")
    out = await bridge.execute_in_colab(code)
    print(out)

async def main():
    bridge = ColabBridge()
    await bridge.connect()
    print("Connected to Colab via Chrome Remote Debugging (port 9222)!")
    
    if len(sys.argv) > 1 and sys.argv[1] == '--sync':
        await sync_repo_to_colab(bridge)
    elif len(sys.argv) > 1 and sys.argv[1] == '--cmd':
        cmd = " ".join(sys.argv[2:])
        print(f"Running command on Colab: {cmd}")
        py_cmd = f"!{cmd}"
        out = await bridge.execute_in_colab(py_cmd)
        print(out)
    else:
        # Default test: query GPU
        out = await bridge.execute_in_colab("!nvidia-smi")
        print(out)
        
    await bridge.close()

if __name__ == '__main__':
    asyncio.run(main())

