"""Single-host NIXL READ transport with a small HTTP control channel.

Only registration metadata and acquire/release messages use HTTP. KV bytes
travel through NIXL/UCX between registered GPU allocations. A producer grants
one reader per transfer attempt and retains its blocks until release. Claimed
blocks NEVER expire: a dead peer requires coordinated engine shutdown/restart.
"""
import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from queue import SimpleQueue
import threading
import time
from urllib.request import Request, build_opener, ProxyHandler

import torch

from nanovllm.distributed.kv_transfer.base import WorkerConnector
from nanovllm.distributed.kv_transfer.metadata import ConnectorOutput, KVTransferFailure


def block_descriptors(info, blocks, num_tokens):
    """Ordered K/V, layer, logical-block regions; omit final-block padding."""
    _, layers, capacity, block_size, heads, dim = info['shape']
    if len(blocks) != (num_tokens + block_size - 1) // block_size:
        raise ValueError("Invalid descriptor block count")
    if len(set(blocks)) != len(blocks) or any(b < 0 or b >= capacity for b in blocks):
        raise ValueError("Invalid descriptor block IDs")
    token_bytes = heads * dim * info['itemsize']
    descriptors = []
    for region in range(2 * layers):
        for logical, block in enumerate(blocks):
            count = min(block_size, num_tokens - logical * block_size)
            address = info['address'] + (region * capacity + block) * block_size * token_bytes
            descriptors.append((address, count * token_bytes, info['device']))
    return descriptors


class NixlWorkerConnector(WorkerConnector):
    def __init__(self, config, model_id):
        super().__init__(config)
        from nixl._api import nixl_agent, nixl_agent_config
        self.agent = nixl_agent(config.engine_id, nixl_agent_config(backends=['UCX']))
        self.model_id = model_id
        self.cache = None
        self.registration = None
        self.info = None
        self._published = {}
        self._lock = threading.Lock()
        self._released = SimpleQueue()
        self._loads = {}
        self._cancelled = set()
        self._agents = {}
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix='kv-control')
        self._events = ConnectorOutput()
        self._server = None
        self._thread = None
        self._closed = False

    def register_kv_cache(self, kv_cache):
        if self.cache is not None:
            raise RuntimeError("KV cache already registered")
        if not kv_cache.is_cuda or not kv_cache.is_contiguous() or kv_cache.ndim != 6:
            raise ValueError("NIXL requires a contiguous six-dimensional CUDA KV cache")
        self.cache = kv_cache
        self.registration = self.agent.get_reg_descs(kv_cache)
        self.agent.register_memory(self.registration)
        self.info = dict(
            engine_id=self.config.engine_id, model_id=self.model_id,
            shape=list(kv_cache.shape), dtype=str(kv_cache.dtype),
            itemsize=kv_cache.element_size(), address=kv_cache.data_ptr(),
            device=kv_cache.device.index,
            agent=base64.b64encode(self.agent.get_agent_metadata()).decode(),
        )
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                try:
                    length = int(self.headers.get('Content-Length', 0))
                    if not 0 < length <= 1024 * 1024:
                        raise ValueError("Invalid control message size")
                    response = owner._control(json.loads(self.rfile.read(length)))
                    encoded = json.dumps(response).encode()
                    self.send_response(200)
                    self.send_header('Content-Length', str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as error:
                    self.send_error(400, str(error))

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer((self.config.host, self.config.port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def _control(self, message):
        key, reader = message['transfer_id'], message['reader']
        with self._lock:
            record = self._published.get(key)
            if message['op'] == 'release':
                if record and record['reader'] == reader:
                    record['released'] = True
                    self._released.put(key)
                return {'status': 'ok'}  # Idempotent after removal.
            if message['op'] != 'acquire':
                raise ValueError("Unknown operation")
            if record is None:
                return {'status': 'wait'}
            if record['released'] or record['cancelled']:
                return {'status': 'error', 'message': 'Transfer no longer available'}
            if record['reader'] not in (None, reader):
                return {'status': 'error', 'message': 'Transfer already claimed'}
            if message['signature'] != self._signature(self.info):
                return {'status': 'error', 'message': 'Model or KV layout mismatch'}
            if message['remote_request_id'] != record['params'].remote_request_id:
                return {'status': 'error', 'message': 'Remote request mismatch'}
            record['reader'] = reader
            return {'status': 'ok', 'info': self.info, 'params': asdict(record['params'])}

    @staticmethod
    def _signature(info):
        shape = info['shape']
        return [info['model_id'], info['dtype'], shape[:2] + shape[3:]]

    def _rpc(self, remote, message):
        # Control plane is local and must not follow inherited HTTP proxies.
        opener = build_opener(ProxyHandler({}))
        request = Request(f'http://{remote.remote_host}:{remote.remote_port}/',
                          json.dumps(message).encode(), {'Content-Type': 'application/json'})
        with opener.open(request, timeout=min(2.0, self.config.transfer_timeout)) as response:
            return json.load(response)

    def _acquire(self, task):
        deadline = time.monotonic() + self.config.transfer_timeout
        message = dict(op='acquire', transfer_id=task.remote.transfer_id,
                       reader=self.config.engine_id,
                       remote_request_id=task.remote.remote_request_id,
                       signature=self._signature(self.info))
        while time.monotonic() < deadline:
            try:
                result = self._rpc(task.remote, message)
                if result['status'] != 'wait':
                    return result
            except (OSError, ValueError):
                pass
            time.sleep(0.02)
        return {'status': 'error', 'message': 'Remote KV acquisition timed out'}

    def _ack(self, task):
        deadline = time.monotonic() + self.config.transfer_timeout
        while time.monotonic() < deadline:
            try:
                self._rpc(task.remote, dict(op='release', transfer_id=task.remote.transfer_id,
                                           reader=self.config.engine_id))
                return
            except (OSError, ValueError):
                time.sleep(0.02)
        raise TimeoutError("Could not confirm remote block release; peer may be unavailable")

    def start_load_kv(self):
        metadata = self._get_connector_metadata()
        for params in metadata.sends:
            if params.remote_engine_id != self.config.engine_id:
                raise ValueError("Cannot publish another engine's KV")
            # Request-granularity fence, no layer overlap in this implementation.
            torch.cuda.synchronize(self.cache.device)
            block_descriptors(self.info, params.remote_block_ids, params.num_tokens)
            with self._lock:
                if params.transfer_id in self._published:
                    raise ValueError("Duplicate published transfer")
                self._published[params.transfer_id] = dict(
                    params=params, reader=None, released=False, cancelled=False,
                    deadline=time.monotonic() + self.config.transfer_timeout,
                )
        for task in metadata.loads:
            key = task.remote.transfer_id
            if key in self._loads:
                raise ValueError("Duplicate load transfer")
            block_descriptors(self.info, task.local_block_ids, task.remote.num_tokens)
            self._loads[key] = dict(task=task, future=self._pool.submit(self._acquire, task),
                                    handle=None, stage='acquire', error=None,
                                    deadline=time.monotonic() + self.config.transfer_timeout)
        self._cancelled.update(metadata.cancels)
        with self._lock:
            for key in metadata.cancels:
                if key in self._published:
                    self._published[key]['cancelled'] = True

    def _begin_read(self, record, response):
        task = record['task']
        remote_info = response['info']
        params = response['params']
        if (remote_info['engine_id'] != task.remote.remote_engine_id
                or self._signature(remote_info) != self._signature(self.info)
                or tuple(params['remote_block_ids']) != task.remote.remote_block_ids
                or params['num_tokens'] != task.remote.num_tokens
                or params['transfer_id'] != task.remote.transfer_id):
            raise ValueError("Remote handoff differs from scheduled transfer")
        engine_id = remote_info['engine_id']
        if engine_id not in self._agents:
            self._agents[engine_id] = self.agent.add_remote_agent(base64.b64decode(remote_info['agent']))
        local = self.agent.get_xfer_descs(
            block_descriptors(self.info, task.local_block_ids, task.remote.num_tokens), 'VRAM')
        remote = self.agent.get_xfer_descs(
            block_descriptors(remote_info, task.remote.remote_block_ids, task.remote.num_tokens), 'VRAM')
        handle = self.agent.initialize_xfer('READ', local, remote, self._agents[engine_id])
        record['handle'] = handle
        # CUDA Graph's previous replay must not still read destination storage.
        torch.cuda.synchronize(self.cache.device)
        if self.agent.transfer(handle) == 'ERR':
            raise RuntimeError("NIXL READ submission failed")
        record['stage'] = 'read'
        record['deadline'] = time.monotonic() + self.config.transfer_timeout

    def _finish_read(self, record):
        handle = record['handle']
        if handle is not None:
            # NIXL raises if an active request cannot be cancelled. In that case
            # retain handle and blocks, and try again on the next progress step.
            self.agent.release_xfer_handle(handle)
            record['handle'] = None
        record['future'] = self._pool.submit(self._ack, record['task'])
        record['stage'] = 'ack'

    def get_finished(self):
        now = time.monotonic()
        with self._lock:
            for key, record in list(self._published.items()):
                expired = record['reader'] is None and now >= record['deadline']
                cancelled = record['cancelled'] and record['reader'] is None
                if record['released'] or expired or cancelled:
                    del self._published[key]
                    if record['cancelled']:
                        self._events.cancelled.add(key)
                    elif expired:
                        self._events.failures.append(KVTransferFailure(key, 'Unclaimed KV expired'))
                    else:
                        self._events.sent.add(key)
                    self._cancelled.discard(key)
        while not self._released.empty():
            self._released.get()
        for key, record in list(self._loads.items()):
            try:
                if record['stage'] == 'acquire':
                    if not record['future'].done():
                        continue
                    result = record['future'].result()
                    if result['status'] != 'ok':
                        record['error'] = result.get('message', 'Acquire failed')
                    if record['error'] or key in self._cancelled:
                        self._finish_read(record)
                    else:
                        self._begin_read(record, result)
                if record['stage'] == 'read':
                    status = self.agent.check_xfer_state(record['handle'])
                    if status == 'ERR':
                        record['error'] = 'NIXL READ failed'
                    if now >= record['deadline']:
                        record['error'] = 'NIXL READ timed out'
                    if status == 'PROC' and not record['error'] and key not in self._cancelled:
                        continue
                    self._finish_read(record)
                if record['stage'] == 'ack' and record['future'].done():
                    record['future'].result()
                    torch.cuda.synchronize(self.cache.device)
                    if key in self._cancelled:
                        self._events.cancelled.add(key)
                    elif record['error']:
                        self._events.failures.append(KVTransferFailure(key, record['error']))
                    else:
                        self._events.received.add(key)
                    del self._loads[key]
                    self._cancelled.discard(key)
            except Exception as error:
                if record['stage'] == 'ack':
                    # Local access has ended; report the failure. Remote P keeps
                    # its lease if the ACK was lost, instead of unsafe expiry.
                    self._events.failures.append(KVTransferFailure(key, str(error)))
                    del self._loads[key]
                    self._cancelled.discard(key)
                else:
                    record['error'] = str(error)
                    try:
                        self._finish_read(record)
                    except Exception:
                        pass  # Cancellation not confirmed: do not release blocks.
        result, self._events = self._events, ConnectorOutput()
        return result

    def has_pending_work(self):
        with self._lock:
            return bool(self._loads or self._published)

    def shutdown(self):
        if self._closed:
            return
        self._cancelled.update(self._loads)
        with self._lock:
            for record in self._published.values():
                record['cancelled'] = True
        deadline = time.monotonic() + self.config.transfer_timeout
        while self.has_pending_work() and time.monotonic() < deadline:
            self.get_finished()
            time.sleep(0.01)
        if self.has_pending_work():
            raise RuntimeError("Active remote access prevents safe KV shutdown; stop peers first")
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._thread.join()
        self._pool.shutdown(wait=True)
        if self.registration is not None:
            self.agent.deregister_memory(self.registration)
        for agent in self._agents.values():
            self.agent.remove_remote_agent(agent)
        self._agents.clear()
        self.registration = None
        # Destroy UCX while the CUDA primary context is still alive.
        self.agent = None
        self.cache = None
        self._closed = True
