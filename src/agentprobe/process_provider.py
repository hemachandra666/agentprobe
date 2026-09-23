"""Spawn-owned inference. Use a context manager; factories must be picklable.

One worker retains its model across actions/tasks. Tools remain in the parent.
A failed worker is never reused. This does not cancel remote Ollama servers.
"""
import multiprocessing as mp
import time


def _worker(connection, factory):
    try:
        provider = factory()
        while True:
            request = connection.recv()
            if request is None:
                return
            question, history = request
            connection.send((True, provider.act(question, history)))
    except EOFError:
        pass
    except BaseException as exc:
        try:
            connection.send((False, f"{type(exc).__name__}: {exc}"))
        except (OSError, EOFError):
            pass
    finally:
        connection.close()


class ProcessProvider:
    def __init__(self, factory):
        self.factory = factory
        self.process = None
        self.connection = None
        self.closed = False
        self.closed_connection = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self, *, force=False):
        self.closed = True
        p = self.process
        if (not force and not self.closed_connection and p is not None
                and p.pid is not None and p.is_alive()):
            try:
                self.connection.send(None)
                p.join(timeout=5)
            except (OSError, EOFError):
                pass
        if self.connection is not None:
            self.connection.close()
        self.closed_connection = True
        if p is not None and p.pid is not None:
            if p.is_alive():
                p.terminate()
            p.join(timeout=2)
            if p.is_alive():
                p.kill()
                p.join(timeout=2)
            if p.is_alive():
                raise RuntimeError("inference worker could not be reaped; stop this process")

    def act_before_deadline(self, question, history, seconds):
        if self.closed:
            raise RuntimeError("worker is closed; create a new ProcessProvider")
        deadline = time.monotonic() + seconds
        try:
            if self.process is None:
                ctx = mp.get_context("spawn")
                self.connection, child = ctx.Pipe()
                self.process = ctx.Process(target=_worker, args=(child, self.factory), daemon=True)
                try:
                    self.process.start()
                finally:
                    child.close()
            self.connection.send((question, list(history)))
            if not self.connection.poll(max(0, deadline - time.monotonic())):
                raise TimeoutError("inference worker exceeded deadline")
            ok, value = self.connection.recv()
            if not ok:
                raise RuntimeError(value)
            if time.monotonic() >= deadline:
                raise TimeoutError("inference worker exceeded deadline")
            return value
        except BaseException:
            self.close(force=True)
            raise
