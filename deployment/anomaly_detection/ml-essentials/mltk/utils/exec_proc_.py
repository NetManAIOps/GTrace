import os
import signal
import subprocess
import sys
import types
from contextlib import contextmanager
from logging import getLogger
from threading import Thread, Semaphore
from typing import *

__all__ = ['timed_wait_proc', 'exec_proc']


def timed_wait_proc(proc: subprocess.Popen, timeout: float) -> Optional[int]:
    """
    Wait a process for at most `timeout` seconds.

    Args:
        proc: The process to wait.
        timeout: The timeout seconds.

    Returns:
        The exit code, or :obj:`None` if the process does not exit.
    """
    try:
        return proc.wait(timeout)
    except subprocess.TimeoutExpired:
        return None


def recursive_kill(proc: subprocess.Popen,
                   ctrl_c_timeout: float = 3,
                   kill_timeout: float = 10) -> Optional[int]:
    """
    Recursively kill a process tree.

    Args:
        proc: The process to kill.
        ctrl_c_timeout: Seconds to wait for the program to respond to
            CTRL+C signal.
        kill_timeout: Seconds to wait for the program to be killed.

    Returns:
        The return code, or None if the process cannot be killed.
    """
    if sys.platform != 'win32':
        try:
            gid = os.getpgid(proc.pid)
        except Exception:
            # indicate pid does not exist
            return

        def kill_fn(s):
            os.killpg(gid, s)
    else:  # pragma: no cover
        def kill_fn(s):
            if s == signal.SIGINT:
                os.kill(proc.pid, signal.CTRL_C_EVENT)
            else:
                proc.terminate()

    # try to kill the process by ctrl+c
    kill_fn(signal.SIGINT)
    code = timed_wait_proc(proc, ctrl_c_timeout)
    if code is None:
        getLogger(__name__).info(
            f'Failed to kill sub-process {proc.pid} by SIGINT, plan to kill '
            f'it by SIGTERM or SIGKILL.')
    else:
        return code

    # try to kill the process by SIGTERM
    if sys.platform != 'win32':
        kill_fn(signal.SIGTERM)
        code = timed_wait_proc(proc, kill_timeout)
        if code is None:
            getLogger(__name__).info(
                f'Failed to kill sub-process {proc.pid} by SIGTERM, plan to '
                f'kill it by SIGKILL.')
        else:
            return code

    # try to kill the process by SIGKILL
    kill_fn(signal.SIGKILL)
    code = timed_wait_proc(proc, kill_timeout)
    if code is None:
        getLogger(__name__).info(
            f'Failed to kill sub-process {proc.pid} by SIGKILL, give up.')

    return code


@contextmanager
def exec_proc(args: Union[str, Iterable[str]],
              on_stdout: Callable[[bytes], None] = None,
              on_stderr: Callable[[bytes], None] = None,
              stderr_to_stdout: bool = False,
              buffer_size: int = 16 * 1024,
              ctrl_c_timeout: float = 3,
              **kwargs) -> Generator[subprocess.Popen, None, None]:
    """
    Execute an external program within a context.

    Args:
        args: Command line or arguments of the program.
            If it is a command line, then `shell = True` will be set.
        on_stdout: Callback for capturing stdout.
        on_stderr: Callback for capturing stderr.
        stderr_to_stdout: Whether or not to redirect stderr to stdout?
            If specified, `on_stderr` will be ignored.
        buffer_size: Size of buffers for reading from stdout and stderr.
        ctrl_c_timeout: Seconds to wait for the program to respond to
            CTRL+C signal.
        \\**kwargs: Other named arguments passed to :func:`subprocess.Popen`.

    Yields:
        The process object.
    """
    # check the arguments
    if stderr_to_stdout:
        kwargs['stderr'] = subprocess.STDOUT
        on_stderr = None
    if on_stdout is not None:
        kwargs['stdout'] = subprocess.PIPE
    if on_stderr is not None:
        kwargs['stderr'] = subprocess.PIPE

    # output reader
    def reader_func(fd, action):
        while not stopped[0]:
            buf = os.read(fd, buffer_size)
            if not buf:
                break
            action(buf)

    def make_reader_thread(fd, action):
        try:
            th = Thread(target=reader_func, args=(fd, action))
            th.daemon = True
            th.start()
            return th
        finally:
            reader_sem.release()

    reader_sem = Semaphore()
    expected_sem_target = int(on_stdout is not None) + int(on_stderr is not None)

    # internal flags
    stopped = [False]

    # launch the process
    stdout_thread = None  # type: Thread
    stderr_thread = None  # type: Thread
    if isinstance(args, (str, bytes)):
        shell = True
    else:
        args = tuple(args)
        shell = False

    if sys.platform != 'win32':
        kwargs.setdefault('preexec_fn', os.setsid)
    proc = subprocess.Popen(args, shell=shell, **kwargs)

    # patch the kill() to ensure the whole process group would be killed,
    # in case `shell = True`.
    def my_kill(self, ctrl_c_timeout=ctrl_c_timeout):
        recursive_kill(self, ctrl_c_timeout=ctrl_c_timeout)

    proc.kill = types.MethodType(my_kill, proc)

    try:
        if on_stdout is not None:
            stdout_thread = make_reader_thread(proc.stdout.fileno(), on_stdout)
        if on_stderr is not None:
            stderr_thread = make_reader_thread(proc.stderr.fileno(), on_stderr)

        for i in range(expected_sem_target):
            reader_sem.acquire()

        try:
            yield proc
        except KeyboardInterrupt:  # pragma: no cover
            if proc.poll() is None:
                # Wait for a while to ensure the program has properly dealt
                # with the interruption signal.  This will help to capture
                # the final output of the program.
                # TODO: use signal.signal instead for better treatment
                _ = timed_wait_proc(proc, 1)

    finally:
        if proc.poll() is None:
            proc.kill()

        # gracefully stop the reader without setting `stopped = True` for
        # a couple of time, so as to ensure the remaining content are read out.
        for th in (stdout_thread, stderr_thread):
            if th is not None:
                try:
                    th.join(3000)
                except TimeoutError:
                    pass

        # Force setting the stopped flag, and wait for the reader threads to exit.
        stopped[0] = True
        for th in (stdout_thread, stderr_thread):
            if th is not None:
                th.join()

        # Ensure all the pipes are closed.
        for f in (proc.stdout, proc.stderr, proc.stdin):
            if f is not None:
                try:
                    f.close()
                except Exception:  # pragma: no cover
                    getLogger(__name__).info(
                        'Failed to close a sub-process pipe.',
                        exc_info=True
                    )
