import functools
import time


def timer(message="Duration: ", precision=3):
    def timer_inner(fn):
        @functools.wraps(fn)
        def fn_with_time(*args, **kwargs):
            start = time.time()
            ret = fn(*args, **kwargs)
            print(f"{message}{round(time.time() - start, precision)}")

            return ret

        return fn_with_time

    return timer_inner
