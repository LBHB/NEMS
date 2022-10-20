import os, sys


# https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
# Alexander C
class PrintingBlocked:
    """Context manager for temporarily blocking `sys.stdout`.
    
    Examples
    --------
    >>> def test(i):
    >>>     print(f'{i} not blocked')
    >>>
    >>> with PrintingBlocked():
    >>>     test(1)
    >>> test(2)
    2 not blocked
    
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# https://stackoverflow.com/questions/3160699/python-progress-bar
# imbr
def progress_bar(iterable, prefix="", character='#', size=60, out=sys.stdout):
    """Simple text-based progress bar.

    Parameters
    ----------
    iterable : iterable.
    prefix : str; default="".
        Fixed text header placed to the left of the progress bar.
    character : str; default='#'.
        Single-character string to represent progress increments.
    size : int; default=60.
        Number of characters to use to represent the progress bar.
    out : I/O stream; default=sys.stdout.
    
    Examples
    --------
    >>> import time    
    >>> for i in progressbar(range(15), "Computing: ", 40):
    >>>     time.sleep(0.1)  # any code you need
    
    """
    count = len(iterable)
    def show(j):
        x = int(size*j/count)
        print(f"{prefix}[{character*x}{('.'*(size-x))}] {j}/{count}", end='\r',
              file=out, flush=True)
    show(0)
    for i, item in enumerate(iterable):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)
