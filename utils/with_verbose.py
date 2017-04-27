class WithVerbose:

    def __init__(self, verbose):
        self.verbose = verbose

    def _print_status(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
