class WithVerbose:

    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose

    @classmethod
    def print_status(cls, verbose, *args, **kwargs) -> None:
        if verbose:
            print(*args, **kwargs)

    def _print_status(self, *args, **kwargs) -> None:
        self.print_status(self.verbose, *args, **kwargs)
