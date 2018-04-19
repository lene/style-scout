class WithVerbose:

    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose

    def _print_status(self, *args, **kwargs) -> None:  # type: ignore
        if self.verbose:
            print(*args, **kwargs)
