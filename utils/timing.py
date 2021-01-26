from datetime import datetime

class Timer:
    def __init__(self, silent = False):
        self.start = datetime.now()
        self.silent = silent
        if not silent: print(f"Started at {self.start}")
    def time(self):
        end = datetime.now()
        if not self.silent: print(f"Finished at {end}")
        return end - self.start