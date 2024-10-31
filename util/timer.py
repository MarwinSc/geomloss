import time
import logging

class Timer:

    def __init__(self, string="") -> None:
        self.tic(string=string)

    def tic(self, string=""):
        self.string = string
        self.start = time.time()

    def toc(self):
        end = time.time()
        logging.info(f"{self.string} in {end - self.start}")
        print(f"{self.string} in {end - self.start}")
