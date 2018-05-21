import time

class timer():
    def __init__(self, name):
        self.start_time = time.time()
        self.name = name

    def passed(self):
        return time.time() - self.start_time

    def estimate_extra_time(self, finished_blocks, total_blocks):
        return self.passed() / finished_blocks * total_blocks

    def log(self, finished_blocks, total_blocks):
        print('timer for {name}. time flies very fast .. {passed_time:.2f} mins passed, about {extra:.2f} mins left... step 2'.format(
            name=self.name,
            passed_time=self.passed() / 60,
            extra=self.estimate_extra_time(finished_blocks, total_blocks)/ 60)
        )

