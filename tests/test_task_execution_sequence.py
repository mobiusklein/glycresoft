import time
import random
from glycresoft.task import TaskExecutionSequence, Pipeline

class MyTask(TaskExecutionSequence):
    def __init__(self, threshold):
        self.threshold = threshold

    def run(self, *args, **kwargs):
        while not self.error_occurred():
            self.log("I am alive!")
            time.sleep(1)
            if random.random() > self.threshold:
                self.log("Erroring out!")
                raise Exception("Failure")
        self.log("Done!")


if __name__ == "__main__":
    task = MyTask(0.2)
    task.start(True, True)
    task2 = MyTask(1.0)
    task2.start(False, True)
    pipe = Pipeline([task, task2])
    pipe.join()
    pipe.log("Stopping")
    pipe.stop()
    pipe.join(10)
    pipe.log("Stopped", task.is_alive(), pipe.error_occurred())
