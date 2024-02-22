from threading import Thread
import time
import torch


def load_data(value):
    # load the data from disk
    t0 = time.time()
    # data = torch.load("test_DELETE.pt")
    time.sleep(5)
    print(f"Loading took {time.time() - t0} seconds")
    return {"data": value}


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


class Data:
    def __init__(self):
        self.values = list(range(50))
        self.threads = []

    def __getitem__(self, item):
        # I need to add to the queue if the queue length is < 10, not if it's empty
        print(f"Queue length: {len(self.threads)}")
        if len(self.threads) < 10:
            for i in range(10 - len(self.threads)):
                if self.values:
                    print(f"Loading {self.values[0]}")
                    value = self.values.pop(0)
                    t = ThreadWithReturnValue(target=self._load_data, args=(value,))
                    t.start()
                    self.threads.append(t)

        # need to wait for the queue to be populated
        value = self.threads.pop(0).join()

        return value

    def _load_data(self, value):
        data = load_data(value)
        return data


def main():
    data = Data()
    for i in range(50):
        print(data[i])
        time.sleep(0.5)


if __name__ == "__main__":
    main()
