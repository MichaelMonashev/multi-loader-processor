import numpy as np
import queue
import os
import time

from multiprocessing import Event, Process, Queue
from multiprocessing.sharedctypes import RawArray

class MyDataProcessor(Process):
    def __init__(self, arrays, q_free, q_used, bail, dtype):
        super().__init__()
        self.arrays = arrays
        self.q_free = q_free
        self.q_used = q_used
        self.bail = bail
        self.dtype = dtype

    def run(self):
        # wrap RawArrays inside ndarrays
        arrays = [np.frombuffer(arr, dtype=self.dtype) for arr in self.arrays]

        while True:
            arr_id = None
            try:
                arr_id = self.q_used.get_nowait()    # wake up loader blocking on put()
                arr = arrays[arr_id]
                #print(arr_id, arr, len(arrays))

                print("Process", arr_id, arr.shape, arr.dtype, type(arr), flush=True)
                _ = arr[0]
            except queue.Empty:
                pass

            if self.bail.is_set():
                # вычитываем всё, что есть из очереди
                while True:
                    try:
                        self.q_used.get_nowait()
                    except queue.Empty:
                        break
                return

            # Записываем в другую очередь, если ранее что-то получили из первой
            if arr_id is not None:
                self.q_free.put(arr_id)

class MyDataLoader(Process):
    def __init__(self, arrays, q_free, q_used, bail, dtype):
        super().__init__()
        self.arrays = arrays
        self.q_free = q_free
        self.q_used = q_used
        self.bail = bail
        self.dtype = dtype

    def run(self):
        # wrap RawArrays inside ndarrays
        arrays = [np.frombuffer(arr, dtype=self.dtype) for arr in self.arrays]

        while True:
            arr_id = None
            try:
                arr_id = self.q_free.get_nowait()
                arr = arrays[arr_id]
                #print(arr_id, arr)

                print("Load data to", arr_id, arr.shape, arr.dtype, type(arr), flush=True)
                arr[0] = 123
            except queue.Empty:
                pass

            if self.bail.is_set():
                # вычитываем всё, что есть из очереди
                while True:
                    try:
                        self.q_free.get_nowait()
                    except queue.Empty:
                        break
                return

            # Записываем в другую очередь, если ранее что-то получили из первой
            if arr_id is not None:
                self.q_used.put(arr_id)

def create_shared_arrays(shape, dtype, num):
    # размер - произведение всех размерностей
    size = np.prod(shape).item()
    dtype = np.dtype(dtype)
    if dtype.isbuiltin and dtype.char in 'bBhHiIlLfd':
        typecode = dtype.char
    else:
        typecode, size = 'B', size * dtype.itemsize

    arrays = []
    for i in range(num):
        arr = RawArray(typecode, size)
        arrays.append(arr)
        print('Create', i+1, 'array of', num, 'with typecode', typecode, 'and size', size)

    return arrays


def main():
    arr_dtype = np.float32

    max_image_size = 20 # 16384 # 6144
    arr_shape = (max_image_size, max_image_size, 3)
    num_of_arrays = os.cpu_count()

    arrays = create_shared_arrays(arr_shape, dtype=arr_dtype, num=num_of_arrays)
    q_free = Queue(len(arrays))
    q_used = Queue(len(arrays))
    bail = Event()

    # заполняем очередь индексами массивов
    for arr_id in range(len(arrays)):
        q_free.put(arr_id)

    loaders = []
    num_of_loaders = os.cpu_count()
    for _ in range(num_of_loaders):
        pr = MyDataLoader(arrays, q_free, q_used, bail, dtype=arr_dtype)
        loaders.append(pr)

    processors = []
    num_of_processors = os.cpu_count()
    for _ in range(num_of_processors):
        pr = MyDataProcessor(arrays, q_free, q_used, bail, dtype=arr_dtype)
        processors.append(pr)

    for pr in loaders:
        pr.start()
        print("\n{} started.".format(pr.name))

    for pr in processors:
        pr.start()
        print("\n{} started.".format(pr.name))

    time.sleep(10)
    bail.set()

    for pr in processors:
        pr.join()
        print("\n{} joined.".format(pr.name))

    for pr in loaders:
        pr.join()
        print("\n{} joined.".format(pr.name))

if __name__ == '__main__':
    main()
