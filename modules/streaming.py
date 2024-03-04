import queue
import threading
from typing import Optional, Union, Any, Callable
import logging
import copy

class Streaming:
    queue_: queue.Queue
    lock_: threading.Lock
    event_: threading.Event

    action_data_: Any

    def __init__(
        self,
        action: Callable,
        sync: Optional[bool] = False,
        limit: Optional[int] = -1
    ) -> None:
        self.queue_ = queue.Queue()
        self.lock_ = threading.Lock()
        self.event_ = threading.Event()

        self.sync_ = sync
        self.limit_ = limit
        
        self.action_ = action
        self.action_data_ = None

        self.thread_ = threading.Thread(target=self.work_loop_)
        self.thread_.start()
        

    def push(self, data: Any) -> int:
        with self.lock_:
            if self.limit_ > 0 and self.queue_.qsize() > self.limit_:
                print('Too much data in the queue, dropping')
                filter_ = int(self.limit_ * 0.6)
                while self.queue_.qsize() > filter_:
                    self.queue_.get()

        self.queue_.put(data)
        
        if self.sync_:
            self.event_.clear()
        return self.queue_.qsize()

    def get(self) -> Any:
        if self.sync_:
            self.event_.wait()
        
        return copy.deepcopy(self.action_data_)

    def work_loop_(
        self,
    ) -> None:
        print('Streaming thread started')
        while True:
            data = self.queue_.get()
            
            if data is None: break
            # skipping
            if not self.sync_:
                with self.lock_:
                    while not self.queue_.empty():
                        data = self.queue_.get()
                if data is None: break

            self.action_data_ = self.action_(data)

            if self.sync_:
                self.event_.set()
        print('Streaming thread stopped')


