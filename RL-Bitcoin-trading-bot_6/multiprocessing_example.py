from multiprocessing import Process, Pipe
import time
import random

class Environment(Process): # creating environment class for multiprocessing
    def __init__(self, env_idx, child_conn):
        super(Environment, self).__init__()
        self.env_idx = env_idx
        self.child_conn = child_conn

    def run(self):
        super(Environment, self).run()
        while True:
            number = self.child_conn.recv()
            self.child_conn.send(number*2)

if __name__ == "__main__":
    works, parent_conns, child_conns = [], [], []
    
    for idx in range(2):
        parent_conn, child_conn = Pipe() # creating a communication pipe
        work = Environment(idx, child_conn) # creating new process 
        work.start() # starting process 
        works.append(work) # saving started procsses to list
        parent_conns.append(parent_conn) # saving communication pipe refference to list
        child_conns.append(child_conn) # saving communication pipe refference to list

    while True:
        for worker_id, parent_conn in enumerate(parent_conns):
            r = random.randint(0, 10) # creating random number between 0 and 10
            parent_conn.send(r) # sending message with random nuber to worker_id running process
            
        time.sleep(1)

        for worker_id, parent_conn in enumerate(parent_conns):
            result = parent_conn.recv() # reading received message from worker_id process
            print(f"From {worker_id} worker received {result}")
