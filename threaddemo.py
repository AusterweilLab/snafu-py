import time
import threading
import random

def print_time( threadName, delay):
  x.append(random.random)


x=[]

while len(x) < 1000:
    if len(threading.enumerate()) < 100:
        threadname="Thread-"+str(len(threading.enumerate()))
        threading.Thread(target=print_time,args=[threadname, 2]).start()
