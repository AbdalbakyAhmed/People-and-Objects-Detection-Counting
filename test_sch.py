import sched, time

s = sched.scheduler(time.time, time.sleep)
def print_time(a='default'):
    print("From print_time", time.time(), a)
def test():
    while True:
        print("other !!")
def print_some_times():
    print(time.time())
    s.enter(0, 5, test)
    s.enter(5, 10, print_time, argument=('positional',))
    s.enter(10, 20, print_time, kwargs={'a': 'keyword'})
    s.run()
    print(time.time())

while True:
    print_some_times()
