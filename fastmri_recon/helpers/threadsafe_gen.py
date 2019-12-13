import threading

# from https://gist.github.com/platdrag/e755f3947552804c42633a99ffd325d4
'''
    A generic iterator and generator that takes any iterator and wrap it to make it thread safe.
    This method was introducted by Anand Chitipothu in http://anandology.com/blog/using-iterators-and-generators/
    but was not compatible with python 3. This modified version is now compatible and works both in python 2.8 and 3.0 
'''
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g
