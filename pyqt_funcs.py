from PyQt5 import QtWidgets
import graphical_panel.popup_window as POPUP
import multiprocessing as mp


def as_process(func):
    """
    Returns a version of func that runs as a separate process
    but returns output like a regular function. This can be used
    as the decorator '@as_process'.
    Applications:
        - forces Tensorflow to release GPU memory when
        the session is done
        - avoids a namespace clash between JAX and Tensorflow
        versions of XLA
    """
    def func_wrapper(*args, **kwargs):
        def queued_func(func, args, kwargs, queue):
            queue.put(func(*args, **kwargs))

        q = mp.Queue()
        p = mp.Process(target=queued_func, args=(func, args, kwargs, q))
        p.start()
        p.join()
        return q.get()
    return func_wrapper

@as_process
def posttrial_popup(sys_argv):
    app = QtWidgets.QApplication(sys_argv)
    w = POPUP.Window()
    # w.setWindowTitle('User Input')
    # w.show()
    retval = [None] * 3
    i = 0
    for ch in w.get_data():
        retval[i] = ch
        i = i + 1
    return retval