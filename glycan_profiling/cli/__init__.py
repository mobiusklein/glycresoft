from .logger_config import configure_logging

configure_logging()


# http://stackoverflow.com/a/15472811/1137920
def _setup_win32_keyboard_interrupt_handler():
    import os
    from scipy import stats
    import thread
    import threading
    import win32api

    def handler(dwCtrlType, hook_sigint=thread.interrupt_main):
        if dwCtrlType == 0:
            hook_sigint()
            print("Keyboard Interrupt Received. Scheduling Interrupt in Main Thread...")
            return 1
        return 0

    win32api.SetConsoleCtrlHandler(handler, 1)


try:
    _setup_win32_keyboard_interrupt_handler()
except ImportError:
    pass

