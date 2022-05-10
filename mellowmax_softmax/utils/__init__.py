def sigterm_decorator(callback=None):

    def decorator(func):

        def term():
            try:
                func()
            except KeyboardInterrupt as e:
                if callback is not None:
                    callback()
                print('Program Terminate')
                exit(1)

        return term

    return decorator
