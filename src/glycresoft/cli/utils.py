class ctxstream(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, string):
        self.stream.write(string)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, attrib: str):
        attr = getattr(self.stream, attrib)
        return attr

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
