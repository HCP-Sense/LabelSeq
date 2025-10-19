
class EventBus:
    def __init__(self):
        self._subs = {}

    def on(self, event, fn):
        self._subs.setdefault(event, []).append(fn)
        return fn

    def off(self, event, fn):
        if event in self._subs and fn in self._subs[event]:
            self._subs[event].remove(fn)

    def emit(self, event, *args, **kwargs):
        for fn in list(self._subs.get(event, [])):
            try:
                fn(*args, **kwargs)
            except Exception:
                pass
