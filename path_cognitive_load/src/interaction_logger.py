"""Interaction behavior tracking - 7 features"""
import time
from collections import deque


class InteractionLogger:
    def __init__(self):
        self.events = deque(maxlen=500)
        self.question_start_time = None
        self.reading_start_time = None
        self.correction_count = 0
        self.keypress_count = 0

    def log_event(self, event_type, data=None):
        self.events.append({
            'timestamp': time.time(),
            'type': event_type,
            'data': data or {}
        })
        if event_type == 'backspace':
            self.correction_count += 1
        elif event_type == 'keypress':
            self.keypress_count += 1

    def start_question(self):
        self.question_start_time = time.time()
        self.correction_count = 0
        self.keypress_count = 0

    def start_reading(self):
        self.reading_start_time = time.time()

    def get_features(self):
        now = time.time()
        tpq = now - self.question_start_time if self.question_start_time else 0
        scroll_evts = [e for e in self.events if e['type'] == 'scroll']
        scroll_speed = (sum(e['data'].get('delta', 0) for e in scroll_evts)
                        / max(len(scroll_evts), 1))
        dirs = [1 if e['data'].get('delta', 0) > 0 else -1 for e in scroll_evts]
        d_changes = sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i - 1])
        corr = self.correction_count / max(self.keypress_count, 1)
        revisits = len([e for e in self.events if e['type'] == 'revisit'])
        r2a = 0
        if self.reading_start_time and self.question_start_time:
            r2a = self.question_start_time - self.reading_start_time
        return [tpq, 20, scroll_speed, d_changes, corr, revisits, r2a]

    def reset(self):
        self.events.clear()
        self.question_start_time = None
        self.reading_start_time = None
        self.correction_count = 0
        self.keypress_count = 0