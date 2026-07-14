"""Mouse / cursor tracking — 8 features.
Uses update(x, y) from tkinter <Motion> — no pyautogui needed.
"""
import time
import math
from collections import deque


class MouseTracker:
    def __init__(self):
        self.trail = deque(maxlen=200)
        self.speeds = deque(maxlen=60)
        self.last_pos = (0, 0)
        self.last_time = time.time()

    def update(self, x, y):
        """Called from tkinter <Motion> event"""
        now = time.time()
        dx = x - self.last_pos[0]
        dy = y - self.last_pos[1]
        dist = math.sqrt(dx ** 2 + dy ** 2)
        dt = now - self.last_time
        speed = dist / dt if dt > 0.001 else 0.0

        self.trail.append({
            'x': x, 'y': y, 'time': now,
            'distance': dist, 'dx': dx, 'dy': dy,
            'speed': speed, 'dt': dt
        })
        self.speeds.append(speed)
        self.last_pos = (x, y)
        self.last_time = now

    @property
    def avg_speed(self):
        s = [v for v in self.speeds if v > 0]
        return sum(s) / len(s) if s else 0.0

    @property
    def jitter(self):
        s = [v for v in self.speeds if v > 0]
        if len(s) < 2:
            return 0.0
        avg = sum(s) / len(s)
        return math.sqrt(sum((v - avg) ** 2 for v in s) / len(s))

    @property
    def idle_seconds(self):
        return time.time() - self.last_time

    def get_features(self):
        if len(self.trail) < 2:
            return [0.0] * 8
        speeds = [v for v in self.speeds if v > 0]
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        jit = self.jitter
        acc = (speeds[-1] - speeds[0]) / len(speeds) if len(speeds) > 1 else 0
        total_dist = sum(p['distance'] for p in self.trail)
        idle = self.idle_seconds
        directions = []
        for i in range(1, len(self.trail)):
            if self.trail[i]['dx'] != 0 or self.trail[i]['dy'] != 0:
                directions.append(math.atan2(self.trail[i]['dy'], self.trail[i]['dx']))
        d_changes = sum(1 for i in range(1, len(directions))
                        if abs(directions[i] - directions[i - 1]) > math.pi / 2)
        s, e = self.trail[0], self.trail[-1]
        straight = math.sqrt((e['x'] - s['x']) ** 2 + (e['y'] - s['y']) ** 2)
        eff = straight / (total_dist + 0.001)
        return [avg_speed, jit, acc, total_dist, idle, d_changes, 0, eff]

    def reset(self):
        self.trail.clear()
        self.speeds.clear()
        self.last_pos = (0, 0)
        self.last_time = time.time()