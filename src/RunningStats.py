import math


class RunningStats():
    def __init__(self):
        self._count = 0
        self._mean = 0
        self._dsquared = 0

    def update(self, new_val):
        self._count += 1
        mean_diff = (new_val - self._mean) / self._count
        new_mean = self._mean + mean_diff
        self._dsquared += (new_val - new_mean) * (new_val - self._mean)
        self._mean = new_mean

    def get_count(self):
        return self._count
        
    def get_mean(self):
        if self._count == 0:
            raise StatsError('require number of samples at least 1')
        return self._mean

    def get_std(self):
        if self._count == 0:
            raise StatsError('require number of samples at least 1')
        return math.sqrt(self._dsquared / self._count)
    
    def clear(self):
        self._count = 0
        self._mean = 0
        self._dsquared = 0


class StatsError(Exception):
    def __init__(self, message):
        self.message = message