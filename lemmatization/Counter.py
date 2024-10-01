from collections import defaultdict

class Counter:
    
    def __init__(self, defaultValue=0.0):
        self.storage = defaultdict(float)
        self.default_value = defaultValue
        self.total_count = 0.0
        
    def count(self, item):
        return self.storage.get(item, self.default_value)