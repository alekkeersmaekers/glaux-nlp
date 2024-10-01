class ReplaceNode:
    
    def __eq__(self,other):
        return hash(self) == hash(other)
    
    def __hash__(self):
        prime = 31
        result = 1
        result = prime * result + (0 if self.input is None else hash(self.input))
        result = prime * result + (0 if self.output is None else hash(self.output))
        return result
    
    def __init__(self, input, output):
        self.input = input
        self.output = output        
    
    def __str__(self):
        return "r({}, {})".format(self.input, self.output)
    
    def getCost(self, builder):
        count = builder.counter.count(str(self)) + 1
        cost = (len(self.input) + len(self.output)) / count
        return cost
    
    def apply(self, input, start, end):
        assert start >= 0
        assert end <= len(input)
    
        length = end - start
    
        if length != len(self.input):
            return None
    
        if input[start:end] != self.input:
            return None
    
        return self.output