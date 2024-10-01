class MatchNode:
    
    hash_code_ = 0
    
    def __eq__(self,other):
        return hash(self) == hash(other)
    
    def __hash__(self):
        if self.hash_code_ == 0:
            prime = 31
            result = 1
            result = prime * result + (0 if self.left is None else hash(self.left))
            result = prime * result + self.left_length
            result = prime * result + (0 if self.right is None else hash(self.right))
            result = prime * result + self.right_length
            self.hash_code_ = result
        return self.hash_code_

    def __init__(self,left,right,left_length,right_length):
        self.left_length = left_length
        self.right_length = right_length
        self.left = left
        self.right = right
    
    def __str__(self):
        string = "("
    
        if self.left is not None:
            string += str(self.left_length) + str(self.left)
    
        if len(string) > 1:
            string += " "
    
        if self.right is not None:
            string += " " + str(self.right_length) + str(self.right)
    
        string += ")"
    
        return string
    
    def apply(self, input, start, end):
        left = ""
        if self.left is not None:
            if start + self.left_length > end:
                return None
            left = self.left.apply(input, start, start + self.left_length)
            if left is None:
                return None
        right = ""
        if self.right is not None:
            if end - self.right_length < start:
                return None
            right = self.right.apply(input, end - self.right_length, end)
            if right is None:
                return None
        middle_length = end - start - self.left_length - self.right_length
        if middle_length <= 0:
            return None
        middle = input[start + self.left_length: start + self.left_length + middle_length]
        return left + middle + right
    
    def getCost(self,builder):
        cost = 0
        if self.left is not None:
            cost += self.left.getCost(builder)
        if self.right is not None:
            cost += self.right.getCost(builder)
        return cost