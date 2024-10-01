class Match:

    def __init__(self, input_start, output_start, length):
        self.input_start = input_start
        self.output_start = output_start
        self.length = length
        self.input_end = input_start + length
        self.output_end = output_start + length
        