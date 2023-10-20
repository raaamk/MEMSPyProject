class GeneralClass:

    val1 = 1
    val2 = 2
    val3 = 'Text'
    val4 = 4
    val5 = []

    # Constructor
    #------------
    def __init__(self, vala=5, valb=6, valc='bla'):
        self.val1 = vala
        self.val2 = valb
        self.val3 = valc

    def print_static_values(self):
        print(self.val1)
        print(self.val2)
        print(self.val3)

    def multiply_with_constant(self, factor):
        self.val5 = self.val4*factor