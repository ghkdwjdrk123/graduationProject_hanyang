class Print():
    def __init__(self):
        print("This is init.\n")

    def __call__(self):
        print("This is call.\n")

def pp():
    p = Print()
    p()

if __name__ == '__main__':
    pp()
