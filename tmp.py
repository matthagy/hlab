
def tmp():
    for x in 1, 2, 3, 4, 5:
        print x

import dis
dis.dis(tmp)
