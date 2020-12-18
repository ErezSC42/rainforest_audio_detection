import multiprocessing as mp

def processInput(i):
        return i * i

if __name__ == '__main__':

    # what are your inputs, and what operation do you want to
    # perform on each input. For example...
    inputs = range(1000000)
    #  removing processes argument makes the code run on all available cores
    pool = mp.Pool(processes=4)
    out1, out2, out3 = zip(*pool.map(processInput, range(0, 10 * offset, offset)))
    print(results)