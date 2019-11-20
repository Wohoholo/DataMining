# DataMining Apriori

import copy
from itertools import combinations

class Aprori:
    def __init__(self, dataFile=None):

        self.dataFile = dataFile
        self.data = self._dataRead()
        self._initail()


    def _initail(self):
        d = []
        for i in range(len(self.data)):
            d.extend(self.data[i])

        self.new_d = sorted(list(set(d)))  # extract all 1-item set



    def _dataRead(self):
        if self.dataFile is None:
            raise TypeError("dataFile could'n be None")

        data = []

        with open(self.dataFile, 'r') as f:
            self.T = int(f.readline().strip())
            self.minsp = int(f.readline().strip())

            for line in f.readlines():
                data.append(line.strip().split(' '))
        return data

    def generate_Ck(self, Lsub1, k):

        # print(list(Lsub1.keys()))
        Ck = {}
        for cb in combinations(self.new_d, k):
            flag = 1

            for i in combinations(cb, k-1):
                # print(i)
                if not self.is_apriori(i, Lsub1):
                    flag = 0
                    break
            if flag == 1:
                temp = frozenset(sorted(cb))
                cb = set(sorted(cb))
                for d in self.data:
                    if cb.issubset(d):
                        if temp not in Ck:
                            Ck[temp] = 1
                        else:
                            Ck[temp] += 1

        return Ck

    def generate_Lk(self, Ck):
        Lk = copy.deepcopy(Ck)
        for k, v in list(Lk.items()):
            # print(k, v)
            if v < 2:
                Lk.pop(k)

        return Lk


    def is_apriori(self, new_cb, Lksub1):
        new_cb = set(new_cb)

        for i in Lksub1:
            if type(i) == str:
                temp = ''.join(new_cb)
                if temp == i:
                    return True
            elif new_cb == set(i):
                return True

        return False

    def displayLk(self, Lk):
        for k, v in list(Lk.items()):
            print(list(sorted(k)), v)

    def displayCk(self, Ck):
        for k, v in list(Ck.items()):
            print(list(sorted(k)), v)

    def Mining(self):

        C = {}
        L = {}
        K = 2

        # generate C1 and L1
        for i in self.new_d:
            for d in self.data:
                if i in d:
                    if i not in C:
                        C[i] = 1
                    else:
                        C[i] += 1

        for k, v in C.items():
            if v >= self.minsp:
                L[k] = v


        while True:
            C = self.generate_Ck(L, K)
            # self.displayCk(C)
            L = self.generate_Lk(C)
            # self.displayCk(L)
            K += 1
            # print(Lk.values())
            if max(list(L.values())) <= self.minsp:
                self.displayLk(L)
                break



if __name__ == '__main__':
    apri = Aprori('data.txt')
    print(apri.T)
    print(apri.minsp)
    print(apri.data)
    apri.Mining()
