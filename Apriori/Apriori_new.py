import json
import csv

class Apriori:

    def __init__(self, dataFile):

        self.dataFile = dataFile
        self.data = self._readCsv()

    def _readCsv(self):
        if self.dataFile is None:
            raise TypeError('DataFile can not be None')

        data = []
        self.min_sup = 2
        with open('data.csv', 'r') as f:
            reader = csv.reader(f)
            items = next(reader)
            print(items[7:16])
            for row in reader:
                tmp = []
                for i in range(len(row)):
                    if row[i] == 'Yes':
                        tmp.append(items[i])
                if tmp != []:
                    data.append(tmp)

        return data

    def _readData(self):
        if self.dataFile is None:
            raise TypeError('DataFile can not be None')

        data = []
        with open(self.dataFile, 'r') as f:
            self.T = int(f.readline().strip())
            self.min_sup = int(f.readline().strip())

            for line in f.readlines():
                data.append(line.strip().split(' '))
        return data


    def run(self, out_file):
        CK, LK = self.generate_C1_and_L1()
        k = 1
        f = open(out_file, 'w')
        while True:
            k += 1
            CK = self.generate_CK_by_LKsub1(LK, k)
            LK, support = self.generate_LK_by_CK(CK)

            if LK and max(list(LK.values())) >= self.min_sup:
                f.write('C'+str(k)+':\n')
                for key, value in CK.items():
                    text = ','.join([str(i) for i in key]) + ': ' + str(value) +'\n'
                    f.write(text)
                f.write('L' + str(k) + ':\n')
                for key, value in LK.items():
                    text = ','.join([str(i) for i in key]) + ': ' + str(value) +'\n'
                    f.write(text)
                f.write('support:\n')
                for key, value in support.items():
                    text = ','.join([str(i) for i in key]) + ': ' + str(value) +'\n'
                    f.write(text)
            else:
                break
        f.close()

    def generate_C1_and_L1(self):
        C1 = {}

        for item in self.data:
            for i in item:
                if i not in C1:
                    C1[i] = 0
                C1[i] += 1

        L1 = {frozenset([key,]):value for key, value in C1.items() if value >= self.min_sup}
        self.item = list(C1.keys())
        return C1, L1

    def is_apriori(self, item, LKsub1):
        list_LK = [sorted(list(key)) for key in LKsub1.keys()]
        for i in item:
            sub_i = item[:]
            sub_i.pop(item.index(i))
            if type(sub_i) is str:
                sub_i = [sub_i]
            if sub_i not in list_LK:
                return False
        return True


    def count_subset(self, l1, l2):
        count = 0
        for l in l2:
            if set(l1) <= set(l):
                count += 1
        return count

    def generate_CK_by_LKsub1(self, LKsub1, k):
        CK = {}
        list_LK =sorted([sorted(list(key)) for key in LKsub1.keys()])
        l = len(list_LK)

        for i in range(l-1):
            for j in range(i+1, l):
                t1 = sorted(list_LK[i])
                t2 = sorted(list_LK[j])

                if t1[0:k-2] == t2[0:k-2]:
                    t1.append(t2[-1])

                    if self.is_apriori(t1, LKsub1):
                        if frozenset(t1) not in CK:
                            CK[frozenset(t1)] = 0
                        CK[frozenset(t1)] += self.count_subset(t1, self.data)

        return CK


    def generate_LK_by_CK(self, CK):
        LK = {}

        for key, value in CK.items():
            if value >= self.min_sup:
                LK[key] = value
            """
            can calculate confidence and support
            """

        support = {}
        for key, value in LK.items():
            support[key] = float(value/len(self.data))

        return LK, support


if __name__ == '__main__':

    a = Apriori('data.csv')
    a.run('output.txt')
