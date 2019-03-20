#!/usr/bin/python3

"""Script to utilize a Boltzmann Machine to obtain a feasible solution to the Travelling Salesman Problem"""
from random import randint, uniform
import math


class Boltzmann(object):
    """Class to define the architecture of the Boltzmann Machine for the Travelling Salesman Problem"""

    def __init__(self):
        """Constructor"""
        self.p = randint(2, 50)
        self.b = randint(1, self.p)
        self.T = uniform(20, 500)
        self.N = 3
        self.dists = None
        self.Umat = None
        self.epochs = 150
        print("NET initialized with the following values: ")
        print("p: ", self.p, "b: ", self.b, "T: ", self.T)
        print("Running for ", self.epochs, " epochs!")

    def get_input(self):
        """Inputs the number of cities, and the details of the distances between each of them"""
        self.N = int(input("Enter the number of cities: "))
        self.dists = [[0 for _ in range(self.N)] for _i in range(self.N)]
        print("Enter the distances for each pair of cities:")
        num_conn = int(self.N * (self.N - 1) / 2)
        print("Enter", num_conn, "entries in each line. Format is:- C1 C2 distance")
        for _ in range(num_conn):
            con_li = list(map(int, input().strip().split()))
            self.dists[con_li[0]][con_li[1]] = con_li[2]
            self.dists[con_li[1]][con_li[0]] = con_li[2]
        self.Umat = [[randint(0, 1) for _ in range(self.N)] for _i in range(self.N)]

    def TSP(self):
        """Performs the updates using TSP"""
        print("Umat:")
        for row in self.Umat:
            print(row)
        for row in self.dists:
            print(row)
        # For each epoch
        for e in range(self.epochs):
            # In each epoch, do N^2 times
            for _ in range(self.N*self.N):
                I = randint(0, self.N-1)
                J = randint(0, self.N-1)
                calc = self.b
                # Calculate row
                for r in range(self.N):
                    if not r==J:
                        calc += -1 * self.p * self.Umat[I][r]
                # Calculate Column
                for c in range(self.N):
                    if not c == I:
                        calc += -1*self.p*self.Umat[c][J]
                # Calculate for all neighbouring cities
                for row in range(self.N):
                    if not row == I:
                        calc += -1*self.dists[row][I]*self.Umat[row][(J-1)%self.N]
                        calc += -1 * self.dists[row][I] * self.Umat[row][(J + 1) % self.N]
                delC = (1 - self.Umat[I][J]) * calc
                # print(self.Umat[I][J], delC)
                AofT = 1/(1 + math.exp(-1*delC/self.T))
                R = uniform(0, 1)
                if R < AofT:
                    self.Umat[I][J] = 1 - self.Umat[I][J]
            # Anneal the parameter
            self.T = 0.95 * self.T
            print("Epoch Number: ", e)
            for row in self.Umat:
                print(row)


def main():
    bm = Boltzmann()
    bm.get_input()
    bm.TSP()


if __name__ == "__main__":
    main()
