#ifndef KNAPSACK
#define KNAPSACK

#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <deque>
#include <unistd.h>

using namespace std;

#define max(a, b) (a > b ? a : b)
#define min(a, b) (a < b ? a : b)
#define END (-1)
#define RUN 1

// more items means more branches and solutions computed at once.
// more kernels means more solutions getting explored at once.
#define MAX_KERNEL_COUNT 4096
#define NUM_ITEMS_AT_ONCE 10    

typedef struct {
    int weight;
    int value;
}Item;

// we store the current value and weight of the bag, obviously. we also store the current item, what this tell us is which items we need to compute next.
// since we are computing many items at once for a given solution, and then we go do another solution, we need to know where to pick back up. Our own little process control block. 
typedef struct {
    int currentValue;
    int currentWeight;
    int currentItem;
    char *itemChoices;
    int upperBound;
}Solution;

ifstream inputFile;

// number of total items we have
int NUM_ITEMS;

// the capacity of the knapsack
int capacity;

// all our items
vector<Item> itemChoices;

// the optimal solution we read in
vector<char> optimalSolution;

// variable that tracks at a given item, what the best value per weight is for everything after it
float *bestPerWeightValues;

// make it thread safe by using atomic
int globalBest;
char *globalBestChoices;

// our array of solution queues.
// they are going to be able of performing load balancing upon each other to steal processes from one another in order to stay busy. 
int RUN_STATUS; 

// function definitions
void checkInputs(int, char *[]);
void readFile(ifstream *);
void initializeBestPerUnitWeights();
__global__ void greedySearch(int *, int *, char *, int *, int, int);
void launchGreedySearch();
__global__ void computeNextItems(int *, int *, int *, int *, int *, float, int, int, int, int, int);
void launchBranchAndBound();

#endif
