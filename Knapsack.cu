#include "Knapsack.cuh"

void checkInputs(int argc, char* argv[]){
    
    if (argc != 2) {
        cerr << "INVALID USAGE: ./knapsack <input file>" << endl;
        exit(1);
    }

    inputFile.open(argv[1]);
    if (!inputFile.is_open()) {
        cerr << "File does not exist or could not be opened" << endl;
        exit(1);
    }
}

void readFile(ifstream *inputFile) {
    // Read the first two things in the file as the number of items and capacity
    *inputFile >> NUM_ITEMS >> capacity;

    // Read each item (value, then weight) into the itemChoices vector
    for (int i = 0; i < NUM_ITEMS; i++) {
        Item item;
        *inputFile >> item.value >> item.weight; // Switch to value first, then weight
        itemChoices.push_back(item);
    }

    // Try reading the optimal solution if it's there
    if (*inputFile >> std::ws) {  // Check if there's more data
        optimalSolution.clear();  // Clear previous optimal solution if any
        for (int i = 0; i < NUM_ITEMS; i++) {
            int bit;
            *inputFile >> bit;
            optimalSolution.push_back(static_cast<char>(bit));
        }
    }

    cout << "Read " << NUM_ITEMS << " items with capacity " << capacity << endl;

}

void initializeBestPerUnitWeights(){
    // using this as a float * rather than vector makes passing it to the GPU easier later. 
    bestPerWeightValues = (float *)malloc(NUM_ITEMS * sizeof(float) + 1);
    // if we iterate back to front, we can do this in linear time rather than having to search the entire rest of the array each time
    for (int i = NUM_ITEMS - 1; i >= 0; i--) {
        float thisRatio = (float)itemChoices[i].value / itemChoices[i].weight;
        // if it's the first item we compute, it's just that value, else it's the max of this value and the last value
        bestPerWeightValues[i] = (i == NUM_ITEMS - 1) ? thisRatio : max(bestPerWeightValues[i + 1], thisRatio);
    }
    bestPerWeightValues[NUM_ITEMS] = 0;
}

__global__ void greedySearch(int *weights, int *values, char *solutions, int *solutionValues, int NUM_ITEMS, int capacity) {
    
    int startItem = blockIdx.x * blockDim.x + threadIdx.x;

    // if we're in range, we copy over the data at our slot. 
    if (startItem >= NUM_ITEMS) return;

    // fetch our weight and value for the first item which we are taking no matter what.
    int weight = weights[startItem];  
    int value = (weight > capacity) ? -1 : values[startItem];    
    // mark this item used 
    solutions[startItem * NUM_ITEMS + startItem] = 1;
    
    // Perform greedy search
    while (weight < capacity) {
        int bestItem = -1;
        int bestItemValue = -1;
        int bestItemWeight = -1;

        // iterate through every item, each time around.
        for (int i = 0; i < NUM_ITEMS; i++) {
            // we need to have a better price than the current best, be available, and not be exceeding capacity
            if ((values[i] > bestItemValue) && (solutions[startItem * NUM_ITEMS + i] == '0') && (weight + weights[i] <= capacity)) {
                bestItem = i;
                bestItemValue = values[i];
                bestItemWeight = weights[i];
            }
        }

        // if we didn't find a suitor, break
        if (bestItem == -1) 
            break;

        // mark the best item as used
        solutions[startItem * NUM_ITEMS + bestItem] = '1';
        
        // update our local variables that track our progress
        weight += bestItemWeight;
        value += bestItemValue;
    }

    // Store the solution value in shared memory and later copy to global memory
    solutionValues[startItem] = value;
}

void launchGreedySearch() {
    
    // Allocate CPU memory for item weights and values
    int *hostItemWeights = (int *)malloc(NUM_ITEMS * sizeof(int));
    int *hostItemValues = (int *)malloc(NUM_ITEMS * sizeof(int));

    // Populate host arrays with item data
    for (int i = 0; i < NUM_ITEMS; i++) {
        hostItemWeights[i] = itemChoices[i].weight;
        hostItemValues[i] = itemChoices[i].value;
    }

    // Allocate device memory for our four arrays
    int *deviceItemWeights, *deviceItemValues, *deviceSolutionValues;
    char *deviceSolutions;

    // Allocate device memory
    cudaMalloc(&deviceItemWeights, NUM_ITEMS * sizeof(int));
    cudaMalloc(&deviceItemValues, NUM_ITEMS * sizeof(int));
    cudaMalloc(&deviceSolutions, NUM_ITEMS * NUM_ITEMS * sizeof(char));
    cudaMalloc(&deviceSolutionValues, NUM_ITEMS * sizeof(int));

    // Copy data to device for those bad boys
    cudaMemcpy(deviceItemWeights, hostItemWeights, NUM_ITEMS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceItemValues, hostItemValues, NUM_ITEMS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(deviceSolutions, '0', NUM_ITEMS * NUM_ITEMS * sizeof(char));                                   // it makes more sense to use global memory here for the solutions than local thread memory, since if our N gets big that is an issue. 
    cudaMemset(deviceSolutionValues, 0, NUM_ITEMS * sizeof(int));                                           

    int numThreadsPerBlock = 1024;
    int numBlocks = (NUM_ITEMS + numThreadsPerBlock - 1) / numThreadsPerBlock;

    // Launch the kernel
    greedySearch<<<numBlocks, numThreadsPerBlock>>>(deviceItemWeights, deviceItemValues, deviceSolutions, deviceSolutionValues, NUM_ITEMS, capacity);
    cudaDeviceSynchronize();

    // Copy the solution values back to the host
    int *hostSolutionValues = (int *)malloc(NUM_ITEMS * sizeof(int));
    char *hostSolutions = (char *)malloc(NUM_ITEMS * NUM_ITEMS * sizeof(char));
    cudaMemcpy(hostSolutionValues, deviceSolutionValues, NUM_ITEMS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostSolutions, deviceSolutions, NUM_ITEMS * NUM_ITEMS * sizeof(char), cudaMemcpyDeviceToHost);

    // update our global best when we find improvements
    for (int i = 0; i < NUM_ITEMS; i++) {
        if (hostSolutionValues[i] > globalBest){
            globalBest = hostSolutionValues[i];
            memcpy(globalBestChoices, &hostSolutions[i * NUM_ITEMS], NUM_ITEMS * sizeof(char));
        }
    }

    // Clean up memory
    free(hostItemWeights);
    free(hostItemValues);
    free(hostSolutionValues);
    free(hostSolutions);
    cudaFree(deviceItemWeights);
    cudaFree(deviceItemValues);
    cudaFree(deviceSolutions);
    cudaFree(deviceSolutionValues);
}

__global__ void computeNextItems(int *returningValues, int *returningWeights, int *returningUpperBounds, int *tableOfValues, int *tableOfWeights, float nextRatio, int numToCompute, int maxBranches, int capacity, int currentWeight, int currentValue) {
   
    extern __shared__ int memory[];

    int iD = blockIdx.x * blockDim.x + threadIdx.x;

    // get these pointers so that we can access our values in shared memory easily. 
    int *itemValues = &memory[0];
    int *itemWeights = &memory[numToCompute];

    // if you're in the range of the amount of items we need to compute, you're getting put to work copying stuff to shared memory
    if (threadIdx.x < numToCompute) {
        itemValues[threadIdx.x] = tableOfValues[threadIdx.x];
        itemWeights[threadIdx.x] = tableOfWeights[threadIdx.x];
    }

    if (iD >= maxBranches) {
        return;
    }

    // if we are over the number of branches we could possible solve, but under the default max, we just put -1's in because we're invalid, but our slots of the array are still allocated.
    if (iD >= 1 << numToCompute) {
        returningValues[iD] = -1;
        returningWeights[iD] = -1;
        returningUpperBounds[iD] = -1;
        return;
    }

    // sync all the threads
    __syncthreads();

    // now we compute the weight and value of choosing the combination which corresponds to the threadIDs
    int weight = currentWeight;
    int value = currentValue;

    // take those items which correspond to our threadID.
    for (int i = 0; i < numToCompute; i++) {
        if (iD & (1 << i)) {
            weight += itemWeights[i];
            value += itemValues[i];
        }
    }

    // put our values back where they belong in their array, if weight is over, we put everything as -1
    if (weight <= capacity) {
        returningValues[iD] = value;
        returningWeights[iD] = weight;
        returningUpperBounds[iD] = value + (int)(nextRatio * (capacity - weight)); // our upper bound is the room we have left in the bag times the best possible value per pound remaining.
    }
    else {
        returningValues[iD] = -1;
        returningWeights[iD] = -1;
        returningUpperBounds[iD] = -1;
    }
}

void launchBranchAndBound(){

    // TODO: find a way to track our solution char *'s as well. 
    deque<Solution *> solutionQueue;
    deque<Solution *> usedSolutions;
    time_t startTime = time(NULL);

    // make and initialize all our streams
    cudaStream_t* streams = new cudaStream_t[MAX_KERNEL_COUNT]; // Dynamically allocate array
    for (int i = 0; i < MAX_KERNEL_COUNT; i++) {
        cudaStreamCreate(&streams[i]); // Initialize each stream
    }

    // initialize our base solution
    char *startingSolution = (char *)malloc(NUM_ITEMS * sizeof(char));
    memset(startingSolution, '0', NUM_ITEMS * sizeof(char));

    // create our initial solution
    Solution *s = (Solution *)malloc(sizeof(Solution));
    s->currentValue = 0;
    s->currentWeight = 0;
    s->upperBound = (int)(bestPerWeightValues[0] * capacity);
    s->currentItem = 0;
    s->itemChoices = (char *)malloc(NUM_ITEMS * sizeof(char));
    memcpy(s->itemChoices, startingSolution, NUM_ITEMS * sizeof(char));
    free(startingSolution);

    // add it to the queue
    solutionQueue.push_back(s);

    // this is how many slots are in every return array how many solutions to do at once
    int numBranches = pow(2, NUM_ITEMS_AT_ONCE);

    // now we initialize all our memory. 
    // 3 arrays for the returning values on host and device. FOR THE AMOUNT OF KERNELS WE HAVE, so MAX_NUM_KERNELS * 6 arrays, 3 and 3 host/device
    int **d_currentValues, **d_currentWeights, **d_upperBounds, **h_currentValues, **h_currentWeights, **h_upperBounds;

    // these two are the lookup tables for the items
    int **d_itemWeights, **d_itemValues;

    // Allocate memory for the arrays of pointers themselves
    d_currentValues = (int **)malloc(MAX_KERNEL_COUNT * sizeof(int *));
    d_currentWeights = (int **)malloc(MAX_KERNEL_COUNT * sizeof(int *));
    d_upperBounds = (int **)malloc(MAX_KERNEL_COUNT * sizeof(int *));
    h_currentValues = (int **)malloc(MAX_KERNEL_COUNT * sizeof(int *));
    h_currentWeights = (int **)malloc(MAX_KERNEL_COUNT * sizeof(int *));
    h_upperBounds = (int **)malloc(MAX_KERNEL_COUNT * sizeof(int *));
    d_itemWeights = (int **)malloc(MAX_KERNEL_COUNT * sizeof(int *));
    d_itemValues = (int **)malloc(MAX_KERNEL_COUNT * sizeof(int *));

    // allocate all our stuff
    for(int i = 0; i < MAX_KERNEL_COUNT; i++){
        
        // the returning values from the computations
        cudaMalloc(&d_currentValues[i], numBranches * sizeof(int));
        cudaMalloc(&d_currentWeights[i], numBranches * sizeof(int));
        cudaMalloc(&d_upperBounds[i], numBranches * sizeof(int));

        // these are the lookup tables for the items we are currently selecting
        cudaMalloc(&d_itemWeights[i], NUM_ITEMS_AT_ONCE * sizeof(int));
        cudaMalloc(&d_itemValues[i], NUM_ITEMS_AT_ONCE * sizeof(int));

        // houses for the returning values
        h_currentValues[i] = (int *)malloc(numBranches * sizeof(int));
        h_currentWeights[i] = (int *)malloc(numBranches * sizeof(int));
        h_upperBounds[i] = (int *)malloc(numBranches * sizeof(int));
    }

    // now, each time we simply go through, blast off a bunch of kernels, join them back and see the results.
    // we stop once the queue is empty after we've generated all solutions
    // use a long for this because this may get hairy
    long int totalKernelLaunches = 0;
    while(solutionQueue.size() > 0){

        // make up to MAX_KERNEL_COUNT kernels
        int numKernels = min(MAX_KERNEL_COUNT, solutionQueue.size());

        // launch all our kernels
        int launchedKernels = 0;
        while(launchedKernels < numKernels && solutionQueue.size() > 0){
            Solution *s = solutionQueue.front();
            solutionQueue.pop_front();

            if (s->currentItem >= NUM_ITEMS){
                free(s->itemChoices);
                free(s);
                continue;
            }

            int itemsToCompute = min(NUM_ITEMS - s->currentItem, NUM_ITEMS_AT_ONCE);

            int h_itemWeights[itemsToCompute];
            int h_itemValues[itemsToCompute];

            for (int i = 0; i < itemsToCompute; i++){
                h_itemWeights[i] = itemChoices[s->currentItem + i].weight;
                h_itemValues[i] = itemChoices[s->currentItem + i].value;
            }

            // copy the items to the device
            cudaMemcpy(d_itemWeights[launchedKernels], h_itemWeights, itemsToCompute * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_itemValues[launchedKernels], h_itemValues, itemsToCompute * sizeof(int), cudaMemcpyHostToDevice);

            int numThreadsPerBlock = 1024;
            int numBlocks = (numBranches/numThreadsPerBlock) + 1;// this allows us to launch 1024 threads per block, and launch many blocks

            // the shared memory size just needs to be the two lookup tables. this will be faster than using the global arrays, apparently by like 100x according to nvidia website.
            size_t sharedMemSize = (NUM_ITEMS_AT_ONCE * 2 * sizeof(int));

            // now we launch our kernel, using the corresponding stream
            computeNextItems<<<numBlocks, numThreadsPerBlock, sharedMemSize, streams[launchedKernels]>>>(d_currentValues[launchedKernels], d_currentWeights[launchedKernels], d_upperBounds[launchedKernels], d_itemValues[launchedKernels], d_itemWeights[launchedKernels], bestPerWeightValues[s->currentItem + itemsToCompute], itemsToCompute, numBranches, capacity, s->currentWeight, s->currentValue);
            launchedKernels++;
            totalKernelLaunches++;

            // put it in the used solutions list, now we can look back at this solution later in processing.
            usedSolutions.push_back(s);
        }

        // get their results
        for(int i = 0; i < launchedKernels; i++){
            // copy the results back to the host
            cudaStreamSynchronize(streams[i]);
            cudaMemcpyAsync(h_currentValues[i], d_currentValues[i], numBranches * sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
            cudaMemcpyAsync(h_currentWeights[i], d_currentWeights[i], numBranches * sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
            cudaMemcpyAsync(h_upperBounds[i], d_upperBounds[i], numBranches * sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
        }

        // put it all back together into solutions
        for(int i = 0; i < launchedKernels; i++){

            // synchronize our stream, make sure it's done copying over stuff
            cudaStreamSynchronize(streams[i]);

            // get the solution which launched this kernel
            Solution *seedSolution = usedSolutions.front();
            usedSolutions.pop_front();

            // now go through every branch, each result of this kernel
            for(int j = 0; j < numBranches; j++){

                // if we're invalid for any reason, we just continue
                if (h_currentValues[i][j] == -1 || h_upperBounds[i][j] < globalBest || h_currentWeights[i][j] > capacity){
                    continue;
                }

                // now we look at all the solutions that seedSolution made, and append their information to seeds basically
                Solution *s = (Solution *)malloc(sizeof(Solution));
                
                // set up the char array representing our current partial solution. 
                // copy over the parent solution, and then use bitwise operation based on the iD of this branch to get the selection this one represents
                s->itemChoices = (char *)malloc(NUM_ITEMS * sizeof(char));
                memcpy(s->itemChoices, seedSolution->itemChoices, NUM_ITEMS * sizeof(char));
                
                // j represents the iD which this solution came from, so the bitwise representation represents all those decisions of this given branch
                for (int k = 0; k < min(NUM_ITEMS - seedSolution->currentItem, NUM_ITEMS_AT_ONCE); k++) {
                    s->itemChoices[k + seedSolution->currentItem] = ((j & (1 << k)) ? '1' : '0');
                }

                // copy the rest of the stuff into our new solution
                s->currentValue = h_currentValues[i][j];
                s->currentWeight = h_currentWeights[i][j];
                s->upperBound = h_upperBounds[i][j];
                s->currentItem = seedSolution->currentItem + NUM_ITEMS_AT_ONCE;

                // if we have a new best value, update it
                if (s->currentValue > globalBest){
                    globalBest = s->currentValue;
                    memcpy(globalBestChoices, s->itemChoices, NUM_ITEMS * sizeof(char));
                }

                // put it into the solutionQueue
                solutionQueue.push_back(s);
            }

            // free our parent Solution
            free(seedSolution->itemChoices);
            free(seedSolution);
        }
    }

    for(int i = 0; i < MAX_KERNEL_COUNT; i++){
        cudaFree(d_currentValues[i]);
        cudaFree(d_currentWeights[i]);
        cudaFree(d_upperBounds[i]);
        cudaFree(d_itemWeights[i]);
        cudaFree(d_itemValues[i]);
        free(h_currentValues[i]);
        free(h_currentWeights[i]);
        free(h_upperBounds[i]);
    }

    free(d_currentValues);
    free(d_currentWeights);
    free(d_upperBounds);
    free(d_itemWeights);
    free(d_itemValues);
    free(h_currentValues);
    free(h_currentWeights);
    free(h_upperBounds);
    
    // free all the streams NOW
    for (int i = 0; i < MAX_KERNEL_COUNT; i++){
        cudaStreamDestroy(streams[i]);
    }
    delete[] streams;
    cout << "\nFinished branch and bound!\n";
    cout <<"\nBranch and bound stats:\n";
    time_t totalTime = time(NULL) - startTime;
    cout << "\tTotal Kernel Launches: " << totalKernelLaunches << endl;
    cout << "\tTotal Time (Seconds): " << totalTime << endl;
}

int main(int argc, char* argv[]) {
    
    checkInputs(argc, argv);
    readFile(&inputFile); // Passing in our ifstream
    
    // this is useful in the branching and bounding phase to compute upper bounds without iterating through the entire array.
    // its a lookup table for the item we are at, and the best value per weight after it. 
    initializeBestPerUnitWeights();

    globalBest = 0;
    globalBestChoices = (char *)malloc(NUM_ITEMS * sizeof(char));
    memset(globalBestChoices, '0', NUM_ITEMS * sizeof(char));

    // run the greedy search to find a good initial solution
    launchGreedySearch();
    cout << "Finished greedy search and found best value of " << globalBest << endl;

    cout << "\nStarting branch and bound...\n";
    // run the branch and bound algorithm now.
    launchBranchAndBound();

    cout << "\nGLOBAL BEST VALUE: " << globalBest << endl;

    // print out the best solution
    cout << "\nMost Optimal Solution: ";
    for (int i = 0; i < NUM_ITEMS; i++) {
        printf("%c ", globalBestChoices[i]);
    }
    printf("\n");

    free(bestPerWeightValues);
    free(globalBestChoices);
    return 0;
}