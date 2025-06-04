#include "neuralnetwork.h"
#include "game.h"
#include "selftrain.h"
#include <string.h>
#include <time.h>

// Global opponent tracking
static OpponentProfile opponentProfiles[MAXPLAYERS];
static bool profilesInitialized = false;

// Create a NN (memory allocation)
NeuralNetwork* createNetwork(int inputSize, int hiddenSize, int outputSize)
{
    NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));
    if (!nn) perror("Could not allocate memory for neural network");

    nn->inputSize = inputSize;
    nn->hiddenSize = hiddenSize;
    nn->outputSize = outputSize;
    nn->learningRate = LEARNING_RATE;
    nn->activationFunction = ACTIVATION_SIGMOID;

    nn->inputLayer = calloc(inputSize, sizeof(Neuron));
    if (!nn->inputLayer) perror("Could not allocate memory for input layer");
    nn->hiddenLayer = calloc(hiddenSize, sizeof(Neuron));
    if (!nn->hiddenLayer) perror("Could not allocate memory for hidden layer");
    nn->outputLayer = calloc(outputSize, sizeof(Neuron));
    if (!nn->outputLayer) perror("Could not allocate memory for output layer");

    nn->weightsInputHidden = malloc(inputSize * sizeof(double*));
    for (int i = 0; i < inputSize; i++) {
        nn->weightsInputHidden[i] = calloc(hiddenSize, sizeof(double));
        if (!nn->weightsInputHidden[i]) perror("Could not allocate memory for weights");
    }

    nn->weightsHiddenOutput = malloc(hiddenSize * sizeof(double*));
    for (int i = 0; i < hiddenSize; i++) {
        nn->weightsHiddenOutput[i] = calloc(outputSize, sizeof(double));
        if (!nn->weightsHiddenOutput[i]) perror("Could not allocate memory for weights");
    }

    nn->biasHidden = calloc(hiddenSize, sizeof(double));
    if (!nn->biasHidden) perror("Could not allocate memory for hidden bias");
    nn->biasOutput = calloc(outputSize, sizeof(double));
    if (!nn->biasOutput) perror("Could not allocate memory for hidden bias");

    // Initialize weights randomly
    initialiseWeights(nn);

    return nn;
}

// He initialisation for sigmoid function
void initialiseWeights(NeuralNetwork *nn)
{

    // Initialise input to hidden weights
    double limitIH = sqrt(2.0 / nn->inputSize);
    for (int i = 0; i < nn->inputSize; i++)
    {
        for (int j = 0; j< nn->hiddenSize; j++)
        {
            nn->weightsInputHidden[i][j] = ((double)rand() / RAND_MAX * 2 - 1) * limitIH;
        }
    }

    // Initialise hidden to output weights
    double limitHO = sqrt(2.0 / nn->hiddenSize);
    for (int i = 0; i < nn->hiddenSize; i++)
    {
        for (int j = 0; j < nn->outputSize; j++)
        {
            nn->weightsHiddenOutput[i][j] = ((double)rand() / RAND_MAX * 2 - 1) * limitHO;
        }
    }

    // Initialize biases to small random values
    for (int i = 0; i < nn->hiddenSize; i++)
    {
        nn->biasHidden[i] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
    }
    for (int i = 0; i < nn->outputSize; i++)
    {
        nn->biasOutput[i] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
    }

}

// Ability to switch activation function
double activate(double x, int activationType)
{
    switch (activationType)
    {
        case ACTIVATION_RELU:
            return x > 0 ? x : 0;
        case ACTIVATION_SIGMOID:
            return 1.0 / (1.0 + exp(-x));
        case ACTIVATION_TANH:
            return tanh(x);
        default:
            return x;
    }
}

// Derivatives for backpropagation
double activateDerivative(double x, int activationType)
{
    switch (activationType)
    {
        case ACTIVATION_RELU:
            return x > 0 ? 1 : 0;
        case ACTIVATION_SIGMOID:
        {
            double sig = activate(x, ACTIVATION_SIGMOID);
            return sig * (1 - sig);
        }
        case ACTIVATION_TANH:
        {
            double t = tanh(x);
            return 1 - t * t;
        }
        default:
            return 1;
    }
}

// Forward prop
void forwardpropagate(NeuralNetwork *nn, double *input)
{
    // Copy input to input layer
    for (int i = 0; i < nn->inputSize; i++)
    {
        nn->inputLayer[i].value = input[i];
    }

    // Calculate hidden layer
    for (int j = 0; j < nn->hiddenSize; j++)
    {
        double sum = nn->biasHidden[j];
        for (int i = 0; i < nn->inputSize; i++)
        {
            sum += nn->inputLayer[i].value * nn->weightsInputHidden[i][j];
        }
        nn->hiddenLayer[j].value = activate(sum, nn->activationFunction);
    }

    // Calculate output layer
    for (int k = 0; k < nn->outputSize; k++)
    {
        double sum = nn->biasOutput[k];
        for (int j = 0; j < nn->hiddenSize; j++)
        {
            sum += nn->hiddenLayer[j].value * nn->weightsHiddenOutput[j][k];
        }
        nn->outputLayer[k].value = activate(sum, ACTIVATION_SIGMOID);
    }
}

// Backpropagation
void backpropagate(NeuralNetwork *nn, double *targetOutput)
{
    // Calculate output layer gradients
    for (int k = 0; k < nn->outputSize; k++)
    {
        double output = nn->outputLayer[k].value;
        double target = targetOutput[k];
        nn->outputLayer[k].gradient = (target - output);
    }

    // Calculate hidden layer gradients
    for (int j = 0; j < nn->hiddenSize; j++)
    {
        double sum = 0;
        for (int k = 0; k < nn->outputSize; k++)
        {
            sum += nn->outputLayer[k].gradient * nn->weightsHiddenOutput[j][k];
        }
        nn->hiddenLayer[j].gradient = sum * activateDerivative(nn->hiddenLayer[j].value, nn->activationFunction);
    }
}

// Update weights based on gradients
void updateWeights(NeuralNetwork *nn)
{
    for (int j = 0; j < nn->hiddenSize; j++)
    {
        for (int k = 0; k < nn->outputSize; k++)
        {
            nn->weightsHiddenOutput[j][k] += nn->learningRate * nn->hiddenLayer[j].value * nn->outputLayer[k].gradient;
        }
    }

    // Update output biases
    for (int k = 0; k < nn->outputSize; k++)
    {
        nn->biasOutput[k] += nn->learningRate * nn->outputLayer[k].gradient;
    }

    // Update input to hidden weights
    for (int i = 0; i < nn->inputSize; i++)
    {
        for (int j = 0; j < nn->hiddenSize; j++)
        {
            nn->weightsInputHidden[i][j] += nn->learningRate * nn->inputLayer[i].value * nn->hiddenLayer[j].gradient;
        }
    }

    // Update hidden biases
    for (int j = 0; j < nn->hiddenSize; j++)
    {
        nn->biasHidden[j] += nn->learningRate * nn->hiddenLayer[j].gradient;
    }
}

// Enhanced game state encoding function
void encodeGameState(Player *player, Hand *communityCards, int pot, int currentBet, 
                           int numPlayers, int position, double *output) {
    memset(output, 0, INPUT_SIZE * sizeof(double));
    
    // Prepare card arrays
    Card playerCards[2];
    Card communityArray[5];
    Card combined[7];
    int numCards = 0;
    int numCommunity = 0;
    
    // Extract player cards
    Card *current = player->hand->first;
    int cardCount = 0;
    while (current && cardCount < 2) {
        playerCards[cardCount] = *current;
        combined[numCards++] = *current;
        current = current->next;
        cardCount++;
    }
    
    // Extract community cards
    current = communityCards->first;
    while (current && numCommunity < 5) {
        communityArray[numCommunity] = *current;
        combined[numCards++] = *current;
        current = current->next;
        numCommunity++;
    }
    
    // Calculate hand score
    HandScore score = findBestHand(combined, numCards);
    
    // FEATURE 0: Hand strength
    output[0] = (double)score.rank / 10.0;
    
    // FEATURE 1: Hand potential
    output[1] = calculateHandPotential(playerCards, communityArray, numCommunity);
    
    // FEATURE 2: Board texture
    output[2] = calculateBoardTexture(communityArray, numCommunity);
    
    // FEATURE 3: Pot odds
    double potOdds = 0.0;
    if (currentBet > player->currentBet && pot > 0) {
        potOdds = (double)(currentBet - player->currentBet) / (pot + currentBet);
    }
    output[3] = fmin(potOdds, 1.0);
    
    // FEATURE 4: Stack to pot ratio
    output[4] = (double)player->credits / (pot + 1);
    
    // FEATURE 5: Position (normalized)
    output[5] = (double)position / (numPlayers - 1);
    
    // FEATURE 6: Number of players (normalized)
    output[6] = (double)numPlayers / MAXPLAYERS;
    
    // FEATURE 7: Current bet relative to big blind
    output[7] = (double)currentBet / BIG_BLIND;
    
    // FEATURE 8: Player's current bet ratio
    output[8] = (double)player->currentBet / (pot + 1);
    
    // FEATURES 9-10: Player's hole cards (normalized)
    if (cardCount >= 2) {
        output[9] = (double)playerCards[0].value / 13.0;
        output[10] = (double)playerCards[1].value / 13.0;
    }
    
    // FEATURE 11: Cards suited (0 or 1)
    if (cardCount >= 2) {
        output[11] = (playerCards[0].suit == playerCards[1].suit) ? 1.0 : 0.0;
    }
    
    // FEATURE 12: Community cards revealed phase
    output[12] = (double)numCommunity / 5.0;
    
    // FEATURE 13: Stack committed percentage
    output[13] = (double)player->currentBet / (player->credits + player->currentBet + 1);
    
    // FEATURES 14-16: Opponent modeling (average characteristics)
    if (profilesInitialized && numPlayers > 1) {
        double avgAggression = 0.0;
        double avgTightness = 0.0;
        double avgFoldToRaise = 0.0;
        int activeOpponents = 0;
        
        for (int i = 0; i < numPlayers; i++) {
            if (i != position) {  // Don't include self
                avgAggression += opponentProfiles[i].aggressionLevel;
                avgTightness += opponentProfiles[i].tightness;
                avgFoldToRaise += opponentProfiles[i].foldToRaise;
                activeOpponents++;
            }
        }
        
        if (activeOpponents > 0) {
            output[14] = avgAggression / activeOpponents;
            output[15] = avgTightness / activeOpponents;
            output[16] = avgFoldToRaise / activeOpponents;
        }
    }
    
    // FEATURE 17: Recent opponent aggression
    if (profilesInitialized && pot > 0) {
        double recentAggression = 0.0;
        
        // Find most recent aggressive action this round
        for (int i = 0; i < numPlayers; i++) {
            if (i != position && opponentProfiles[i].aggressiveActionThisRound) {
                double aggressionLevel = opponentProfiles[i].lastAggressiveAmount / (pot + 1);
                recentAggression = fmax(recentAggression, aggressionLevel);
            }
        }
        
        output[17] = fmin(recentAggression, 1.0);
    }
    
    // FEATURE 18: Pair in hand
    if (cardCount >= 2) {
        output[18] = (playerCards[0].value == playerCards[1].value) ? 1.0 : 0.0;
    }
    
    // FEATURE 19: High card strength
    if (cardCount >= 2) {
        int highCard = (playerCards[0].value > playerCards[1].value) ? 
                       playerCards[0].value : playerCards[1].value;
        output[19] = (double)highCard / 13.0;
    }
}

// Decision making function
int makeEnhancedDecision(NeuralNetwork *nn, Player *player, Hand *communityCards, 
                        int pot, int currentBet, int numPlayers, int position) {
    double input[INPUT_SIZE];
    
    // Use game state encoding
    encodeGameState(player, communityCards, pot, currentBet, numPlayers, position, input);
    
    // Forward propagation
    forwardpropagate(nn, input);
    
    // Find best action with improved logic
    int bestAction = 0;
    double bestProb = nn->outputLayer[0].value;
    
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (nn->outputLayer[i].value > bestProb) {
            bestProb = nn->outputLayer[i].value;
            bestAction = i;
        }
    }
    
    // Add some controlled randomness to prevent predictability
    // This replaces pure randomness with informed randomness
    double randomFactor = (double)rand() / RAND_MAX;
    if (randomFactor < 0.1) {  // 10% chance of second-best move
        double secondBest = 0.0;
        int secondAction = bestAction;
        
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            if (i != bestAction && nn->outputLayer[i].value > secondBest) {
                secondBest = nn->outputLayer[i].value;
                secondAction = i;
            }
        }
        
        // Only take second-best if it's reasonable (> 20% confidence)
        if (secondBest > 0.2) {
            bestAction = secondAction;
        }
    }
    
    return bestAction;
}

// ===================================================================
// OPPONENT MODELING FUNCTIONS
// ===================================================================

// Initialize opponent profiles at start of game
void initializeOpponentProfiles(int numPlayers) {
    if (!profilesInitialized) {
        for (int i = 0; i < numPlayers; i++) {
            opponentProfiles[i].aggressionLevel = 0.5;  // Start neutral
            opponentProfiles[i].tightness = 0.5;
            opponentProfiles[i].bluffFrequency = 0.1;   // Conservative estimate
            opponentProfiles[i].foldToRaise = 0.5;
            opponentProfiles[i].totalActions = 0;
            opponentProfiles[i].raiseCount = 0;
            opponentProfiles[i].callCount = 0;
            opponentProfiles[i].foldCount = 0;
            opponentProfiles[i].handsPlayed = 0;
            opponentProfiles[i].voluntaryPuts = 0;
        }
        profilesInitialized = true;
        printf("Opponent profiles initialized for %d players.\n", numPlayers);
    }
}

// Update opponent profile based on their action
void updateOpponentProfile(int playerIndex, int action, bool voluntaryAction, int betAmount, int potSize) {
    if (playerIndex < 0 || playerIndex >= MAXPLAYERS) return;
    
    OpponentProfile *profile = &opponentProfiles[playerIndex];
    profile->totalActions++;
    
    switch(action) {
        case 0: // Fold
            profile->foldCount++;
            profile->aggressiveActionThisRound = false;  // Reset aggression flag
            break;
        case 1: // Call/Check
            profile->callCount++;
            if (voluntaryAction) profile->voluntaryPuts++;
            // Don't reset aggression flag - calls don't change round aggression
            break;
        case 2: // Raise
            profile->raiseCount++;
            profile->voluntaryPuts++;
            
            // NEW: Track recent aggression
            profile->lastAggressiveAmount = betAmount;
            profile->aggressiveActionThisRound = true;
            break;
    }
    
    // Update aggression level (how often they bet/raise vs call/fold)
    if (profile->totalActions > 5) {  // Need some data first
        profile->aggressionLevel = (double)(profile->raiseCount) / profile->totalActions;
        profile->foldToRaise = (double)(profile->foldCount) / profile->totalActions;
        
        // Calculate tightness (how selective they are)
        if (profile->handsPlayed > 0) {
            profile->tightness = 1.0 - ((double)profile->voluntaryPuts / profile->handsPlayed);
        }
    }
}

void resetRoundAggression() {
    if (!profilesInitialized) return;
    
    for (int i = 0; i < MAXPLAYERS; i++) {
        opponentProfiles[i].aggressiveActionThisRound = false;
        opponentProfiles[i].lastAggressiveAmount = 0.0;
    }
}

// Calculate hand potential - how likely the hand is to improve
double calculateHandPotential(Card playerCards[], Card communityCards[], int numCommunity) {
    if (numCommunity < 3) return 0.5;  // Pre-flop, assume average potential
    
    double potential = 0.0;
    int improvements = 0;
    
    // Check for drawing possibilities
    int suits[4] = {0};
    int values[14] = {0};
    
    // Count player cards
    for (int i = 0; i < 2; i++) {
        suits[playerCards[i].suit]++;
        values[playerCards[i].value]++;
    }
    
    // Count community cards
    for (int i = 0; i < numCommunity; i++) {
        suits[communityCards[i].suit]++;
        values[communityCards[i].value]++;
    }
    
    // Check for flush draws
    for (int i = 0; i < 4; i++) {
        if (suits[i] == 4 && numCommunity < 5) {  // 4 to a flush with cards to come
            potential += 0.35;  // ~35% chance to complete flush
            improvements++;
        }
    }
    
    // Check for straight draws (simplified)
    int consecutive = 0;
    int maxConsecutive = 0;
    for (int i = 1; i <= 13; i++) {
        if (values[i] > 0) {
            consecutive++;
            if (consecutive > maxConsecutive) maxConsecutive = consecutive;
        } else {
            consecutive = 0;
        }
    }
    
    if (maxConsecutive >= 4 && numCommunity < 5) {  // Open-ended straight draw
        potential += 0.32;  // ~32% chance
        improvements++;
    } else if (maxConsecutive == 3 && numCommunity < 5) {  // Gutshot
        potential += 0.17;  // ~17% chance
        improvements++;
    }
    
    // Normalize potential
    if (improvements > 0) {
        potential = fmin(potential, 1.0);
    }
    
    return potential;
}

// Calculate board texture (how coordinated/dangerous the board is)
double calculateBoardTexture(Card communityCards[], int numCommunity) {
    if (numCommunity < 3) return 0.5;  // Pre-flop
    
    double texture = 0.0;  // 0 = dry board, 1 = wet board
    
    int suits[4] = {0};
    int values[14] = {0};
    int pairs = 0;
    
    // Analyze community cards
    for (int i = 0; i < numCommunity; i++) {
        suits[communityCards[i].suit]++;
        values[communityCards[i].value]++;
    }
    
    // Check for flush possibilities
    for (int i = 0; i < 4; i++) {
        if (suits[i] >= 3) texture += 0.3;  // Flush possible
        if (suits[i] >= 2) texture += 0.1;  // Flush draw possible
    }
    
    // Check for pairs/trips
    for (int i = 1; i <= 13; i++) {
        if (values[i] >= 2) {
            pairs++;
            texture += 0.2;  // Paired board
        }
    }
    
    // Check for straight possibilities (simplified)
    int consecutive = 0;
    for (int i = 1; i <= 13; i++) {
        if (values[i] > 0) {
            consecutive++;
        } else {
            if (consecutive >= 3) texture += 0.2;  // Straight possible
            consecutive = 0;
        }
    }
    
    return fmin(texture, 1.0);
}

// ===================================================================
// NETWORK FILE I/O AND UTILITY FUNCTIONS
// ===================================================================

void saveNetwork(NeuralNetwork *nn, const char *filename)
{
    if (!nn) {
        printf("Cannot save NULL network\n");
        return;
    }

    FILE *file = fopen(filename, "wb");
    if(!file) {
        perror("Could not save network");
        return;
    }

    // Save
    fwrite(&nn->inputSize, sizeof(int), 1, file);
    fwrite(&nn->hiddenSize, sizeof(int), 1, file);
    fwrite(&nn->outputSize, sizeof(int), 1, file);

    // Save weights
    for (int i = 0; i < nn->inputSize; i++) {
        fwrite(nn->weightsInputHidden[i], sizeof(double), nn->hiddenSize, file);
    }

    for (int i = 0; i < nn->hiddenSize; i++) {
        fwrite(nn->weightsHiddenOutput[i], sizeof(double), nn->outputSize, file);
    }

    // Save biases
    fwrite(nn->biasHidden, sizeof(double), nn->hiddenSize, file);
    fwrite(nn->biasOutput, sizeof(double), nn->outputSize, file);

    fclose(file);
}

NeuralNetwork* loadNetwork(const char *filename)
{
    FILE *file = fopen(filename, "rb");
    if (!file)
    {
        perror("Could not load network");
        return NULL;
    }

    int inputSize, hiddenSize, outputSize;
    fread(&inputSize, sizeof(int), 1, file);
    fread(&hiddenSize, sizeof(int), 1, file);
    fread(&outputSize, sizeof(int), 1, file);

    NeuralNetwork *nn = createNetwork(inputSize, hiddenSize, outputSize);

    // Load weights
    for (int i = 0; i < nn->inputSize; i++) {
        fread(nn->weightsInputHidden[i], sizeof(double), nn->hiddenSize, file);
    }

    for (int i = 0; i < nn->hiddenSize; i++) {
        fread(nn->weightsHiddenOutput[i], sizeof(double), nn->outputSize, file);
    }

    // Load biases
    fread(nn->biasHidden, sizeof(double), nn->hiddenSize, file);
    fread(nn->biasOutput, sizeof(double), nn->outputSize, file);

    fclose(file);
    return nn;
}

// Free network memory
void freeNetwork(NeuralNetwork *nn) {
    if (!nn) perror("No network to free");

    free(nn->inputLayer);
    free(nn->hiddenLayer);
    free(nn->outputLayer);

    for (int i = 0; i < nn->inputSize; i++) {
        free(nn->weightsInputHidden[i]);
    }
    free(nn->weightsInputHidden);

    for (int i = 0; i < nn->hiddenSize; i++) {
        free(nn->weightsHiddenOutput[i]);
    }
    free(nn->weightsHiddenOutput);

    free(nn->biasHidden);
    free(nn->biasOutput);
    free(nn);
}

// For debugging
void printNetworkState(NeuralNetwork *nn) {
    printf("\nNeural Network State:\n");
    printf("Input Layer: ");
    for (int i = 0; i < nn->inputSize && i < 5; i++) {
        printf("%.2f ", nn->inputLayer[i].value);
    }
    printf("...\n");

    printf("Hidden Layer: ");
    for (int i = 0; i < nn->hiddenSize && i < 5; i++) {
        printf("%.2f ", nn->hiddenLayer[i].value);
    }
    printf("...\n");

    printf("Output Layer: ");
    for (int i = 0; i < nn->outputSize; i++) {
        printf("%.2f ", nn->outputLayer[i].value);
    }
    printf("\n");
    printf("Decision: %s\n", nn->outputLayer[0].value > nn->outputLayer[1].value ?
           (nn->outputLayer[0].value > nn->outputLayer[2].value ? "FOLD" : "RAISE") :
           (nn->outputLayer[1].value > nn->outputLayer[2].value ? "CALL" : "RAISE"));
}

// ===================================================================
// TRAINING MONITORING FUNCTIONS
// ===================================================================

// Add this helper function at the top of the file, after the includes
void printRepeatedChar(char c, int count) {
    for (int i = 0; i < count; i++) {
        printf("%c", c);
    }
}

// Initialize training statistics
TrainingStats* initializeTrainingStats(int maxEpochs) {
    TrainingStats *stats = malloc(sizeof(TrainingStats));
    if (!stats) {
        printf("Error: Could not allocate memory for training statistics\n");
        return NULL;
    }
    
    stats->lossHistory = malloc(maxEpochs * sizeof(double));
    stats->epochNumbers = malloc(maxEpochs * sizeof(int));
    
    if (!stats->lossHistory || !stats->epochNumbers) {
        printf("Error: Could not allocate memory for training history\n");
        free(stats);
        return NULL;
    }
    
    stats->currentEpoch = 0;
    stats->maxEpochs = maxEpochs;
    stats->bestLoss = 1000000.0;  // Initialize to very high value
    stats->bestEpoch = 0;
    stats->initialLoss = 0.0;
    stats->startTime = clock();
    
    // NEW: Overfitting detection
    stats->recentLossAverage = 0.0;
    stats->overfittingDetected = false;
    stats->stagnationCount = 0;
    
    // Open log file for training progress
    stats->logFile = fopen("training_log.csv", "w");
    if (stats->logFile) {
        fprintf(stats->logFile, "Epoch,Loss,Time_Elapsed,Learning_Rate\n");
    }
    
    printf("Training monitoring initialized for %d epochs\n", maxEpochs);
    printf("Progress will be saved to 'training_log.csv'\n\n");
    
    return stats;
}

// Calculate mean squared error loss
double calculateLoss(NeuralNetwork *nn, double **inputs, double **targets, int numSamples) {
    double totalLoss = 0.0;
    
    for (int sample = 0; sample < numSamples; sample++) {
        // Forward pass
        forwardpropagate(nn, inputs[sample]);
        
        // Calculate squared error for each output
        for (int i = 0; i < nn->outputSize; i++) {
            double error = targets[sample][i] - nn->outputLayer[i].value;
            totalLoss += error * error;
        }
    }
    
    return totalLoss / (numSamples * nn->outputSize);  // Mean squared error
}

// Calculate prediction accuracy
double calculateAccuracy(NeuralNetwork *nn, double **inputs, double **targets, int numSamples) {
    int correctPredictions = 0;
    
    for (int sample = 0; sample < numSamples; sample++) {
        forwardpropagate(nn, inputs[sample]);
        
        // Find predicted action (highest probability)
        int predictedAction = 0;
        double maxProb = nn->outputLayer[0].value;
        for (int i = 1; i < nn->outputSize; i++) {
            if (nn->outputLayer[i].value > maxProb) {
                maxProb = nn->outputLayer[i].value;
                predictedAction = i;
            }
        }
        
        // Find target action (highest probability)
        int targetAction = 0;
        double maxTarget = targets[sample][0];
        for (int i = 1; i < nn->outputSize; i++) {
            if (targets[sample][i] > maxTarget) {
                maxTarget = targets[sample][i];
                targetAction = i;
            }
        }
        
        if (predictedAction == targetAction) {
            correctPredictions++;
        }
    }
    
    return (double)correctPredictions / numSamples;
}

// Update training statistics for current epoch
void updateTrainingStats(TrainingStats *stats, NeuralNetwork *nn, double **inputs, 
                        double **targets, int numSamples, double learningRate) {
    if (!stats || stats->currentEpoch >= stats->maxEpochs) return;
    
    // Calculate current loss
    double currentLoss = calculateLoss(nn, inputs, targets, numSamples);
    
    // Store in history
    stats->lossHistory[stats->currentEpoch] = currentLoss;
    stats->epochNumbers[stats->currentEpoch] = stats->currentEpoch;
    
    // Track best performance
    if (currentLoss < stats->bestLoss) {
        stats->bestLoss = currentLoss;
        stats->bestEpoch = stats->currentEpoch;
        stats->stagnationCount = 0;  // Reset stagnation counter
    } else {
        stats->stagnationCount++;
    }
    
    // Set initial loss on first epoch
    if (stats->currentEpoch == 0) {
        stats->initialLoss = currentLoss;
    }
    
    // NEW: Overfitting detection
    if (stats->currentEpoch >= 50) {
        // Calculate recent average loss (last 50 epochs)
        double recentSum = 0.0;
        for (int i = stats->currentEpoch - 49; i <= stats->currentEpoch; i++) {
            recentSum += stats->lossHistory[i];
        }
        stats->recentLossAverage = recentSum / 50.0;
        
        // Detect overfitting: recent average higher than best loss by significant margin
        if (stats->recentLossAverage > stats->bestLoss * 1.5 && stats->stagnationCount > 100) {
            stats->overfittingDetected = true;
        }
    }
    
    // Calculate elapsed time
    clock_t currentTime = clock();
    double elapsedSeconds = ((double)(currentTime - stats->startTime)) / CLOCKS_PER_SEC;
    
    // Log to file
    if (stats->logFile) {
        fprintf(stats->logFile, "%d,%.6f,%.2f,%.6f\n", 
                stats->currentEpoch, currentLoss, elapsedSeconds, learningRate);
        fflush(stats->logFile);  // Ensure data is written immediately
    }
    
    stats->currentEpoch++;
}

// Calculate network confidence/decision distribution
void analyzeNetworkConfidence(NeuralNetwork *nn, double **inputs, int numSamples) {
    double totalConfidence = 0.0;
    int decisions[3] = {0}; // Count of fold, call, raise decisions
    double avgOutputs[3] = {0.0, 0.0, 0.0};
    
    printf("\nNETWORK CONFIDENCE ANALYSIS:\n");
    printRepeatedChar('-', 40);
    printf("\n");
    
    for (int i = 0; i < numSamples; i++) {
        forwardpropagate(nn, inputs[i]);
        
        // Find highest probability decision
        int decision = 0;
        double maxProb = nn->outputLayer[0].value;
        for (int j = 1; j < 3; j++) {
            if (nn->outputLayer[j].value > maxProb) {
                maxProb = nn->outputLayer[j].value;
                decision = j;
            }
        }
        
        decisions[decision]++;
        totalConfidence += maxProb;
        
        // Accumulate average outputs
        for (int j = 0; j < 3; j++) {
            avgOutputs[j] += nn->outputLayer[j].value;
        }
    }
    
    // Calculate averages
    totalConfidence /= numSamples;
    for (int i = 0; i < 3; i++) {
        avgOutputs[i] /= numSamples;
    }
    
    printf("Average Confidence: %.3f (%.1f%%)\n", totalConfidence, totalConfidence * 100);
    printf("Decision Distribution:\n");
    printf("  Fold:  %d samples (%.1f%%) | Avg Prob: %.3f\n", 
           decisions[0], (decisions[0] * 100.0) / numSamples, avgOutputs[0]);
    printf("  Call:  %d samples (%.1f%%) | Avg Prob: %.3f\n", 
           decisions[1], (decisions[1] * 100.0) / numSamples, avgOutputs[1]);
    printf("  Raise: %d samples (%.1f%%) | Avg Prob: %.3f\n", 
           decisions[2], (decisions[2] * 100.0) / numSamples, avgOutputs[2]);
    
    // Analyze decision balance
    double maxDecisionPct = fmax(fmax(decisions[0], decisions[1]), decisions[2]) * 100.0 / numSamples;
    double minDecisionPct = fmin(fmin(decisions[0], decisions[1]), decisions[2]) * 100.0 / numSamples;
    
    printf("\nStrategy Balance:\n");
    if (maxDecisionPct > 70) {
        printf("⚠ UNBALANCED: Too focused on one action (%.1f%% vs %.1f%%)\n", maxDecisionPct, minDecisionPct);
    } else if (maxDecisionPct < 45) {
        printf("✓ BALANCED: Good action diversity\n");
    } else {
        printf("→ MODERATE: Reasonable action balance\n");
    }
}

// ===================================================================
// MINIMAL BOOTSTRAP TRAINING (PHASE 1)
// ===================================================================

// Generate minimal bootstrap data - just basic rules to prevent crashes
void generateMinimalBootstrap(double **inputs, double **outputs, int numSamples) {
    printf("Generating minimal bootstrap data (%d samples)...\n", numSamples);
    
    for (int i = 0; i < numSamples; i++) {
        // Create basic game scenarios
        double handStrength = (double)rand() / RAND_MAX;
        double stackRatio = (double)rand() / RAND_MAX * 5.0;
        double currentBet = (double)rand() / RAND_MAX;
        double playerBet = (double)rand() / RAND_MAX;
        
        // Fill inputs with reasonable values (20 features total)
        inputs[i][0] = handStrength;  // Hand strength
        inputs[i][1] = 0.5;          // Hand potential
        inputs[i][2] = 0.5;          // Board texture
        inputs[i][3] = 0.5;          // Pot odds
        inputs[i][4] = stackRatio;   // Stack to pot ratio
        inputs[i][5] = (double)rand() / RAND_MAX;  // Position
        inputs[i][6] = 0.5;          // Number of players (normalized)
        inputs[i][7] = currentBet;   // Current bet level
        inputs[i][8] = playerBet;    // Player's bet ratio
        inputs[i][9] = (double)rand() / RAND_MAX;   // First hole card
        inputs[i][10] = (double)rand() / RAND_MAX;  // Second hole card
        inputs[i][11] = (double)rand() / RAND_MAX;  // Cards suited
        inputs[i][12] = (double)rand() / RAND_MAX;  // Game phase
        inputs[i][13] = 0.5;         // Stack committed
        inputs[i][14] = 0.5;         // Avg opponent aggression
        inputs[i][15] = 0.5;         // Avg opponent tightness
        inputs[i][16] = 0.5;         // Avg opponent fold rate
        inputs[i][17] = 0.3;         // Recent opponent aggression
        inputs[i][18] = (double)rand() / RAND_MAX;  // Pair in hand
        inputs[i][19] = (double)rand() / RAND_MAX;  // High card strength
        
        // MINIMAL STRATEGY - Only basic rules to prevent crashes/illegal moves
        double fold = 0.33, call = 0.33, raise = 0.33;  // Start neutral
        
        // Rule 1: Don't go all-in with terrible hands (prevents instant losses)
        if (handStrength < 0.1 && currentBet > 0.8) {
            fold = 0.9; call = 0.08; raise = 0.02;
        }
        
        // Rule 2: Don't fold when you can check for free
        if (currentBet == 0.0) {  // Can check for free
            fold = 0.05; call = 0.7; raise = 0.25;  // Heavily favor checking/betting
        }
        
        // Rule 3: With very strong hands, don't fold (basic hand ranking)
        if (handStrength > 0.8) {
            fold = 0.05; call = 0.4; raise = 0.55;
        }
        
        // Rule 4: Don't bet more than you have (stack ratio check)
        if (stackRatio < 0.2 && currentBet > 0.5) {  // Low chips, high bet
            fold = 0.6; call = 0.35; raise = 0.05;  // Be more conservative
        }
        
        // Normalize to valid probabilities
        double total = fold + call + raise;
        outputs[i][0] = fold / total;
        outputs[i][1] = call / total;
        outputs[i][2] = raise / total;
    }
    
    printf("Minimal bootstrap data generated - basic rules only.\n");
}

// Phase 1: Train minimal bootstrap (just enough to make legal moves)
NeuralNetwork* trainMinimalBootstrap() {
    printf("\n");
    printRepeatedChar('=', 60);
    printf("\n");
    printf("PHASE 1: MINIMAL BOOTSTRAP TRAINING\n");
    printRepeatedChar('=', 60);
    printf("\n");
    printf("Teaching basic rules to prevent crashes and illegal moves...\n");
    printf("This is NOT strategy - just survival basics!\n\n");
    
    // Small dataset - just enough to learn basic rules
    int numSamples = 100;
    double **inputs = malloc(numSamples * sizeof(double*));
    double **outputs = malloc(numSamples * sizeof(double*));
    
    for (int i = 0; i < numSamples; i++) {
        inputs[i] = malloc(INPUT_SIZE * sizeof(double));
        outputs[i] = malloc(OUTPUT_SIZE * sizeof(double));
        if (!inputs[i] || !outputs[i]) {
            printf("Error: Memory allocation failed\n");
            return NULL;
        }
    }
    
    generateMinimalBootstrap(inputs, outputs, numSamples);
    
    printf("Creating fresh neural network...\n");
    NeuralNetwork *nn = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    if (!nn) {
        printf("Error: Failed to create neural network\n");
        return NULL;
    }
    
    nn->learningRate = 0.05;  // Higher learning rate for quick bootstrap
    
    // Quick training - just enough to learn basic rules
    int bootstrapEpochs = 200;  // Much fewer epochs than full training
    printf("Quick bootstrap training (%d epochs)...\n", bootstrapEpochs);
    
    for (int epoch = 0; epoch < bootstrapEpochs; epoch++) {
        for (int sample = 0; sample < numSamples; sample++) {
            forwardpropagate(nn, inputs[sample]);
            backpropagate(nn, outputs[sample]);
            updateWeights(nn);
        }
        
        // Simple progress indicator
        if (epoch % 50 == 0 || epoch == bootstrapEpochs - 1) {
            double loss = calculateLoss(nn, inputs, outputs, numSamples);
            printf("Epoch %d/%d | Loss: %.4f\n", epoch, bootstrapEpochs - 1, loss);
        }
    }
    
    printf("\nTesting bootstrap network...\n");
    // Test a few examples
    for (int i = 0; i < 3; i++) {
        forwardpropagate(nn, inputs[i]);
        printf("Test %d: Expected[%.2f,%.2f,%.2f] Got[%.2f,%.2f,%.2f]\n", 
               i+1, outputs[i][0], outputs[i][1], outputs[i][2],
               nn->outputLayer[0].value, nn->outputLayer[1].value, nn->outputLayer[2].value);
    }
    
    printf("\nSaving bootstrap network...\n");
    saveNetwork(nn, "poker_ai_bootstrap.dat");
    
    printf("\n");
    printRepeatedChar('=', 60);
    printf("\n");
    printf("BOOTSTRAP COMPLETE!\n");
    printf("Network knows basic rules - ready for self-play learning.\n");
    printf("Bootstrap saved as: poker_ai_bootstrap.dat\n");
    printRepeatedChar('=', 60);
    printf("\n");
    
    // Cleanup training data
    for (int i = 0; i < numSamples; i++) {
        free(inputs[i]);
        free(outputs[i]);
    }
    free(inputs);
    free(outputs);
    
    return nn;  // Return the bootstrap network
}

// Phase 2: Pure self-play learning starting from bootstrap
void pureReinforcementLearning(int numGames, int numPlayers) {
    printf("\n");
    printRepeatedChar('=', 70);
    printf("\n");
    printf("PHASE 2: PURE REINFORCEMENT LEARNING WITH LOSS TRACKING\n");
    printRepeatedChar('=', 70);
    printf("\n");
    printf("Starting self-play with comprehensive loss monitoring...\n");
    printf("Bootstrap knowledge may be completely overwritten.\n");
    printRepeatedChar('=', 70);
    printf("\n\n");
    
    // Initialize opponent profiles
    initializeOpponentProfiles(numPlayers);
    
    // Create networks starting from bootstrap
    NeuralNetwork **networks = malloc(numPlayers * sizeof(NeuralNetwork*));
    
    // Try to load bootstrap network first
    NeuralNetwork *bootstrap = loadNetwork("poker_ai_bootstrap.dat");
    if (!bootstrap) {
        printf("No bootstrap found - creating fresh networks...\n");
        bootstrap = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    }
    
    // Create diverse AIs starting from bootstrap
    for (int i = 0; i < numPlayers; i++) {
        networks[i] = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
        
        // Copy bootstrap weights to each network
        if (bootstrap) {
            // Copy all weights from bootstrap
            for (int inp = 0; inp < INPUT_SIZE; inp++) {
                for (int hid = 0; hid < HIDDEN_SIZE; hid++) {
                    networks[i]->weightsInputHidden[inp][hid] = bootstrap->weightsInputHidden[inp][hid];
                }
            }
            for (int hid = 0; hid < HIDDEN_SIZE; hid++) {
                for (int out = 0; out < OUTPUT_SIZE; out++) {
                    networks[i]->weightsHiddenOutput[hid][out] = bootstrap->weightsHiddenOutput[hid][out];
                }
                networks[i]->biasHidden[hid] = bootstrap->biasHidden[hid];
            }
            for (int out = 0; out < OUTPUT_SIZE; out++) {
                networks[i]->biasOutput[out] = bootstrap->biasOutput[out];
            }
        }
        
        // Add noise for diversity (each AI will evolve differently)
        addNoiseToWeights(networks[i], 0.15);
        printf("AI_%d initialized from bootstrap + noise\n", i);
    }
    
    ReplayBuffer *rb = createReplayBuffer(100000);
    
    // Initialize self-play loss tracking *** NEW ***
    int progressCheckpoints = 20;
    SelfPlayStats *lossStats = initializeSelfPlayStats(progressCheckpoints);
    if (!lossStats) {
        printf("Warning: Could not initialize loss tracking, continuing without it...\n");
    }
    
    // Training tracking
    int wins[MAXPLAYERS] = {0};
    double avgCredits[MAXPLAYERS] = {0};
    int gamesPerCheckpoint = numGames / progressCheckpoints;
    
    FILE *progressFile = fopen("selfplay_progress.csv", "w");
    if (progressFile) {
        fprintf(progressFile, "Game,AI_0_WinRate,AI_1_WinRate,AI_2_WinRate,AI_3_WinRate,AvgCredits,ExperienceBufferSize\n");
    }
    
    clock_t startTime = clock();
    
    printf("Starting pure self-play learning with loss monitoring...\n");
    printf("Each AI will develop its own strategy through experience!\n\n");
    
    // Main self-play training loop
    for (int game = 0; game < numGames; game++) {
        GameRecord record = selfPlayGame(networks, numPlayers, rb);
        
        // Update statistics
        wins[record.winner]++;
        for (int i = 0; i < numPlayers; i++) {
            avgCredits[i] = (avgCredits[i] * game + record.finalCredits[i]) / (game + 1);
        }
        
        // Train from experience (this overwrites bootstrap knowledge!)
        if (rb->size > 500 && game % 3 == 0) {
            for (int i = 0; i < numPlayers; i++) {
                trainFromExperience(networks[i], rb, 128);
            }
        }
        
        // Enhanced progress monitoring with loss tracking *** MODIFIED ***
        if (game > 0 && (game % gamesPerCheckpoint == 0 || game == numGames - 1)) {
            clock_t currentTime = clock();
            double elapsed = ((double)(currentTime - startTime)) / CLOCKS_PER_SEC;
            
            // Update self-play loss statistics *** NEW ***
            if (lossStats) {
                updateSelfPlayStats(lossStats, networks, numPlayers, rb, wins, game, avgCredits);
                displaySelfPlayProgress(lossStats, game, numGames, wins, numPlayers, avgCredits, rb->size);
            } else {
                // Fallback to original progress display
                printf("\n");
                printRepeatedChar('-', 50);
                printf("\n");
                printf("SELF-PLAY PROGRESS: Game %d/%d (%.1f%%)\n", game, numGames, (game * 100.0) / numGames);
                printRepeatedChar('-', 50);
                printf("\n");
                printf("Time elapsed: %.1f seconds (%.2f games/sec)\n", elapsed, game / elapsed);
                printf("Experience buffer: %d samples\n", rb->size);
                
                printf("\nCurrent win rates (evolved from bootstrap):\n");
                for (int i = 0; i < numPlayers; i++) {
                    double winRate = (game > 0) ? (wins[i] * 100.0) / game : 0.0;
                    printf("  AI_%d: %.1f%% (%d wins) | Avg Credits: %.0f\n", 
                           i, winRate, wins[i], avgCredits[i]);
                }
            }
            
            // Log to file (keep existing logging)
            if (progressFile) {
                fprintf(progressFile, "%d", game);
                for (int i = 0; i < numPlayers; i++) {
                    double winRate = (game > 0) ? (wins[i] * 100.0) / game : 0.0;
                    fprintf(progressFile, ",%.2f", winRate);
                }
                fprintf(progressFile, ",%.1f,%d\n", 
                        (avgCredits[0] + avgCredits[1] + avgCredits[2] + avgCredits[3]) / numPlayers,
                        rb->size);
                fflush(progressFile);
            }
        }
        
        // Progress dots (keep existing)
        if (game % (numGames / 100) == 0 && game % gamesPerCheckpoint != 0) {
            printf(".");
            fflush(stdout);
        }
    }
    
    clock_t endTime = clock();
    double totalTime = ((double)(endTime - startTime)) / CLOCKS_PER_SEC;
    
    printf("\n\n");
    printRepeatedChar('=', 70);
    printf("\n");
    printf("PURE REINFORCEMENT LEARNING COMPLETE!\n");
    printRepeatedChar('=', 70);
    printf("\n");
    printf("Total training time: %.2f seconds (%.2f minutes)\n", totalTime, totalTime / 60.0);
    printf("Games played: %d\n", numGames);
    printf("Experience samples: %d\n", rb->size);
    
    // Enhanced final summary with loss information *** NEW ***
    if (lossStats) {
        displaySelfPlaySummary(lossStats, numGames, wins, numPlayers, avgCredits);
    }
    
    printf("\nFINAL EVOLVED STRATEGIES:\n");
    int bestAI = 0;
    int maxWins = wins[0];
    
    for (int i = 0; i < numPlayers; i++) {
        double winRate = (double)wins[i] / numGames;
        printf("AI_%d: %.1f%% (%d wins) | Final Credits: %.0f\n", 
               i, winRate * 100, wins[i], avgCredits[i]);
        
        if (wins[i] > maxWins) {
            maxWins = wins[i];
            bestAI = i;
        }
    }
    
    printf("\nBest evolved AI: AI_%d (%.1f%% win rate)\n", bestAI, (wins[bestAI] * 100.0) / numGames);
    printf("This AI discovered its own strategy through pure self-play!\n");
    
    // Save the best evolved network
    saveNetwork(networks[bestAI], "poker_ai_evolved.dat");
    printf("Best evolved AI saved as: poker_ai_evolved.dat\n");
    
    // Save all networks as backups
    for (int i = 0; i < numPlayers; i++) {
        char filename[100];
        sprintf(filename, "evolved_ai_%d.dat", i);
        saveNetwork(networks[i], filename);
    }
    
    printf("All evolved AIs saved.\n");
    printf("Training logs: selfplay_progress.csv, selfplay_loss_log.csv\n");
    printRepeatedChar('=', 70);
    printf("\n");
    
    // Cleanup *** MODIFIED ***
    if (progressFile) fclose(progressFile);
    if (bootstrap) freeNetwork(bootstrap);
    if (lossStats) freeSelfPlayStats(lossStats);  // *** NEW ***
    for (int i = 0; i < numPlayers; i++) {
        freeNetwork(networks[i]);
    }
    free(networks);
    free(rb->buffer);
    free(rb);
}

// Combined two-phase training function
void trainTwoPhaseAI(int numGames, int numPlayers) {
    printf("\n");
    printRepeatedChar('*', 70);
    printf("\n");
    printf("TWO-PHASE AI TRAINING\n");
    printf("Phase 1: Minimal Bootstrap (basic rules)\n");
    printf("Phase 2: Pure Self-Play Learning (strategy discovery)\n");
    printRepeatedChar('*', 70);
    printf("\n");
    
    // Phase 1: Quick bootstrap
    NeuralNetwork *bootstrap = trainMinimalBootstrap();
    if (!bootstrap) {
        printf("Bootstrap training failed!\n");
        return;
    }
    
    printf("\nBootstrap complete! Press Enter to begin self-play learning...");
    getchar();
    
    // Free bootstrap network (it's saved to file)
    freeNetwork(bootstrap);
    
    // Phase 2: Use existing self-play system
    pureReinforcementLearning(numGames, numPlayers);
    
    printf("\n");
    printRepeatedChar('*', 70);
    printf("\n");
    printf("TWO-PHASE TRAINING COMPLETE!\n");
    printf("Your AI has learned through pure self-play!\n");
    printf("Files created:\n");
    printf("  - poker_ai_bootstrap.dat (Phase 1 result)\n");
    printf("  - poker_ai_evolved.dat (Final self-play champion)\n");
    printf("  - evolved_ai_0.dat through evolved_ai_3.dat (All trained AIs)\n");
    printf("  - selfplay_progress.csv (Training progress log)\n");
    printRepeatedChar('*', 70);
    printf("\n");
}