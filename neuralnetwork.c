#include "neuralnetwork.h"
#include "game.h"
#include "selftrain.h"
#include <string.h>
#include <time.h>

// Global opponent tracking
static OpponentProfile opponentProfiles[MAXPLAYERS];
static bool profilesInitialized = false;

// ===================================================================
// CORE NEURAL NETWORK FUNCTIONS
// ===================================================================

// Create a NN (memory allocation)
NeuralNetwork* createNetwork(int inputSize, int hiddenSize, int outputSize)
{
    NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));
    if (!nn) {
        perror("Could not allocate memory for neural network");
        return NULL;
    }

    nn->inputSize = inputSize;
    nn->hiddenSize = hiddenSize;
    nn->outputSize = outputSize;
    nn->learningRate = LEARNING_RATE;
    nn->activationFunction = ACTIVATION_SIGMOID;

    nn->inputLayer = calloc(inputSize, sizeof(Neuron));
    if (!nn->inputLayer) {
        perror("Could not allocate memory for input layer");
        free(nn);
        return NULL;
    }
    nn->hiddenLayer = calloc(hiddenSize, sizeof(Neuron));
    if (!nn->hiddenLayer) {
        perror("Could not allocate memory for hidden layer");
        free(nn->inputLayer);
        free(nn);
        return NULL;
    }
    nn->outputLayer = calloc(outputSize, sizeof(Neuron));
    if (!nn->outputLayer) {
        perror("Could not allocate memory for output layer");
        free(nn->inputLayer);
        free(nn->hiddenLayer);
        free(nn);
        return NULL;
    }    nn->weightsInputHidden = malloc(inputSize * sizeof(double*));
    if (!nn->weightsInputHidden) {
        perror("Could not allocate memory for weightsInputHidden");
        free(nn->inputLayer);
        free(nn->hiddenLayer);
        free(nn->outputLayer);
        free(nn);
        return NULL;
    }
    
    for (int i = 0; i < inputSize; i++) {
        nn->weightsInputHidden[i] = calloc(hiddenSize, sizeof(double));
        if (!nn->weightsInputHidden[i]) {
            perror("Could not allocate memory for weights");
            // Cleanup previous allocations
            for (int j = 0; j < i; j++) {
                free(nn->weightsInputHidden[j]);
            }
            free(nn->weightsInputHidden);
            free(nn->inputLayer);
            free(nn->hiddenLayer);
            free(nn->outputLayer);
            free(nn);
            return NULL;
        }
    }

    nn->weightsHiddenOutput = malloc(hiddenSize * sizeof(double*));
    if (!nn->weightsHiddenOutput) {
        perror("Could not allocate memory for weightsHiddenOutput");
        // Cleanup
        for (int i = 0; i < inputSize; i++) {
            free(nn->weightsInputHidden[i]);
        }
        free(nn->weightsInputHidden);
        free(nn->inputLayer);
        free(nn->hiddenLayer);
        free(nn->outputLayer);
        free(nn);
        return NULL;
    }
    
    for (int i = 0; i < hiddenSize; i++) {
        nn->weightsHiddenOutput[i] = calloc(outputSize, sizeof(double));
        if (!nn->weightsHiddenOutput[i]) {
            perror("Could not allocate memory for weights");
            // Cleanup
            for (int j = 0; j < i; j++) {
                free(nn->weightsHiddenOutput[j]);
            }
            free(nn->weightsHiddenOutput);
            for (int j = 0; j < inputSize; j++) {
                free(nn->weightsInputHidden[j]);
            }
            free(nn->weightsInputHidden);
            free(nn->inputLayer);
            free(nn->hiddenLayer);
            free(nn->outputLayer);
            free(nn);
            return NULL;
        }
    }

    nn->biasHidden = calloc(hiddenSize, sizeof(double));
    if (!nn->biasHidden) {
        perror("Could not allocate memory for hidden bias");
        // Cleanup all previous allocations
        for (int i = 0; i < hiddenSize; i++) {
            free(nn->weightsHiddenOutput[i]);
        }
        free(nn->weightsHiddenOutput);
        for (int i = 0; i < inputSize; i++) {
            free(nn->weightsInputHidden[i]);
        }
        free(nn->weightsInputHidden);
        free(nn->inputLayer);
        free(nn->hiddenLayer);
        free(nn->outputLayer);
        free(nn);
        return NULL;
    }
    
    nn->biasOutput = calloc(outputSize, sizeof(double));
    if (!nn->biasOutput) {
        perror("Could not allocate memory for output bias");
        // Final cleanup
        free(nn->biasHidden);
        for (int i = 0; i < hiddenSize; i++) {
            free(nn->weightsHiddenOutput[i]);
        }
        free(nn->weightsHiddenOutput);
        for (int i = 0; i < inputSize; i++) {
            free(nn->weightsInputHidden[i]);
        }
        free(nn->weightsInputHidden);
        free(nn->inputLayer);
        free(nn->hiddenLayer);
        free(nn->outputLayer);
        free(nn);
        return NULL;
    }

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
    if (!nn || !input) {
        printf("ERROR: NULL parameter passed to forwardpropagate\n");
        return;
    }
    
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

// ===================================================================
// ENHANCED GAME STATE ENCODING AND DECISION MAKING
// ===================================================================

// Enhanced game state encoding function
void encodeEnhancedGameState(Player *player, Hand *communityCards, int pot, int currentBet, 
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
    
    // FEATURE 0: Hand strength (improved)
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
    
    // FEATURE 17: Recent opponent aggression (NEW IMPLEMENTATION)
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
    
    // FEATURE 18: Pair in hand (RENUMBERED from 20)
    if (cardCount >= 2) {
        output[18] = (playerCards[0].value == playerCards[1].value) ? 1.0 : 0.0;
    }
    
    // FEATURE 19: High card strength (RENUMBERED from 21)
    if (cardCount >= 2) {
        int highCard = (playerCards[0].value > playerCards[1].value) ? 
                       playerCards[0].value : playerCards[1].value;
        output[19] = (double)highCard / 13.0;
    }
}

// Enhanced decision making function
int makeEnhancedDecision(NeuralNetwork *nn, Player *player, Hand *communityCards, 
                        int pot, int currentBet, int numPlayers, int position) {
    double input[INPUT_SIZE];
    
    // Use enhanced game state encoding
    encodeEnhancedGameState(player, communityCards, pot, currentBet, numPlayers, position, input);
    
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

void debugSelfPlayGame() {
    printf("\n=== DEBUGGING SELF-PLAY GAME ===\n");
    
    // Create test networks
    NeuralNetwork *networks[4];
    for (int i = 0; i < 4; i++) {
        networks[i] = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
        printf("Created network %d\n", i);
    }
    
    // Create replay buffer
    ReplayBuffer *rb = createReplayBuffer(1000);
    printf("Created replay buffer\n");
    
    // Test timing
    printf("Starting game simulation...\n");
    clock_t start = clock();
    
    GameRecord record = playEnhancedSelfPlayGame(networks, 4, rb);
    
    clock_t end = clock();
    double gameTime = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("\n=== GAME RESULTS ===\n");
    printf("Game time: %.4f seconds\n", gameTime);
    printf("Winner: %d\n", record.winner);
    printf("Total hands: %d\n", record.totalHands);
    printf("Number of players: %d\n", record.numPlayers);
    
    // Check credits
    double totalCredits = 0;
    printf("Final credits:\n");
    for (int i = 0; i < 4; i++) {
        printf("  Player %d: %.1f credits\n", i, record.finalCredits[i]);
        totalCredits += record.finalCredits[i];
    }
    printf("Total credits: %.1f (should be %d)\n", totalCredits, STARTING_CREDITS * 4);
    
    // Check for impossible values
    bool hasNegative = false;
    bool hasExcessive = false;
    for (int i = 0; i < 4; i++) {
        if (record.finalCredits[i] < 0) {
            hasNegative = true;
            printf("ERROR: Player %d has negative credits!\n", i);
        }
        if (record.finalCredits[i] > STARTING_CREDITS * 5) {
            hasExcessive = true;
            printf("ERROR: Player %d has excessive credits!\n", i);
        }
    }
    
    // Check experience buffer
    printf("Experience buffer size: %d\n", rb->size);
    if (rb->size == 0) {
        printf("ERROR: No experiences recorded!\n");
    }
    
    // Analyze decision patterns
    if (rb->size > 0) {
        int actionCounts[3] = {0, 0, 0}; // Fold, Call, Raise
        for (int i = 0; i < rb->size; i++) {
            if (rb->buffer[i].action >= 0 && rb->buffer[i].action < 3) {
                actionCounts[rb->buffer[i].action]++;
            }
        }
        printf("Decision distribution: Fold=%d, Call=%d, Raise=%d\n", 
               actionCounts[0], actionCounts[1], actionCounts[2]);
        
        if (actionCounts[0] == rb->size) {
            printf("ERROR: All decisions were folds!\n");
        }
    }
    
    // Performance analysis
    printf("\n=== PERFORMANCE ANALYSIS ===\n");
    if (gameTime < 0.001) {
        printf("ERROR: Game finished too quickly (< 1ms)\n");
    } else if (gameTime > 10.0) {
        printf("WARNING: Game took very long (> 10s)\n");
    } else {
        printf("Game timing seems reasonable\n");
    }
    
    if (record.totalHands == 0) {
        printf("CRITICAL ERROR: No hands were played!\n");
    } else if (record.totalHands > 200) {
        printf("WARNING: Too many hands played (%d)\n", record.totalHands);
    } else {
        printf("Hand count seems reasonable (%d hands)\n", record.totalHands);
    }
    
    if (fabs(totalCredits - (STARTING_CREDITS * 4)) > 1.0) {
        printf("CRITICAL ERROR: Credit conservation violated!\n");
        printf("  Expected: %d, Actual: %.1f, Difference: %.1f\n", 
               STARTING_CREDITS * 4, totalCredits, totalCredits - (STARTING_CREDITS * 4));
    } else {
        printf("Credit conservation OK\n");
    }
    
    // Cleanup
    for (int i = 0; i < 4; i++) {
        freeNetwork(networks[i]);
    }
    free(rb->buffer);
    free(rb);
    
    printf("=== END GAME DEBUG ===\n\n");
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
// SIMPLIFIED TRAINING MONITORING FUNCTIONS
// ===================================================================

// Simple repeated character print (no emojis)
void printChar(char c, int count) {
    for (int i = 0; i < count; i++) {
        printf("%c", c);
    }
}

// Initialize training statistics (simplified)
TrainingStats* initTrainingStats(int maxEpochs) {
    TrainingStats *stats = malloc(sizeof(TrainingStats));
    if (!stats) return NULL;
    
    stats->lossHistory = malloc(maxEpochs * sizeof(double));
    stats->epochNumbers = malloc(maxEpochs * sizeof(int));
    
    if (!stats->lossHistory || !stats->epochNumbers) {
        free(stats);
        return NULL;
    }
    
    stats->currentEpoch = 0;
    stats->maxEpochs = maxEpochs;
    stats->bestLoss = 1000000.0;
    stats->bestEpoch = 0;
    stats->initialLoss = 0.0;
    stats->startTime = clock();
    stats->recentLossAverage = 0.0;
    stats->overfittingDetected = false;
    stats->stagnationCount = 0;
    
    stats->logFile = fopen("training_log.csv", "w");
    if (stats->logFile) {
        fprintf(stats->logFile, "Epoch,Loss,Time_Elapsed,Learning_Rate\n");
    }
    
    return stats;
}

// Update training statistics
void updateTrainingStats(TrainingStats *stats, NeuralNetwork *nn, double **inputs, 
                        double **targets, int numSamples, double learningRate) {
    if (!stats || stats->currentEpoch >= stats->maxEpochs) return;
    
    double currentLoss = calculateLoss(nn, inputs, targets, numSamples);
    
    stats->lossHistory[stats->currentEpoch] = currentLoss;
    stats->epochNumbers[stats->currentEpoch] = stats->currentEpoch;
    
    if (currentLoss < stats->bestLoss) {
        stats->bestLoss = currentLoss;
        stats->bestEpoch = stats->currentEpoch;
        stats->stagnationCount = 0;
    } else {
        stats->stagnationCount++;
    }
    
    if (stats->currentEpoch == 0) {
        stats->initialLoss = currentLoss;
    }
    
    // Overfitting detection
    if (stats->currentEpoch >= 50) {
        double recentSum = 0.0;
        for (int i = stats->currentEpoch - 49; i <= stats->currentEpoch; i++) {
            recentSum += stats->lossHistory[i];
        }
        stats->recentLossAverage = recentSum / 50.0;
        
        if (stats->recentLossAverage > stats->bestLoss * 1.5 && stats->stagnationCount > 100) {
            stats->overfittingDetected = true;
        }
    }
    
    // Log to file
    if (stats->logFile) {
        clock_t currentTime = clock();
        double elapsedSeconds = ((double)(currentTime - stats->startTime)) / CLOCKS_PER_SEC;
        fprintf(stats->logFile, "%d,%.6f,%.2f,%.6f\n", 
                stats->currentEpoch, currentLoss, elapsedSeconds, learningRate);
        fflush(stats->logFile);
    }
    
    stats->currentEpoch++;
}

// Simplified training progress display
void displayTrainingProgress(TrainingStats *stats) {
    if (!stats || stats->currentEpoch == 0) return;
    
    int epoch = stats->currentEpoch - 1;
    double currentLoss = stats->lossHistory[epoch];
    
    // Show progress every 100 epochs
    if (epoch % 100 == 0 || epoch == stats->maxEpochs - 1) {
        printf("Epoch %4d/%d | Loss: %.6f", epoch, stats->maxEpochs - 1, currentLoss);
        
        if (epoch > 0) {
            double lossChange = currentLoss - stats->lossHistory[epoch - 1];
            if (lossChange < -0.001) {
                printf(" (improving)");
            } else if (lossChange > 0.001) {
                printf(" (degrading)");
            }
        }
        
        if (epoch == stats->bestEpoch) {
            printf(" BEST");
        }
        
        if (stats->overfittingDetected) {
            printf(" OVERFITTING");
        }
        
        printf("\n");
    }
    
    // Progress bar every 10%
    if (epoch % (stats->maxEpochs / 10) == 0) {
        int progress = (epoch * 50) / stats->maxEpochs;
        printf("Progress: [");
        for (int i = 0; i < 50; i++) {
            printf(i < progress ? "#" : ".");
        }
        printf("] %d%%\n", (epoch * 100) / stats->maxEpochs);
    }
}

// Simplified training summary
void displayTrainingSummary(TrainingStats *stats) {
    if (!stats) return;
    
    clock_t endTime = clock();
    double totalTime = ((double)(endTime - stats->startTime)) / CLOCKS_PER_SEC;
    
    printf("\n");
    printChar('=', 60);
    printf("\nTRAINING COMPLETE\n");
    printChar('=', 60);
    printf("\n");
    
    printf("Summary:\n");
    printf("  Total Epochs: %d\n", stats->currentEpoch);
    printf("  Training Time: %.2f minutes\n", totalTime / 60.0);
    printf("  Initial Loss: %.6f\n", stats->initialLoss);
    printf("  Final Loss: %.6f\n", stats->lossHistory[stats->currentEpoch - 1]);
    printf("  Best Loss: %.6f (Epoch %d)\n", stats->bestLoss, stats->bestEpoch);
    
    double improvement = ((stats->initialLoss - stats->bestLoss) / stats->initialLoss) * 100;
    printf("  Improvement: %.1f%% reduction\n", improvement);
    
    if (stats->overfittingDetected) {
        printf("  Status: Overfitting detected\n");
    } else if (stats->stagnationCount > 50) {
        printf("  Status: Training plateaued\n");
    } else {
        printf("  Status: Normal completion\n");
    }
    
    printf("\nLog saved to: training_log.csv\n");
    printChar('=', 60);
    printf("\n");
}

// Core monitored training function (simplified)
void trainWithMonitoring(NeuralNetwork *nn, double **inputs, double **outputs, 
                        int numSamples, int epochs) {
    printf("Training: %d samples, %d epochs, LR: %.4f\n", 
           numSamples, epochs, nn->learningRate);
    
    TrainingStats *stats = initTrainingStats(epochs);
    if (!stats) {
        printf("Error: Could not initialize training monitoring\n");
        return;
    }
    
    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Train on all samples
        for (int sample = 0; sample < numSamples; sample++) {
            forwardpropagate(nn, inputs[sample]);
            backpropagate(nn, outputs[sample]);
            updateWeights(nn);
        }
        
        updateTrainingStats(stats, nn, inputs, outputs, numSamples, nn->learningRate);
        displayTrainingProgress(stats);
        
        // Early stopping
        if (stats->lossHistory[epoch] < 0.0001) {
            printf("\nEarly stopping: Loss became very small\n");
            break;
        }
        
        if (stats->overfittingDetected) {
            printf("\nEarly stopping: Overfitting detected\n");
            break;
        }
    }
    
    displayTrainingSummary(stats);
    
    // Cleanup
    if (stats->logFile) fclose(stats->logFile);
    free(stats->lossHistory);
    free(stats->epochNumbers);
    free(stats);
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

// ===================================================================
// SIMPLIFIED BOOTSTRAP TRAINING (RENAMED)
// ===================================================================

// Generate basic bootstrap data (renamed and simplified)
void generateBootstrapData(double **inputs, double **outputs, int numSamples) {
    printf("Generating bootstrap data (%d samples)...\n", numSamples);
    
    for (int i = 0; i < numSamples; i++) {
        double handStrength = (double)rand() / RAND_MAX;
        double stackRatio = (double)rand() / RAND_MAX * 5.0;
        double currentBet = (double)rand() / RAND_MAX;
        double playerBet = (double)rand() / RAND_MAX;
        
        // Fill inputs (20 features)
        inputs[i][0] = handStrength;
        inputs[i][1] = 0.5;  // Hand potential
        inputs[i][2] = 0.5;  // Board texture
        inputs[i][3] = 0.5;  // Pot odds
        inputs[i][4] = stackRatio;
        inputs[i][5] = (double)rand() / RAND_MAX;  // Position
        inputs[i][6] = 0.5;  // Number of players
        inputs[i][7] = currentBet;
        inputs[i][8] = playerBet;
        for (int j = 9; j < INPUT_SIZE; j++) {
            inputs[i][j] = (double)rand() / RAND_MAX;
        }
        
        // Basic strategy rules
        double fold = 0.33, call = 0.33, raise = 0.33;
        
        // Rule 1: Don't go all-in with terrible hands
        if (handStrength < 0.1 && currentBet > 0.8) {
            fold = 0.9; call = 0.08; raise = 0.02;
        }
        
        // Rule 2: Don't fold when you can check for free
        if (currentBet == 0.0) {
            fold = 0.05; call = 0.7; raise = 0.25;
        }
        
        // Rule 3: With very strong hands, don't fold
        if (handStrength > 0.8) {
            fold = 0.05; call = 0.4; raise = 0.55;
        }
        
        // Rule 4: Be conservative with low chips
        if (stackRatio < 0.2 && currentBet > 0.5) {
            fold = 0.6; call = 0.35; raise = 0.05;
        }
        
        // Normalize
        double total = fold + call + raise;
        outputs[i][0] = fold / total;
        outputs[i][1] = call / total;
        outputs[i][2] = raise / total;
    }
}

// Bootstrap training (renamed and simplified)
NeuralNetwork* trainBootstrap() {
    printf("\n");
    printChar('=', 50);
    printf("\nPHASE 1: BOOTSTRAP TRAINING\n");
    printChar('=', 50);
    printf("\nTeaching basic rules to prevent crashes...\n");
    
    int numSamples = 500;
    double **inputs = malloc(numSamples * sizeof(double*));
    double **outputs = malloc(numSamples * sizeof(double*));
    
    for (int i = 0; i < numSamples; i++) {
        inputs[i] = malloc(INPUT_SIZE * sizeof(double));
        outputs[i] = malloc(OUTPUT_SIZE * sizeof(double));
    }
    
    generateBootstrapData(inputs, outputs, numSamples);
    
    NeuralNetwork *nn = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    nn->learningRate = 0.05;
    
    trainWithMonitoring(nn, inputs, outputs, numSamples, 200);
    
    saveNetwork(nn, "poker_ai_bootstrap.dat");
    printf("Bootstrap saved as: poker_ai_bootstrap.dat\n");
    
    // Cleanup
    for (int i = 0; i < numSamples; i++) {
        free(inputs[i]);
        free(outputs[i]);
    }
    free(inputs);
    free(outputs);
    
    return nn;
}

// Phase 2: Pure self-play learning starting from bootstrap
void trainSelfPlay(int numGames, int numPlayers) {
    printf("\n");
    printChar('=', 50);
    printf("\nPHASE 2: SELF-PLAY LEARNING\n");
    printChar('=', 50);
    printf("\nStarting self-play from bootstrap...\n");
    
    initializeOpponentProfiles(numPlayers);
    
    NeuralNetwork **networks = malloc(numPlayers * sizeof(NeuralNetwork*));
    NeuralNetwork *bootstrap = loadNetwork("poker_ai_bootstrap.dat");
    
    if (!bootstrap) {
        printf("No bootstrap found - creating fresh networks\n");
        bootstrap = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    }
    
    // Create diverse AIs from bootstrap
    for (int i = 0; i < numPlayers; i++) {
        networks[i] = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
        
        if (bootstrap) {
            // Copy bootstrap weights
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
        
        addGeneticNoise(networks[i], 0.15);
        printf("AI_%d initialized\n", i);
    }
    
    ReplayBuffer *rb = createReplayBuffer(100000);
    int wins[MAXPLAYERS] = {0};
    double avgCredits[MAXPLAYERS] = {0};
    
    FILE *progressFile = fopen("selfplay_progress.csv", "w");
    if (progressFile) {
        fprintf(progressFile, "Game,AI_0_WinRate,AI_1_WinRate,AI_2_WinRate,AI_3_WinRate\n");
    }
    
    clock_t startTime = clock();
    
    // Main self-play loop
    for (int game = 0; game < numGames; game++) {
        GameRecord record = playEnhancedSelfPlayGame(networks, numPlayers, rb);
        
        wins[record.winner]++;
        for (int i = 0; i < numPlayers; i++) {
            avgCredits[i] = (avgCredits[i] * game + record.finalCredits[i]) / (game + 1);
        }
        
        // Training from experience
        if (rb->size > 500 && game % 3 == 0) {
            for (int i = 0; i < numPlayers; i++) {
                trainFromExperience(networks[i], rb, 128);
            }
        }
        
        // Progress checkpoints
        if (game % (numGames / 20) == 0 && game > 0) {
            printf("Game %d/%d (%.1f%%)\n", game, numGames, (game * 100.0) / numGames);
            
            if (progressFile) {
                fprintf(progressFile, "%d", game);
                for (int i = 0; i < numPlayers; i++) {
                    double winRate = (game > 0) ? (wins[i] * 100.0) / game : 0.0;
                    fprintf(progressFile, ",%.2f", winRate);
                }
                fprintf(progressFile, "\n");
                fflush(progressFile);
            }
        }
        
        if (game % (numGames / 100) == 0) {
            printf(".");
            fflush(stdout);
        }
    }
    
    clock_t endTime = clock();
    double totalTime = ((double)(endTime - startTime)) / CLOCKS_PER_SEC;
    
    printf("\n\nSelf-play learning complete!\n");
    printf("Time: %.2f minutes | Games: %d | Experience: %d\n", 
           totalTime / 60.0, numGames, rb->size);
    
    // Find best AI
    int bestAI = 0;
    int maxWins = wins[0];
    for (int i = 0; i < numPlayers; i++) {
        double winRate = (double)wins[i] / numGames;
        printf("AI_%d: %.1f%% win rate (%d wins)\n", i, winRate * 100, wins[i]);
        
        if (wins[i] > maxWins) {
            maxWins = wins[i];
            bestAI = i;
        }
    }
    
    printf("Best AI: AI_%d\n", bestAI);
    saveNetwork(networks[bestAI], "poker_ai_evolved.dat");
    
    // Save all networks
    for (int i = 0; i < numPlayers; i++) {
        char filename[100];
        sprintf(filename, "evolved_ai_%d.dat", i);
        saveNetwork(networks[i], filename);
    }
    
    // Cleanup
    if (progressFile) fclose(progressFile);
    if (bootstrap) freeNetwork(bootstrap);
    for (int i = 0; i < numPlayers; i++) {
        freeNetwork(networks[i]);
    }
    free(networks);
    free(rb->buffer);
    free(rb);
}

// Combined two-phase training (simplified)
void trainTwoPhaseAI(int numGames, int numPlayers) {
    printf("\n");
    printChar('*', 60);
    printf("\nTWO-PHASE AI TRAINING\n");
    printf("Phase 1: Bootstrap (basic rules)\n");
    printf("Phase 2: Self-play (strategy discovery)\n");
    printChar('*', 60);
    printf("\n");
    
    // Phase 1
    NeuralNetwork *bootstrap = trainBootstrap();
    freeNetwork(bootstrap);
    
    printf("\nBootstrap complete. Press Enter to begin self-play...");
    getchar();
    
    // Phase 2
    trainSelfPlay(numGames, numPlayers);
    
    printf("\nTwo-phase training complete!\n");
    printf("Files created:\n");
    printf("  - poker_ai_bootstrap.dat\n");
    printf("  - poker_ai_evolved.dat\n");
    printf("  - evolved_ai_*.dat\n");
    printf("  - selfplay_progress.csv\n");
}


// ===================================================================
// EVOLUTIONARY TRAINER CREATION AND MANAGEMENT
// ===================================================================

EvolutionaryTrainer* createEvolutionaryTrainer(int populationSize, int maxGenerations) {
    printf("Creating evolutionary trainer for %d AIs over %d generations...\n", 
           populationSize, maxGenerations);
    
    EvolutionaryTrainer *trainer = malloc(sizeof(EvolutionaryTrainer));
    if (!trainer) {
        printf("Error: Could not allocate memory for evolutionary trainer\n");
        return NULL;
    }
    
    // Initialize basic parameters
    trainer->populationSize = populationSize;
    trainer->maxGenerations = maxGenerations;
    trainer->currentGeneration = 0;
    trainer->gamesPerGeneration = 50;  // REDUCED: Start with 50 games per generation for testing
    
    // FIXED Evolution parameters - better balance
    trainer->selectionRate = 0.20;     // Keep top 20%
    trainer->mutationRate = 0.40;      // 40% mutation (reduced)
    trainer->crossoverRate = 0.30;     // 30% crossover  
    trainer->freshRate = 0.10;         // 10% fresh blood (was 0.20, causing overflow)
    
    // Verify parameters add up correctly
    double totalRate = trainer->selectionRate + trainer->mutationRate + 
                      trainer->crossoverRate + trainer->freshRate;
    if (totalRate > 1.0) {
        printf("Warning: Evolution rates sum to %.2f, adjusting...\n", totalRate);
        trainer->freshRate = 1.0 - trainer->selectionRate - trainer->mutationRate - trainer->crossoverRate;
        if (trainer->freshRate < 0) trainer->freshRate = 0;
    }
    
    printf("Evolution rates: Keep %.0f%% | Mutate %.0f%% | Cross %.0f%% | Fresh %.0f%%\n",
           trainer->selectionRate * 100, trainer->mutationRate * 100,
           trainer->crossoverRate * 100, trainer->freshRate * 100);
    
    // Performance tracking
    trainer->bestFitnessEver = 0.0;
    trainer->bestIndividualEver = -1;
    for (int i = 0; i < 50; i++) {
        trainer->avgFitnessHistory[i] = 0.0;
    }
    for (int i = 0; i < 10; i++) {
        trainer->hallOfFame[i] = NULL;
    }
    
    // Allocate population
    trainer->population = malloc(populationSize * sizeof(Individual));
    if (!trainer->population) {
        printf("Error: Could not allocate memory for population\n");
        free(trainer);
        return NULL;
    }
    
    // Tournament setup (4 players per table)
    trainer->numTables = populationSize / 4;
    trainer->tables = malloc(trainer->numTables * sizeof(TournamentTable));
    if (!trainer->tables) {
        printf("Error: Could not allocate memory for tournament tables\n");
        free(trainer->population);
        free(trainer);
        return NULL;
    }
    
    // Open evolution log
    trainer->evolutionLog = fopen("evolution_log.csv", "w");
    if (trainer->evolutionLog) {
        fprintf(trainer->evolutionLog, "Generation,Individual,Fitness,WinRate,AvgCredits,Strategy,ParentA,ParentB\n");
    }
    
    printf("Evolutionary trainer created successfully!\n");
    printf("Population: %d AIs | Generations: %d | Tables: %d | Games/Gen: %d\n", 
           populationSize, maxGenerations, trainer->numTables, trainer->gamesPerGeneration);
    
    return trainer;
}

void freeEvolutionaryTrainer(EvolutionaryTrainer *trainer) {
    if (!trainer) return;
    
    // Free all neural networks in population
    for (int i = 0; i < trainer->populationSize; i++) {
        if (trainer->population[i].network) {
            freeNetwork(trainer->population[i].network);
        }
    }
    
    // Free arrays
    free(trainer->population);
    free(trainer->tables);
    
    // Close log file
    if (trainer->evolutionLog) {
        fclose(trainer->evolutionLog);
    }
    
    free(trainer);
    printf("Evolutionary trainer cleaned up.\n");
}

// ===================================================================
// POPULATION INITIALIZATION
// ===================================================================

void initializePopulation(EvolutionaryTrainer *trainer) {
    printf("Initializing diverse population of %d AIs...\n", trainer->populationSize);
    
    // Try to load bootstrap network as starting point
    NeuralNetwork *bootstrap = loadNetwork("poker_ai_bootstrap.dat");
    if (!bootstrap) {
        printf("No bootstrap found - creating from scratch\n");
        bootstrap = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    } else {
        printf("Using bootstrap as evolutionary starting point\n");
    }
    
    // Create diverse population
    for (int i = 0; i < trainer->populationSize; i++) {
        Individual *individual = &trainer->population[i];
        
        // Create network (copy bootstrap + noise)
        individual->network = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
        
        if (bootstrap) {
            // Copy bootstrap weights
            for (int inp = 0; inp < INPUT_SIZE; inp++) {
                for (int hid = 0; hid < HIDDEN_SIZE; hid++) {
                    individual->network->weightsInputHidden[inp][hid] = 
                        bootstrap->weightsInputHidden[inp][hid];
                }
            }
            for (int hid = 0; hid < HIDDEN_SIZE; hid++) {
                for (int out = 0; out < OUTPUT_SIZE; out++) {
                    individual->network->weightsHiddenOutput[hid][out] = 
                        bootstrap->weightsHiddenOutput[hid][out];
                }
                individual->network->biasHidden[hid] = bootstrap->biasHidden[hid];
            }
            for (int out = 0; out < OUTPUT_SIZE; out++) {
                individual->network->biasOutput[out] = bootstrap->biasOutput[out];
            }
        }
        
        // Add increasing levels of noise for diversity
        double noiseLevel = 0.1 + (i / (double)trainer->populationSize) * 0.4; // 0.1 to 0.5
        addGeneticNoise(individual->network, noiseLevel);
        
        // Initialize individual stats
        individual->fitness = 0.0;
        individual->wins = 0;
        individual->games = 0;
        individual->totalCredits = 0.0;
        individual->avgCredits = 0.0;
        individual->winRate = 0.0;
        individual->generation = 0;
        individual->parentA = -1;  // No parents for generation 0
        individual->parentB = -1;
        strcpy(individual->strategy, "Unknown");
        
        // Progress indicator
        if (i % 100 == 0) {
            printf("Created %d/%d AIs...\n", i, trainer->populationSize);
        }
    }
    
    if (bootstrap) {
        freeNetwork(bootstrap);
    }
    
    printf("Population initialized with diverse strategies!\n");
}

// ===================================================================
// GENETIC OPERATIONS
// ===================================================================

void addGeneticNoise(NeuralNetwork *nn, double noiseLevel) {
    // Add random noise to all weights for genetic diversity
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            double noise = ((double)rand() / RAND_MAX - 0.5) * 2.0 * noiseLevel;
            nn->weightsInputHidden[i][j] += noise;
        }
    }
    
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            double noise = ((double)rand() / RAND_MAX - 0.5) * 2.0 * noiseLevel;
            nn->weightsHiddenOutput[i][j] += noise;
        }
        double noise = ((double)rand() / RAND_MAX - 0.5) * 2.0 * noiseLevel;
        nn->biasHidden[i] += noise;
    }
    
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        double noise = ((double)rand() / RAND_MAX - 0.5) * 2.0 * noiseLevel;
        nn->biasOutput[i] += noise;
    }
}

void mutateWeights(NeuralNetwork *nn, double mutationStrength) {
    // Stronger mutation than genetic noise - for evolution
    int totalWeights = (INPUT_SIZE * HIDDEN_SIZE) + (HIDDEN_SIZE * OUTPUT_SIZE) + 
                       HIDDEN_SIZE + OUTPUT_SIZE;
    int mutationsToMake = (int)(totalWeights * mutationStrength);
    
    for (int m = 0; m < mutationsToMake; m++) {
        int weightType = rand() % 4;
        
        switch (weightType) {
            case 0: // Input-Hidden weights
                {
                    int i = rand() % INPUT_SIZE;
                    int j = rand() % HIDDEN_SIZE;
                    double mutation = ((double)rand() / RAND_MAX - 0.5) * 0.5;
                    nn->weightsInputHidden[i][j] += mutation;
                }
                break;
                
            case 1: // Hidden-Output weights
                {
                    int i = rand() % HIDDEN_SIZE;
                    int j = rand() % OUTPUT_SIZE;
                    double mutation = ((double)rand() / RAND_MAX - 0.5) * 0.5;
                    nn->weightsHiddenOutput[i][j] += mutation;
                }
                break;
                
            case 2: // Hidden biases
                {
                    int i = rand() % HIDDEN_SIZE;
                    double mutation = ((double)rand() / RAND_MAX - 0.5) * 0.5;
                    nn->biasHidden[i] += mutation;
                }
                break;
                
            case 3: // Output biases
                {
                    int i = rand() % OUTPUT_SIZE;
                    double mutation = ((double)rand() / RAND_MAX - 0.5) * 0.5;
                    nn->biasOutput[i] += mutation;
                }
                break;
        }
    }
}

NeuralNetwork* crossoverNetworks(NeuralNetwork *parentA, NeuralNetwork *parentB) {
    NeuralNetwork *child = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    
    // Randomly choose weights from either parent
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            if (rand() % 2) {
                child->weightsInputHidden[i][j] = parentA->weightsInputHidden[i][j];
            } else {
                child->weightsInputHidden[i][j] = parentB->weightsInputHidden[i][j];
            }
        }
    }
    
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (rand() % 2) {
                child->weightsHiddenOutput[i][j] = parentA->weightsHiddenOutput[i][j];
            } else {
                child->weightsHiddenOutput[i][j] = parentB->weightsHiddenOutput[i][j];
            }
        }
        
        if (rand() % 2) {
            child->biasHidden[i] = parentA->biasHidden[i];
        } else {
            child->biasHidden[i] = parentB->biasHidden[i];
        }
    }
    
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        if (rand() % 2) {
            child->biasOutput[i] = parentA->biasOutput[i];
        } else {
            child->biasOutput[i] = parentB->biasOutput[i];
        }
    }
    
    return child;
}

Individual* createMutatedOffspring(Individual *parent, int parentIndex, int generation) {
    Individual *child = malloc(sizeof(Individual));
    
    // Copy parent network and mutate
    child->network = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    
    // Copy all weights from parent
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            child->network->weightsInputHidden[i][j] = parent->network->weightsInputHidden[i][j];
        }
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            child->network->weightsHiddenOutput[i][j] = parent->network->weightsHiddenOutput[i][j];
        }
        child->network->biasHidden[i] = parent->network->biasHidden[i];
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        child->network->biasOutput[i] = parent->network->biasOutput[i];
    }
    
    // Apply mutation
    mutateWeights(child->network, 0.05 + (double)rand() / RAND_MAX * 0.15); // 5-20% mutation
    
    // Initialize child stats
    child->fitness = 0.0;
    child->wins = 0;
    child->games = 0;
    child->totalCredits = 0.0;
    child->avgCredits = 0.0;
    child->winRate = 0.0;
    child->generation = generation;
    child->parentA = parentIndex;  // Store parent index
    child->parentB = -1;
    strcpy(child->strategy, "Mutated");
    
    return child;
}

Individual* createCrossoverOffspring(Individual *parentA, Individual *parentB, int parentAIndex, int parentBIndex, int generation) {
    Individual *child = malloc(sizeof(Individual));
    
    // Create hybrid network
    child->network = crossoverNetworks(parentA->network, parentB->network);
    
    // Initialize child stats
    child->fitness = 0.0;
    child->wins = 0;
    child->games = 0;
    child->totalCredits = 0.0;
    child->avgCredits = 0.0;
    child->winRate = 0.0;
    child->generation = generation;
    child->parentA = parentAIndex;  // Store parent A index
    child->parentB = parentBIndex;  // Store parent B index
    strcpy(child->strategy, "Crossover");
    
    return child;
}

Individual* createFreshIndividual(int generation) {
    Individual *individual = malloc(sizeof(Individual));
    
    // Create completely new random network
    individual->network = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    addGeneticNoise(individual->network, 0.3); // Moderate noise for fresh individuals
    
    // Initialize stats
    individual->fitness = 0.0;
    individual->wins = 0;
    individual->games = 0;
    individual->totalCredits = 0.0;
    individual->avgCredits = 0.0;
    individual->winRate = 0.0;
    individual->generation = generation;
    individual->parentA = -1;
    individual->parentB = -1;
    strcpy(individual->strategy, "Fresh");
    
    return individual;
}


// ===================================================================
// TOURNAMENT SYSTEM
// ===================================================================

void setupTournamentTables(EvolutionaryTrainer *trainer) {
    // Randomly assign AIs to tables for this generation
    int *availableAIs = malloc(trainer->populationSize * sizeof(int));
    for (int i = 0; i < trainer->populationSize; i++) {
        availableAIs[i] = i;
    }
    
    // Shuffle the available AIs
    for (int i = trainer->populationSize - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = availableAIs[i];
        availableAIs[i] = availableAIs[j];
        availableAIs[j] = temp;
    }
    
    // Assign to tables
    for (int table = 0; table < trainer->numTables; table++) {
        for (int seat = 0; seat < 4; seat++) {
            trainer->tables[table].playerIndices[seat] = availableAIs[table * 4 + seat];
        }
        trainer->tables[table].completed = false;
        trainer->tables[table].winner = -1;
    }
    
    free(availableAIs);
}

void runTournamentGenerationDebug(EvolutionaryTrainer *trainer) {
    printf("Running generation %d tournament with DEBUG...\n", trainer->currentGeneration + 1);
    trainer->generationStartTime = clock();
    
    // Run initial diagnostic
    if (trainer->currentGeneration == 0) {
        printf("Running self-play game diagnostic...\n");
        debugSelfPlayGame();
    }
    
    // Reset stats
    for (int i = 0; i < trainer->populationSize; i++) {
        trainer->population[i].wins = 0;
        trainer->population[i].games = 0;
        trainer->population[i].totalCredits = 0.0;
    }
    
    int totalGamePlaysNeeded = trainer->populationSize * trainer->gamesPerGeneration;
    int totalTableGamesNeeded = totalGamePlaysNeeded / 4;
    int roundsPerGeneration = totalTableGamesNeeded / trainer->numTables;
    
    printf("Target: %d games per AI | Table games needed: %d | Rounds: %d\n", 
           trainer->gamesPerGeneration, totalTableGamesNeeded, roundsPerGeneration);
    
    // Track timing
    clock_t roundStart = clock();
    int suspiciousTables = 0;
    
    for (int round = 0; round < roundsPerGeneration; round++) {
        printf("  Round %d/%d: ", round + 1, roundsPerGeneration);
        
        setupTournamentTables(trainer);
        
        for (int table = 0; table < trainer->numTables; table++) {
            bool verbose = (table < 3 && round == 0); // Verbose for first few tables
            runSingleTableDebug(trainer, table, verbose);
            
            if (table % 10 == 0) {
                printf(".");
                fflush(stdout);
            }
        }
        printf(" Complete\n");
        
        // Check timing after first round
        if (round == 0) {
            clock_t roundEnd = clock();
            double roundTime = ((double)(roundEnd - roundStart)) / CLOCKS_PER_SEC;
            printf("  First round time: %.2f seconds\n", roundTime);
            
            if (roundTime < 0.1) {
                printf("  WARNING: Round finished too quickly!\n");
            }
        }
    }    // Final validation
    int minGames = trainer->populationSize;
    int maxGames = 0;
    int negativeCredits = 0;
    double totalProfit = 0;
    
    for (int i = 0; i < trainer->populationSize; i++) {
        Individual *individual = &trainer->population[i];
        if (individual->games > 0) {
            individual->winRate = (double)individual->wins / individual->games;
            individual->avgCredits = individual->totalCredits / individual->games;
            
            if (individual->games < minGames) minGames = individual->games;
            if (individual->games > maxGames) maxGames = individual->games;
            
            if (individual->avgCredits < 0) negativeCredits++;
            
            double profit = individual->avgCredits - STARTING_CREDITS;
            totalProfit += profit;
        }
    }
    
    double avgProfit = totalProfit / trainer->populationSize;
    
    printf("\n=== GENERATION VALIDATION ===\n");
    printf("Games per AI: min=%d, max=%d (target: %d)\n", minGames, maxGames, trainer->gamesPerGeneration);
    printf("AIs with negative avg credits: %d/%d\n", negativeCredits, trainer->populationSize);
    printf("Average profit per AI: %.1f credits (%.1f%%)\n", avgProfit, (avgProfit / STARTING_CREDITS) * 100);
    
    if (negativeCredits > trainer->populationSize * 0.8) {
        printf("CRITICAL: >80%% of AIs have negative credits - game simulation is broken!\n");
    }
    
    if (avgProfit < -STARTING_CREDITS * 0.5) {
        printf("CRITICAL: Average loss >50%% - something is very wrong!\n");
    }
    
    printf("Generation %d complete\n", trainer->currentGeneration + 1);
}

void runSingleTableDebug(EvolutionaryTrainer *trainer, int tableIndex, bool verbose) {
    TournamentTable *table = &trainer->tables[tableIndex];
    
    if (verbose) printf("Running table %d with AIs: ", tableIndex);
    
    // Create temporary player array
    Player players[4];
    NeuralNetwork *networks[4];
    
    // Set up players
    for (int i = 0; i < 4; i++) {
        int aiIndex = table->playerIndices[i];
        networks[i] = trainer->population[aiIndex].network;
        
        sprintf(players[i].name, "AI_%d", aiIndex);
        players[i].credits = STARTING_CREDITS;
        players[i].currentBet = 0;
        players[i].hand = NULL;
        players[i].status = ACTIVE;
        players[i].dealer = (i == 0);
        
        if (verbose) printf("%d ", aiIndex);
    }
    if (verbose) printf("\n");
    
    // Create replay buffer
    ReplayBuffer *tempBuffer = createReplayBuffer(1000);
    if (!tempBuffer) {
        printf("ERROR: Could not create temp buffer for table %d\n", tableIndex);
        return;
    }
    
    // Time the game
    clock_t gameStart = clock();
    GameRecord record = playEnhancedSelfPlayGame(networks, 4, tempBuffer);
    clock_t gameEnd = clock();
    double gameTime = ((double)(gameEnd - gameStart)) / CLOCKS_PER_SEC;
    
    if (verbose) {
        printf("  Game time: %.4f seconds\n", gameTime);
        printf("  Hands played: %d\n", record.totalHands);
        printf("  Winner: %d\n", record.winner);
    }
    
    // Validate results before updating stats
    bool valid = true;
    double totalCredits = 0;
    
    for (int i = 0; i < 4; i++) {
        totalCredits += record.finalCredits[i];
        
        if (record.finalCredits[i] < 0) {
            if (verbose) printf("  ERROR: AI %d has negative credits: %.1f\n", 
                               table->playerIndices[i], record.finalCredits[i]);
            record.finalCredits[i] = 0; // Fix negative credits
            valid = false;
        }
        
        if (record.finalCredits[i] > STARTING_CREDITS * 10) {
            if (verbose) printf("  ERROR: AI %d has excessive credits: %.1f\n", 
                               table->playerIndices[i], record.finalCredits[i]);
            record.finalCredits[i] = STARTING_CREDITS * 4; // Cap excessive credits
            valid = false;
        }
    }
    
    // Check credit conservation
    double expectedTotal = 4 * STARTING_CREDITS;
    if (fabs(totalCredits - expectedTotal) > 1.0) {
        if (verbose) printf("  ERROR: Credit conservation violated. Expected: %.0f, Actual: %.1f\n", 
                           expectedTotal, totalCredits);
        
        // Fix by redistributing proportionally
        if (totalCredits > 0) {
            double ratio = expectedTotal / totalCredits;
            for (int i = 0; i < 4; i++) {
                record.finalCredits[i] *= ratio;
            }
        } else {
            // Fallback: equal distribution
            for (int i = 0; i < 4; i++) {
                record.finalCredits[i] = STARTING_CREDITS;
            }
        }
        valid = false;
    }
    
    // Update individual statistics
    for (int i = 0; i < 4; i++) {
        int aiIndex = table->playerIndices[i];
        Individual *individual = &trainer->population[aiIndex];
        
        individual->games++;
        individual->totalCredits += record.finalCredits[i];
        
        if (record.winner == i) {
            individual->wins++;
        }
        
        table->finalCredits[i] = record.finalCredits[i];
    }
    
    table->winner = record.winner;
    table->completed = true;
    
    // Log suspicious games
    if (record.totalHands == 0) {
        printf("SUSPICIOUS TABLE %d: valid=%s, time=%.4fs, hands=%d\n", 
               tableIndex, valid ? "true" : "false", gameTime, record.totalHands);
    }
    
    // Cleanup
    if (tempBuffer->buffer) {
        free(tempBuffer->buffer);
    }
    free(tempBuffer);
    
    for (int i = 0; i < 4; i++) {
        if (players[i].hand) {
            freeHand(players[i].hand, 1);
            players[i].hand = NULL;
        }
    }
}
// ===================================================================
// FITNESS EVALUATION
// ===================================================================

double calculateIndividualFitness(Individual *individual) {
    if (individual->games == 0) return 0.0;
    
    // IMPROVED: More balanced multi-objective fitness function
    double winRateScore = individual->winRate * 40.0;  // 0-40 points (reduced weight)
    
    // Profit-based scoring with diminishing returns to prevent runaway values
    double profitMargin = (individual->avgCredits - STARTING_CREDITS) / STARTING_CREDITS;
    double profitScore = 0.0;
    
    if (profitMargin >= 0) {
        // Positive profits: diminishing returns after 100% profit
        if (profitMargin <= 1.0) {
            profitScore = profitMargin * 60.0;  // 0-60 points for 0-100% profit
        } else {
            // Diminishing returns for excessive profits
            profitScore = 60.0 + (profitMargin - 1.0) * 20.0;  // Additional 20 points per 100% above 100%
            if (profitScore > 100.0) profitScore = 100.0;  // Cap at 100
        }
    } else {
        // Negative profits: penalty but not too harsh
        profitScore = profitMargin * 40.0;  // Can go down to -40 for 100% loss
        if (profitScore < -50.0) profitScore = -50.0;  // Floor at -50
    }
    
    // Consistency bonus (smaller impact)
    double consistencyBonus = 0.0;
    if (individual->games >= 15) {
        if (individual->avgCredits > STARTING_CREDITS * 1.2) consistencyBonus = 8.0;
        else if (individual->avgCredits > STARTING_CREDITS * 1.0) consistencyBonus = 5.0;
        else if (individual->avgCredits > STARTING_CREDITS * 0.8) consistencyBonus = 2.0;
    }
    
    // Experience bonus (reduced impact)
    double experienceBonus = fmin(individual->games / 30.0, 3.0); // Up to 3 points
    
    // ADDED: Diversity bonus to encourage different strategies
    double diversityBonus = 0.0;
    if (individual->winRate < 0.95 && individual->winRate > 0.05) {
        diversityBonus = 2.0;  // Small bonus for not being too extreme
    }
    
    // Total fitness
    double fitness = winRateScore + profitScore + consistencyBonus + experienceBonus + diversityBonus;
    
    return fmax(fitness, 0.1); // Minimum 0.1 to avoid zero fitness
}

void evaluatePopulationFitness(EvolutionaryTrainer *trainer) {
    printf("Evaluating fitness...\n");
    
    double bestFitness = 0.0;
    int bestIndex = 0;
    
    for (int i = 0; i < trainer->populationSize; i++) {
        Individual *individual = &trainer->population[i];
        individual->fitness = calculateIndividualFitness(individual);
        
        if (individual->fitness > bestFitness) {
            bestFitness = individual->fitness;
            bestIndex = i;
        }
        
        assignStrategyLabel(individual);
    }
    
    // Update global best
    if (bestFitness > trainer->bestFitnessEver) {
        trainer->bestFitnessEver = bestFitness;
        trainer->bestIndividualEver = bestIndex;
        updateHallOfFame(trainer, &trainer->population[bestIndex]);
        printf("NEW BEST AI FOUND! Fitness: %.2f\n", bestFitness);
    }
    
    double avgFitness = 0;
    for (int i = 0; i < trainer->populationSize; i++) {
        avgFitness += trainer->population[i].fitness;
    }
    avgFitness /= trainer->populationSize;
    trainer->avgFitnessHistory[trainer->currentGeneration] = avgFitness;
    
    printf("Fitness - Best: %.2f | Average: %.2f\n", bestFitness, avgFitness);
}

void assignStrategyLabel(Individual *individual) {
    if (individual->games < 5) {
        strcpy(individual->strategy, "Untested");
        return;
    }
    
    double winRate = individual->winRate;
    double avgCredits = individual->avgCredits;
    double profitMargin = (avgCredits - STARTING_CREDITS) / STARTING_CREDITS;
    
    // FIXED: More balanced strategy classification with higher thresholds
    // This should create more diverse strategy distribution
    if (winRate >= 0.75 && profitMargin >= 1.5) {
        strcpy(individual->strategy, "Dominant");
    } else if (winRate >= 0.60 && profitMargin >= 1.0) {
        strcpy(individual->strategy, "Aggressive");
    } else if (winRate >= 0.45 && profitMargin >= 0.5) {
        strcpy(individual->strategy, "Conservative");
    } else if (winRate >= 0.30 && profitMargin >= 0.0) {
        strcpy(individual->strategy, "Survivor");
    } else if (profitMargin >= -0.3) {
        strcpy(individual->strategy, "Struggling");
    } else {
        strcpy(individual->strategy, "Weak");
    }
    
    // Special case: High win rate but low profits = too passive
    if (winRate >= 0.50 && profitMargin < 0.3) {
        strcpy(individual->strategy, "Passive");
    }
    
    // Special case: Low win rate but high profits = lucky/aggressive
    if (winRate < 0.40 && profitMargin > 0.8) {
        strcpy(individual->strategy, "Lucky");
    }
    
    // Special case: Very high profits regardless of win rate
    if (profitMargin >= 2.0) {
        strcpy(individual->strategy, "Elite");
    }
}
void updateHallOfFame(EvolutionaryTrainer *trainer, Individual *candidate) {
    // Create a copy of the candidate for hall of fame
    Individual *candidateCopy = malloc(sizeof(Individual));
    if (!candidateCopy) return;
    
    // Copy all data (but not the network pointer)
    candidateCopy->fitness = candidate->fitness;
    candidateCopy->wins = candidate->wins;
    candidateCopy->games = candidate->games;
    candidateCopy->totalCredits = candidate->totalCredits;
    candidateCopy->avgCredits = candidate->avgCredits;
    candidateCopy->winRate = candidate->winRate;
    candidateCopy->generation = candidate->generation;
    candidateCopy->parentA = candidate->parentA;
    candidateCopy->parentB = candidate->parentB;
    strcpy(candidateCopy->strategy, candidate->strategy);
    candidateCopy->network = NULL; // Don't copy network pointer
    
    // Find insertion point in hall of fame
    for (int i = 0; i < 10; i++) {
        if (trainer->hallOfFame[i] == NULL || 
            candidateCopy->fitness > trainer->hallOfFame[i]->fitness) {
            
            // Shift others down and free displaced entry
            if (trainer->hallOfFame[9] != NULL) {
                free(trainer->hallOfFame[9]);
            }
            
            for (int j = 9; j > i; j--) {
                trainer->hallOfFame[j] = trainer->hallOfFame[j-1];
            }
            
            // Insert new candidate
            trainer->hallOfFame[i] = candidateCopy;
            return;
        }
    }
    
    // If we get here, candidate wasn't good enough
    free(candidateCopy);
}


// Comparison function for qsort
int compareFitness(const void *a, const void *b) {
    const Individual *individualA = (const Individual *)a;
    const Individual *individualB = (const Individual *)b;
    
    if (individualB->fitness > individualA->fitness) return 1;
    if (individualB->fitness < individualA->fitness) return -1;
    return 0;
}

void rankPopulation(EvolutionaryTrainer *trainer) {
    qsort(trainer->population, trainer->populationSize, sizeof(Individual), compareFitness);
    
    printf("Top 3 AIs:\n");
    for (int i = 0; i < 3; i++) {
        Individual *ai = &trainer->population[i];
        printf("  #%d: Fitness %.2f | Win Rate %.1f%% | %s\n",
               i + 1, ai->fitness, ai->winRate * 100, ai->strategy);
    }
}

// ADD TO neuralnetwork.c - Evolution Operations and Main Training Loop

// ===================================================================
// EVOLUTION OPERATIONS
// ===================================================================

void evolvePopulation(EvolutionaryTrainer *trainer) {
    printf("Evolving population to generation %d...\n", trainer->currentGeneration + 2);
    
    // Calculate how many of each type to create
    int numToKeep = (int)(trainer->populationSize * trainer->selectionRate);
    int numMutations = (int)(trainer->populationSize * trainer->mutationRate);
    int numCrossovers = (int)(trainer->populationSize * trainer->crossoverRate);
    int numFresh = trainer->populationSize - numToKeep - numMutations - numCrossovers;
    
    printf("Evolution plan: Keep %d | Mutate %d | Crossover %d | Fresh %d\n",
           numToKeep, numMutations, numCrossovers, numFresh);
    
    // Create new population array
    Individual *newPopulation = malloc(trainer->populationSize * sizeof(Individual));
    int newIndex = 0;
    
    // 1. Keep top performers (already sorted by rankPopulation)
    printf("Keeping top %d performers...\n", numToKeep);
    for (int i = 0; i < numToKeep; i++) {
        newPopulation[newIndex] = trainer->population[i];
        newPopulation[newIndex].generation = trainer->currentGeneration + 1;
        newIndex++;
    }
    
    // 2. Create mutations from top performers
    printf("Creating %d mutations...\n", numMutations);
    for (int i = 0; i < numMutations; i++) {
        // Select random parent from top 50%
        int parentIndex = rand() % (trainer->populationSize / 2);
        Individual *parent = &trainer->population[parentIndex];
        
        // Create mutated offspring using helper function
        // But we need to manually copy since we're working with array slots
        newPopulation[newIndex].network = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
        
        // Copy parent weights
        for (int inp = 0; inp < INPUT_SIZE; inp++) {
            for (int hid = 0; hid < HIDDEN_SIZE; hid++) {
                newPopulation[newIndex].network->weightsInputHidden[inp][hid] = 
                    parent->network->weightsInputHidden[inp][hid];
            }
        }
        for (int hid = 0; hid < HIDDEN_SIZE; hid++) {
            for (int out = 0; out < OUTPUT_SIZE; out++) {
                newPopulation[newIndex].network->weightsHiddenOutput[hid][out] = 
                    parent->network->weightsHiddenOutput[hid][out];
            }
            newPopulation[newIndex].network->biasHidden[hid] = parent->network->biasHidden[hid];
        }
        for (int out = 0; out < OUTPUT_SIZE; out++) {
            newPopulation[newIndex].network->biasOutput[out] = parent->network->biasOutput[out];
        }
        
        // Apply mutation
        mutateWeights(newPopulation[newIndex].network, 0.05 + (double)rand() / RAND_MAX * 0.15);
        
        // Initialize stats
        newPopulation[newIndex].fitness = 0.0;
        newPopulation[newIndex].wins = 0;
        newPopulation[newIndex].games = 0;
        newPopulation[newIndex].totalCredits = 0.0;
        newPopulation[newIndex].avgCredits = 0.0;
        newPopulation[newIndex].winRate = 0.0;
        newPopulation[newIndex].generation = trainer->currentGeneration + 1;
        newPopulation[newIndex].parentA = parentIndex;
        newPopulation[newIndex].parentB = -1;
        strcpy(newPopulation[newIndex].strategy, "Mutated");
        
        newIndex++;
    }
    
    // 3. Create crossovers from top performers
    printf("Creating %d crossovers...\n", numCrossovers);
    for (int i = 0; i < numCrossovers; i++) {
        // Select two different parents from top 50%
        int parentAIndex = rand() % (trainer->populationSize / 2);
        int parentBIndex = rand() % (trainer->populationSize / 2);
        while (parentBIndex == parentAIndex) {
            parentBIndex = rand() % (trainer->populationSize / 2);
        }
        
        // Create crossover child
        newPopulation[newIndex].network = crossoverNetworks(
            trainer->population[parentAIndex].network,
            trainer->population[parentBIndex].network
        );
        
        // Initialize stats
        newPopulation[newIndex].fitness = 0.0;
        newPopulation[newIndex].wins = 0;
        newPopulation[newIndex].games = 0;
        newPopulation[newIndex].totalCredits = 0.0;
        newPopulation[newIndex].avgCredits = 0.0;
        newPopulation[newIndex].winRate = 0.0;
        newPopulation[newIndex].generation = trainer->currentGeneration + 1;
        newPopulation[newIndex].parentA = parentAIndex;
        newPopulation[newIndex].parentB = parentBIndex;
        strcpy(newPopulation[newIndex].strategy, "Crossover");
        
        newIndex++;
    }
    
    // 4. Create fresh random individuals
    printf("Creating %d fresh individuals...\n", numFresh);
    for (int i = 0; i < numFresh; i++) {
        newPopulation[newIndex].network = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
        addGeneticNoise(newPopulation[newIndex].network, 0.3);
        
        // Initialize stats
        newPopulation[newIndex].fitness = 0.0;
        newPopulation[newIndex].wins = 0;
        newPopulation[newIndex].games = 0;
        newPopulation[newIndex].totalCredits = 0.0;
        newPopulation[newIndex].avgCredits = 0.0;
        newPopulation[newIndex].winRate = 0.0;
        newPopulation[newIndex].generation = trainer->currentGeneration + 1;
        newPopulation[newIndex].parentA = -1;
        newPopulation[newIndex].parentB = -1;
        strcpy(newPopulation[newIndex].strategy, "Fresh");
        
        newIndex++;
    }
    
    // Free old networks (except those kept in new population)
    for (int i = numToKeep; i < trainer->populationSize; i++) {
        freeNetwork(trainer->population[i].network);
    }
    
    // Replace population
    free(trainer->population);
    trainer->population = newPopulation;
    trainer->currentGeneration++;
    
    printf("Evolution complete! Generation %d ready.\n", trainer->currentGeneration);
}

// ===================================================================
// PROGRESS TRACKING AND DISPLAY
// ===================================================================
void displayGenerationSummary(EvolutionaryTrainer *trainer) {
    clock_t currentTime = clock();
    double generationTime = ((double)(currentTime - trainer->generationStartTime)) / CLOCKS_PER_SEC;
    double totalTime = ((double)(currentTime - trainer->startTime)) / CLOCKS_PER_SEC;
    
    printf("\n");
    printChar('=', 50);
    printf("\nGENERATION %d SUMMARY\n", trainer->currentGeneration);
    printChar('=', 50);
    printf("\n");
    
    // Fixed time display
    printf("Time: %.1fs (Total: %.1fm)\n", generationTime, totalTime / 60.0);
    
    Individual *best = &trainer->population[0];
    printf("Best AI: Fitness %.2f | Win Rate %.1f%% | Avg Credits %.0f\n",
           best->fitness, best->winRate * 100, best->avgCredits);
    printf("Strategy: %s | Games: %d | Profit: %.1f%%\n", 
           best->strategy, best->games, 
           ((best->avgCredits - STARTING_CREDITS) / STARTING_CREDITS) * 100);
      // Population diversity with better counting
    int strategyCounts[10] = {0}; // UPDATED: More strategy categories
    for (int i = 0; i < trainer->populationSize; i++) {
        if (strcmp(trainer->population[i].strategy, "Elite") == 0) strategyCounts[0]++;
        else if (strcmp(trainer->population[i].strategy, "Dominant") == 0) strategyCounts[1]++;
        else if (strcmp(trainer->population[i].strategy, "Aggressive") == 0) strategyCounts[2]++;
        else if (strcmp(trainer->population[i].strategy, "Conservative") == 0) strategyCounts[3]++;
        else if (strcmp(trainer->population[i].strategy, "Survivor") == 0) strategyCounts[4]++;
        else if (strcmp(trainer->population[i].strategy, "Passive") == 0) strategyCounts[5]++;
        else if (strcmp(trainer->population[i].strategy, "Lucky") == 0) strategyCounts[6]++;
        else if (strcmp(trainer->population[i].strategy, "Struggling") == 0) strategyCounts[7]++;
        else if (strcmp(trainer->population[i].strategy, "Weak") == 0) strategyCounts[8]++;
        else strategyCounts[9]++;
    }
    
    printf("Strategies: Elite:%d Dom:%d Agg:%d Con:%d Sur:%d Pass:%d Lucky:%d Str:%d Weak:%d Other:%d\n", 
           strategyCounts[0], strategyCounts[1], strategyCounts[2], strategyCounts[3], 
           strategyCounts[4], strategyCounts[5], strategyCounts[6], strategyCounts[7],
           strategyCounts[8], strategyCounts[9]);
    
    // Progress estimate
    double avgTimePerGeneration = totalTime / (trainer->currentGeneration + 1);
    double estimatedRemaining = avgTimePerGeneration * (trainer->maxGenerations - trainer->currentGeneration - 1);
    printf("Progress: %d/%d (%.1f%%) | ETA: %.1fm\n",
           trainer->currentGeneration + 1, trainer->maxGenerations,
           ((trainer->currentGeneration + 1) * 100.0) / trainer->maxGenerations,
           estimatedRemaining / 60.0);
    
    printChar('=', 50);
    printf("\n");
    
    // Log to file
    if (trainer->evolutionLog) {
        for (int i = 0; i < 5; i++) {
            Individual *ai = &trainer->population[i];
            fprintf(trainer->evolutionLog, "%d,%d,%.4f,%.4f,%.2f,%s,%d,%d\n",
                    trainer->currentGeneration, i, ai->fitness, ai->winRate,
                    ai->avgCredits, ai->strategy, ai->parentA, ai->parentB);
        }
        fflush(trainer->evolutionLog);
    }
}

void displayEvolutionResults(EvolutionaryTrainer *trainer) {
    clock_t endTime = clock();
    double totalTime = ((double)(endTime - trainer->startTime)) / CLOCKS_PER_SEC;
    
    printf("\n");
    printChar('*', 60);
    printf("\nEVOLUTIONARY TRAINING COMPLETE!\n");
    printChar('*', 60);
    printf("\n");
    
    printf("Summary:\n");
    printf("  Generations: %d\n", trainer->maxGenerations);
    printf("  Population: %d AIs\n", trainer->populationSize);
    printf("  Training Time: %.2f hours\n", totalTime / 3600.0);
    printf("  Total Games: ~%d\n", 
           trainer->populationSize * trainer->gamesPerGeneration * trainer->maxGenerations);
    
    Individual *champion = &trainer->population[0];
    printf("\nChampion AI:\n");
    printf("  Fitness: %.2f\n", champion->fitness);
    printf("  Win Rate: %.1f%% (%d/%d)\n", 
           champion->winRate * 100, champion->wins, champion->games);
    printf("  Avg Credits: %.0f\n", champion->avgCredits);
    printf("  Profit Margin: %.1f%%\n", 
           ((champion->avgCredits - STARTING_CREDITS) / STARTING_CREDITS) * 100);
    printf("  Strategy: %s\n", champion->strategy);
    printf("  Generation: %d\n", champion->generation);
      printf("\nFinal Population Strategy Distribution:\n");
    int strategyCounts[10] = {0};  // INCREASED: More strategy categories
    for (int i = 0; i < trainer->populationSize; i++) {
        if (strcmp(trainer->population[i].strategy, "Elite") == 0) strategyCounts[0]++;
        else if (strcmp(trainer->population[i].strategy, "Dominant") == 0) strategyCounts[1]++;
        else if (strcmp(trainer->population[i].strategy, "Aggressive") == 0) strategyCounts[2]++;
        else if (strcmp(trainer->population[i].strategy, "Conservative") == 0) strategyCounts[3]++;
        else if (strcmp(trainer->population[i].strategy, "Survivor") == 0) strategyCounts[4]++;
        else if (strcmp(trainer->population[i].strategy, "Passive") == 0) strategyCounts[5]++;
        else if (strcmp(trainer->population[i].strategy, "Lucky") == 0) strategyCounts[6]++;
        else if (strcmp(trainer->population[i].strategy, "Struggling") == 0) strategyCounts[7]++;
        else if (strcmp(trainer->population[i].strategy, "Weak") == 0) strategyCounts[8]++;
        else strategyCounts[9]++;  // Other/Unknown
    }
    
    printf("  Elite: %d (%.1f%%)\n", strategyCounts[0], (strategyCounts[0] * 100.0) / trainer->populationSize);
    printf("  Dominant: %d (%.1f%%)\n", strategyCounts[1], (strategyCounts[1] * 100.0) / trainer->populationSize);
    printf("  Aggressive: %d (%.1f%%)\n", strategyCounts[2], (strategyCounts[2] * 100.0) / trainer->populationSize);
    printf("  Conservative: %d (%.1f%%)\n", strategyCounts[3], (strategyCounts[3] * 100.0) / trainer->populationSize);
    printf("  Survivor: %d (%.1f%%)\n", strategyCounts[4], (strategyCounts[4] * 100.0) / trainer->populationSize);
    printf("  Passive: %d (%.1f%%)\n", strategyCounts[5], (strategyCounts[5] * 100.0) / trainer->populationSize);
    printf("  Lucky: %d (%.1f%%)\n", strategyCounts[6], (strategyCounts[6] * 100.0) / trainer->populationSize);
    printf("  Struggling: %d (%.1f%%)\n", strategyCounts[7], (strategyCounts[7] * 100.0) / trainer->populationSize);
    printf("  Weak: %d (%.1f%%)\n", strategyCounts[8], (strategyCounts[8] * 100.0) / trainer->populationSize);
    printf("  Other: %d (%.1f%%)\n", strategyCounts[9], (strategyCounts[9] * 100.0) / trainer->populationSize);
    
    // Fixed Hall of Fame display (show current best performers, not broken references)
    printf("\nHall of Fame (Current Top 5):\n");
    for (int i = 0; i < 5 && i < trainer->populationSize; i++) {
        Individual *top = &trainer->population[i]; // Current sorted population
        printf("  #%d: Fitness %.2f | Win Rate %.1f%% | Credits %.0f | %s\n",
               i + 1, top->fitness, top->winRate * 100, top->avgCredits, top->strategy);
    }
    
    // Show historical best if different
    if (trainer->bestIndividualEver >= 0 && trainer->bestFitnessEver > champion->fitness) {
        printf("\nHistorical Best (Generation %d):\n", 
               trainer->population[trainer->bestIndividualEver].generation);
        printf("  Fitness: %.2f (better than current champion)\n", trainer->bestFitnessEver);
    }
    
    printf("\nFiles saved:\n");
    printf("  - evolution_log.csv (detailed log)\n");
    printf("  - poker_ai_evolved_champion.dat (best AI)\n");
    printf("  - evolved_champion_0-9.dat (top 10 AIs)\n");
    
    // Performance analysis
    double avgProfit = 0;
    int profitableAIs = 0;
    for (int i = 0; i < trainer->populationSize; i++) {
        double profit = trainer->population[i].avgCredits - STARTING_CREDITS;
        avgProfit += profit;
        if (profit > 0) profitableAIs++;
    }
    avgProfit /= trainer->populationSize;
    
    printf("\nPerformance Analysis:\n");
    printf("  Profitable AIs: %d/%d (%.1f%%)\n", 
           profitableAIs, trainer->populationSize, (profitableAIs * 100.0) / trainer->populationSize);
    printf("  Average Profit: %.1f credits (%.1f%%)\n", 
           avgProfit, (avgProfit / STARTING_CREDITS) * 100);
    
    if (strategyCounts[3] > trainer->populationSize * 0.7) {
        printf("\nNOTE: High percentage of 'Survivor' strategies suggests overly conservative play.\n");
        printf("Consider adjusting fitness function or increasing training duration.\n");
    }
    
    if (strategyCounts[0] + strategyCounts[1] < trainer->populationSize * 0.1) {
        printf("\nNOTE: Few Dominant/Aggressive strategies emerged.\n");
        printf("AIs may need more training to develop advanced strategies.\n");
    }
    
    printChar('*', 60);
    printf("\n");
}

// ===================================================================
// MAIN EVOLUTIONARY TRAINING FUNCTION
// ===================================================================

void trainEvolutionaryAI(int populationSize, int maxGenerations) {
    printf("\n");
    printChar('=', 60);
    printf("\nEVOLUTIONARY AI TRAINING\n");
    printf("Population-based optimization with natural selection\n");
    printChar('=', 60);
    printf("\n");
    
    // Run self-play test first
    printf("Running pre-training diagnostics...\n");
    debugSelfPlayGame();
    
    EvolutionaryTrainer *trainer = createEvolutionaryTrainer(populationSize, maxGenerations);
    if (!trainer) {
        printf("Failed to create evolutionary trainer!\n");
        return;
    }
    
    trainer->startTime = clock();
    initializeOpponentProfiles(8);
    initializePopulation(trainer);
    
    printf("Starting evolution: %d AIs over %d generations\n", populationSize, maxGenerations);
    printf("Games per generation: %d (reduced for debugging)\n", trainer->gamesPerGeneration);
    
    // Main evolution loop with diagnostics
    for (trainer->currentGeneration = 0; trainer->currentGeneration < trainer->maxGenerations; trainer->currentGeneration++) {
        printf("\n>>> GENERATION %d/%d <<<\n", 
               trainer->currentGeneration + 1, trainer->maxGenerations);
        
        clock_t genStart = clock();
        runTournamentGenerationDebug(trainer);
        clock_t genEnd = clock();
        
        double genTime = ((double)(genEnd - genStart)) / CLOCKS_PER_SEC;
        printf("Tournament time: %.2f seconds\n", genTime);
        
        evaluatePopulationFitness(trainer);
        rankPopulation(trainer);
        
        // Run diagnostics after first generation
        if (trainer->currentGeneration == 0) {
            diagnoseTrainingIssues(trainer);
        }
        
        displayGenerationSummary(trainer);
        
        if (trainer->currentGeneration % 5 == 0) {
            saveEvolutionCheckpoint(trainer);
        }
        
        if (trainer->currentGeneration < trainer->maxGenerations - 1) {
            evolvePopulation(trainer);
        }
    }
    
    displayEvolutionResults(trainer);
    
    // Save the champion and top 10
    saveNetwork(trainer->population[0].network, "poker_ai_evolved_champion.dat");
    for (int i = 0; i < 10; i++) {
        char filename[100];
        sprintf(filename, "evolved_champion_%d.dat", i);
        saveNetwork(trainer->population[i].network, filename);
    }
    
    freeEvolutionaryTrainer(trainer);
    printf("Evolutionary training complete!\n");
}

void saveEvolutionCheckpoint(EvolutionaryTrainer *trainer) {
    // Save current best AI as checkpoint
    char filename[100];
    sprintf(filename, "evolution_checkpoint_gen_%d.dat", trainer->currentGeneration);
    saveNetwork(trainer->population[0].network, filename);
    
    printf("Checkpoint saved: %s\n", filename);
}


// Diagnostic function to check training health
void diagnoseTrainingIssues(EvolutionaryTrainer *trainer) {
    printf("\n=== TRAINING DIAGNOSTICS ===\n");
    
    // Check game counts
    int totalGames = 0;
    int aisWith10Plus = 0;
    int aisWith1Game = 0;
    double totalCredits = 0;
    
    for (int i = 0; i < trainer->populationSize; i++) {
        Individual *ai = &trainer->population[i];
        totalGames += ai->games;
        totalCredits += ai->totalCredits;
        
        if (ai->games >= 10) aisWith10Plus++;
        if (ai->games == 1) aisWith1Game++;
    }
    
    printf("Total games played: %d\n", totalGames);
    printf("Average games per AI: %.2f (target: %d)\n", 
           (double)totalGames / trainer->populationSize, trainer->gamesPerGeneration);
    printf("AIs with 10+ games: %d/%d (%.1f%%)\n", 
           aisWith10Plus, trainer->populationSize, (aisWith10Plus * 100.0) / trainer->populationSize);
    printf("AIs with only 1 game: %d/%d (%.1f%%)\n", 
           aisWith1Game, trainer->populationSize, (aisWith1Game * 100.0) / trainer->populationSize);
    printf("Average total credits: %.2f\n", totalCredits / trainer->populationSize);
    
    // Check for impossible values
    int impossibleWinRates = 0;
    int impossibleCredits = 0;
    
    for (int i = 0; i < trainer->populationSize; i++) {
        Individual *ai = &trainer->population[i];
        if (ai->winRate > 1.0 || (ai->winRate == 1.0 && ai->games > 1)) {
            impossibleWinRates++;
        }
        if (ai->avgCredits < 0 || ai->avgCredits > STARTING_CREDITS * 5) {
            impossibleCredits++;
        }
    }
    
    printf("Impossible win rates: %d\n", impossibleWinRates);
    printf("Impossible credit values: %d\n", impossibleCredits);
    
    // Check tournament system
    printf("Tournament tables: %d\n", trainer->numTables);
    printf("Players per table: 4\n");
    printf("Expected table games per round: %d\n", trainer->numTables);
    
    printf("=== END DIAGNOSTICS ===\n\n");
}
