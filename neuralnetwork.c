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
// ENHANCED TRAINING DATA GENERATION
// ===================================================================

// Enhanced training data generation
void generateEnhancedTrainingData(double **inputs, double **outputs, int numSamples) {
    
    printf("Generating training data with %d samples...\n", numSamples);
    
    for (int i = 0; i < numSamples; i++) {
        // Create game states
        double handStrength = (double)rand() / RAND_MAX;
        double handPotential = (double)rand() / RAND_MAX;
        double boardTexture = (double)rand() / RAND_MAX;
        double potOdds = (double)rand() / RAND_MAX;
        double position = (double)rand() / RAND_MAX;
        double stackRatio = (double)rand() / RAND_MAX * 5.0;
        
        // Build enhanced input vector
        inputs[i][0] = handStrength;
        inputs[i][1] = handPotential;
        inputs[i][2] = boardTexture;
        inputs[i][3] = potOdds;
        inputs[i][4] = stackRatio;
        inputs[i][5] = position;
        
        // Fill remaining inputs
        for (int j = 6; j < INPUT_SIZE; j++) {
            inputs[i][j] = (double)rand() / RAND_MAX;
        }
        
        // Enhanced strategy logic
        double fold = 0.0, call = 0.0, raise = 0.0;
        
        double effectiveStrength = handStrength + (handPotential * 0.3);
        
        // Position adjustment
        if (position > 0.7) effectiveStrength += 0.1;
        
        // Pot odds consideration
        if (potOdds < 0.3 && effectiveStrength > 0.4) {
            effectiveStrength += 0.1;
        }
        
        // Board texture adjustment
        if (boardTexture > 0.7 && effectiveStrength < 0.6) {
            effectiveStrength -= 0.1;
        }
        
        // Strategy based on effective strength
        if (effectiveStrength < 0.2) {
            fold = 0.8; call = 0.15; raise = 0.05;
        } else if (effectiveStrength < 0.4) {
            fold = 0.5; call = 0.4; raise = 0.1;
        } else if (effectiveStrength < 0.6) {
            fold = 0.2; call = 0.6; raise = 0.2;
        } else if (effectiveStrength < 0.8) {
            fold = 0.05; call = 0.3; raise = 0.65;
        } else {
            fold = 0.02; call = 0.18; raise = 0.8;
        }
        
        // Add controlled randomness
        double noise = ((double)rand() / RAND_MAX - 0.5) * 0.2;
        fold += noise; call += noise; raise += noise;
        
        // Normalize to valid probabilities
        double total = fold + call + raise;
        if (total <= 0) total = 1.0;
        
        outputs[i][0] = fmax(0.01, fold / total);
        outputs[i][1] = fmax(0.01, call / total);
        outputs[i][2] = fmax(0.01, raise / total);
    }
    
    printf("Enhanced training data generated successfully.\n");
}

// Enhanced basic AI training with monitoring
void trainEnhancedBasicAIWithMonitoring() {
    printf("\n");
    printRepeatedChar('=', 60);
    printf("\n");
    printf("ENHANCED AI TRAINING WITH FULL MONITORING\n");
    printRepeatedChar('=', 60);
    printf("\n");
    
    // Create larger, more diverse training dataset
    int numSamples = 3000;  // Increased for better learning
    double **inputs = malloc(numSamples * sizeof(double*));
    double **outputs = malloc(numSamples * sizeof(double*));
    
    printf("Allocating memory for %d training samples...\n", numSamples);
    
    for (int i = 0; i < numSamples; i++) {
        inputs[i] = malloc(INPUT_SIZE * sizeof(double));
        outputs[i] = malloc(OUTPUT_SIZE * sizeof(double));
        if (!inputs[i] || !outputs[i]) {
            printf("Error: Memory allocation failed at sample %d\n", i);
            return;
        }
    }
    
    printf("Generating enhanced training data...\n");
    generateEnhancedTrainingData(inputs, outputs, numSamples);
    
    printf("Creating neural network...\n");
    NeuralNetwork *nn = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    if (!nn) {
        printf("Error: Failed to create neural network\n");
        return;
    }
    
    // Set optimal learning parameters
    nn->learningRate = 0.01;  // Conservative learning rate for stability
    
    printf("Starting training with comprehensive monitoring...\n\n");
    
    // Train with full monitoring
    trainWithMonitoring(nn, inputs, outputs, numSamples, TRAINING_EPOCHS);
    
    printf("\nTesting trained network on sample predictions...\n");
    printRepeatedChar('=', 60);
    printf("\n");
    
    // Test the network on several examples
    for (int i = 0; i < 10; i += 2) {  // Test every other sample
        forwardpropagate(nn, inputs[i]);
        
        printf("Test %d:\n", i/2 + 1);
        printf("  Input features: Hand=%.2f, Potential=%.2f, Texture=%.2f, PotOdds=%.2f\n",
               inputs[i][0], inputs[i][1], inputs[i][2], inputs[i][3]);
        printf("  Expected: Fold=%.2f Call=%.2f Raise=%.2f\n",
               outputs[i][0], outputs[i][1], outputs[i][2]);
        printf("  Predicted: Fold=%.2f Call=%.2f Raise=%.2f\n",
               nn->outputLayer[0].value, nn->outputLayer[1].value, nn->outputLayer[2].value);
        
        // Determine predicted action
        int predictedAction = 0;
        double maxProb = nn->outputLayer[0].value;
        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (nn->outputLayer[j].value > maxProb) {
                maxProb = nn->outputLayer[j].value;
                predictedAction = j;
            }
        }
        
        int expectedAction = 0;
        double maxExpected = outputs[i][0];
        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (outputs[i][j] > maxExpected) {
                maxExpected = outputs[i][j];
                expectedAction = j;
            }
        }
        
        printf("  Decision: %s (expected: %s) %s\n",
               predictedAction == 0 ? "FOLD" : (predictedAction == 1 ? "CALL" : "RAISE"),
               expectedAction == 0 ? "FOLD" : (expectedAction == 1 ? "CALL" : "RAISE"),
               predictedAction == expectedAction ? "✓ CORRECT" : "✗ WRONG");
        printf("\n");
    }
    
    printf("Saving enhanced AI model...\n");
    saveNetwork(nn, "poker_ai_enhanced_monitored.dat");
    
    printf("\n");
    printRepeatedChar('=', 60);
    printf("\n");
    printf("ENHANCED AI TRAINING COMPLETE!\n");
    printf("Model saved as: poker_ai_enhanced_monitored.dat\n");
    printf("Training log saved as: training_log.csv\n");
    printRepeatedChar('=', 60);
    printf("\n");
    
    // Cleanup
    for (int i = 0; i < numSamples; i++) {
        free(inputs[i]);
        free(outputs[i]);
    }
    free(inputs);
    free(outputs);
    freeNetwork(nn);
}

// Enhanced self-play training with monitoring
void enhancedSelfPlayTrainingWithMonitoring(int numGames, int numPlayers) {
    printf("\n");
    printRepeatedChar('=', 70);
    printf("\n");
    printf("ENHANCED SELF-PLAY TRAINING WITH MONITORING\n");
    printRepeatedChar('=', 70);
    printf("\n");
    printf("Games: %d | Players: %d | Enhanced Features: ON\n", numGames, numPlayers);
    printRepeatedChar('=', 70);
    printf("\n\n");
    
    // Initialize components
    initializeOpponentProfiles(numPlayers);
    
    NeuralNetwork **networks = malloc(numPlayers * sizeof(NeuralNetwork*));
    for (int i = 0; i < numPlayers; i++) {
        networks[i] = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
        addNoiseToWeights(networks[i], 0.1);  // Diversify initial strategies
    }
    
    ReplayBuffer *rb = createReplayBuffer(100000);
    
    // Training tracking
    int wins[MAXPLAYERS] = {0};
    double avgCredits[MAXPLAYERS] = {0};
    double winRateHistory[1000];  // Track win rate evolution
    int progressCheckpoints = 20;
    int gamesPerCheckpoint = numGames / progressCheckpoints;
    
    FILE *progressFile = fopen("selfplay_progress.csv", "w");
    if (progressFile) {
        fprintf(progressFile, "Game,AI_0_WinRate,AI_1_WinRate,AI_2_WinRate,AI_3_WinRate,AvgCredits,ExperienceBufferSize\n");
    }
    
    clock_t startTime = clock();
    
    printf("Starting self-play training with detailed monitoring...\n\n");
    
    // Training loop with enhanced monitoring
    for (int game = 0; game < numGames; game++) {
        GameRecord record = playEnhancedSelfPlayGame(networks, numPlayers, rb);
        
        // Update statistics
        wins[record.winner]++;
        for (int i = 0; i < numPlayers; i++) {
            avgCredits[i] = (avgCredits[i] * game + record.finalCredits[i]) / (game + 1);
        }
        
        // Enhanced training frequency
        if (rb->size > 500 && game % 3 == 0) {  // Train every 3 games
            for (int i = 0; i < numPlayers; i++) {
                trainFromExperience(networks[i], rb, 128);  // Larger batches
            }
        }
        
        // Detailed progress monitoring
        if (game > 0 && (game % gamesPerCheckpoint == 0 || game == numGames - 1)) {
            clock_t currentTime = clock();
            double elapsed = ((double)(currentTime - startTime)) / CLOCKS_PER_SEC;
            
            printf("\n");
            printRepeatedChar('-', 50);
            printf("\n");
            printf("PROGRESS CHECKPOINT: Game %d/%d (%.1f%%)\n", game, numGames, (game * 100.0) / numGames);
            printRepeatedChar('-', 50);
            printf("\n");
            printf("Time elapsed: %.1f seconds (%.2f games/sec)\n", elapsed, game / elapsed);
            printf("Experience buffer size: %d\n", rb->size);
            
            printf("\nCurrent win rates:\n");
            for (int i = 0; i < numPlayers; i++) {
                double winRate = (game > 0) ? (wins[i] * 100.0) / game : 0.0;
                printf("  AI_%d: %.1f%% (%d wins) | Avg Credits: %.0f\n", 
                       i, winRate, wins[i], avgCredits[i]);
            }
            
            // Calculate strategy diversity (how different are the AIs?)
            double diversityScore = 0.0;
            for (int i = 0; i < numPlayers; i++) {
                for (int j = i + 1; j < numPlayers; j++) {
                    double winRateDiff = fabs((double)wins[i] - wins[j]) / game;
                    diversityScore += winRateDiff;
                }
            }
            diversityScore = diversityScore / ((numPlayers * (numPlayers - 1)) / 2);
            
            printf("Strategy diversity: %.3f (higher = more varied strategies)\n", diversityScore);
            
            // Log to file
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
            
            // Early stopping if one AI becomes too dominant
            for (int i = 0; i < numPlayers; i++) {
                double winRate = (double)wins[i] / game;
                if (winRate > 0.8 && game > numGames / 4) {
                    printf("\nEarly stopping: AI_%d became too dominant (%.1f%% win rate)\n", 
                           i, winRate * 100);
                    goto training_complete;
                }
            }
        }
        
        // Simple progress indicator for intermediate games
        if (game % (numGames / 100) == 0 && game % gamesPerCheckpoint != 0) {
            printf(".");
            fflush(stdout);
        }
    }
    
    training_complete:
    
    clock_t endTime = clock();
    double totalTime = ((double)(endTime - startTime)) / CLOCKS_PER_SEC;
    
    printf("\n\n");
    printRepeatedChar('=', 70);
    printf("\n");
    printf("SELF-PLAY TRAINING COMPLETE!\n");
    printRepeatedChar('=', 70);
    printf("\n");
    printf("Total time: %.2f seconds (%.2f minutes)\n", totalTime, totalTime / 60.0);
    printf("Games played: %d\n", numGames);
    printf("Experience samples: %d\n", rb->size);
    printf("Training efficiency: %.2f games/second\n", numGames / totalTime);
    
    printf("\nFINAL RESULTS:\n");
    int bestAI = 0;
    int maxWins = wins[0];
    
    for (int i = 0; i < numPlayers; i++) {
        double winRate = (double)wins[i] / numGames;
        printf("AI_%d: %.1f%% (%d wins) | Final Avg Credits: %.0f\n", 
               i, winRate * 100, wins[i], avgCredits[i]);
        
        if (wins[i] > maxWins) {
            maxWins = wins[i];
            bestAI = i;
        }
    }
    
    printf("\nBest performing AI: AI_%d\n", bestAI);
    printf("Saving best network as: poker_ai_selfplay_monitored.dat\n");
    
    saveNetwork(networks[bestAI], "poker_ai_selfplay_monitored.dat");
    
    // Save all networks as backups
    for (int i = 0; i < numPlayers; i++) {
        char filename[100];
        sprintf(filename, "selfplay_ai_%d.dat", i);
        saveNetwork(networks[i], filename);
    }
    
    printf("All networks saved. Training logs: selfplay_progress.csv\n");
    printRepeatedChar('=', 70);
    printf("\n");
    
    // Cleanup
    if (progressFile) fclose(progressFile);
    for (int i = 0; i < numPlayers; i++) {
        freeNetwork(networks[i]);
    }
    free(networks);
    free(rb->buffer);
    free(rb);
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

// Display training progress
void displayTrainingProgress(TrainingStats *stats, bool verbose) {
    if (!stats || stats->currentEpoch == 0) return;
    
    int epoch = stats->currentEpoch - 1;  // Last completed epoch
    double currentLoss = stats->lossHistory[epoch];
    
    // Calculate elapsed time
    clock_t currentTime = clock();
    double elapsedSeconds = ((double)(currentTime - stats->startTime)) / CLOCKS_PER_SEC;
    
    // Show progress every 100 epochs or on last epoch
    if (epoch % 100 == 0 || epoch == stats->maxEpochs - 1) {
        printf("Epoch %4d/%d | Loss: %.6f | LR: %.6f", 
               epoch, stats->maxEpochs - 1, currentLoss, 0.01);  // Assuming default LR
        
        // Show improvement indicators
        if (epoch > 0) {
            double lossChange = currentLoss - stats->lossHistory[epoch - 1];
            if (lossChange < -0.001) {
                printf(" ↓");  // Loss decreased significantly
            } else if (lossChange > 0.001) {
                printf(" ↑");  // Loss increased significantly
            } else {
                printf(" →");  // Loss roughly stable
            }
        }
        
        // Show if this is the best so far
        if (epoch == stats->bestEpoch) {
            printf(" ★ BEST");
        }
        
        // Show overfitting warning
        if (stats->overfittingDetected) {
            printf(" ⚠ OVERFITTING");
        }
        
        printf("\n");
    }
    
    // Progress bar every 10% of training
    if (epoch % (stats->maxEpochs / 10) == 0 && !verbose) {
        int progress = (epoch * 50) / stats->maxEpochs;  // 50 chars wide
        printf("Progress: [");
        for (int i = 0; i < 50; i++) {
            printf(i < progress ? "█" : "░");
        }
        printf("] %d%% (Loss: %.4f)\n", (epoch * 100) / stats->maxEpochs, currentLoss);
    }
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


// Display final training summary
void displayTrainingSummary(TrainingStats *stats, NeuralNetwork *nn, double **inputs, int numSamples) {
    if (!stats) return;
    
    clock_t endTime = clock();
    double totalTime = ((double)(endTime - stats->startTime)) / CLOCKS_PER_SEC;
    
    printf("\n");
    printRepeatedChar('=', 70);
    printf("\n");
    printf("TRAINING COMPLETED!\n");
    printf("\n");
    printRepeatedChar('=', 70);
    printf("\n");
    
    // Basic training info
    printf("TRAINING SUMMARY:\n");
    printf("Total Epochs:     %d\n", stats->currentEpoch);
    printf("Total Time:       %.2f seconds (%.2f minutes)\n", totalTime, totalTime / 60.0);
    printf("Time per Epoch:   %.4f seconds\n", totalTime / stats->currentEpoch);
    
    // Loss progression analysis
    printf("\nLOSS PROGRESSION:\n");
    printf("Initial Loss:     %.6f\n", stats->initialLoss);
    printf("Final Loss:       %.6f\n", stats->lossHistory[stats->currentEpoch - 1]);
    printf("Best Loss:        %.6f (Epoch %d)\n", stats->bestLoss, stats->bestEpoch);
    printf("Loss Improvement: %.6f (%.1f%% reduction)\n", 
           stats->initialLoss - stats->bestLoss,
           ((stats->initialLoss - stats->bestLoss) / stats->initialLoss) * 100);
    
    // Overfitting detection results
    printf("\nOVERFITTING ANALYSIS:\n");
    if (stats->overfittingDetected) {
        printf("⚠ OVERFITTING DETECTED at epoch %d\n", stats->bestEpoch + stats->stagnationCount);
        printf("  Recent loss average: %.6f vs Best loss: %.6f\n", 
               stats->recentLossAverage, stats->bestLoss);
        printf("  Stagnation period: %d epochs\n", stats->stagnationCount);
    } else if (stats->stagnationCount > 50) {
        printf("→ TRAINING PLATEAUED after epoch %d (%d epochs without improvement)\n", 
               stats->bestEpoch, stats->stagnationCount);
    } else {
        printf("✓ NO OVERFITTING DETECTED - Training progressed normally\n");
    }
    
    // Comparative analysis (before/after)
    printf("\nCOMPARATIVE ANALYSIS:\n");
    double improvementRatio = stats->initialLoss / stats->bestLoss;
    if (improvementRatio > 5.0) {
        printf("✓ EXCELLENT: %.1fx improvement - Strong learning occurred\n", improvementRatio);
    } else if (improvementRatio > 2.0) {
        printf("✓ GOOD: %.1fx improvement - Solid learning progress\n", improvementRatio);
    } else if (improvementRatio > 1.2) {
        printf("→ MODERATE: %.1fx improvement - Some learning occurred\n", improvementRatio);
    } else {
        printf("⚠ POOR: %.1fx improvement - Limited learning\n", improvementRatio);
    }
    
    // Network confidence and decision distribution
    if (nn && inputs) {
        analyzeNetworkConfidence(nn, inputs, fmin(numSamples, 1000));  // Analyze up to 1000 samples
    }
    
    printf("\nDetailed log saved to: training_log.csv\n");
    printRepeatedChar('=', 70);
    printf("\n");
}

// Enhanced training function with monitoring
void trainWithMonitoring(NeuralNetwork *nn, double **trainingInputs, double **trainingOutputs, 
                        int numSamples, int epochs) {
    printf("Starting enhanced training with monitoring...\n");
    printf("Samples: %d | Epochs: %d | Learning Rate: %.4f\n", 
           numSamples, epochs, nn->learningRate);
    
    // Initialize training statistics
    TrainingStats *stats = initializeTrainingStats(epochs);
    if (!stats) {
        printf("Error: Could not initialize training monitoring\n");
        return;
    }
    
    // Training loop with enhanced monitoring
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Train on all samples
        for (int sample = 0; sample < numSamples; sample++) {
            forwardpropagate(nn, trainingInputs[sample]);
            backpropagate(nn, trainingOutputs[sample]);
            updateWeights(nn);
        }
        
        // Update statistics and display progress
        updateTrainingStats(stats, nn, trainingInputs, trainingOutputs, numSamples, nn->learningRate);
        
        // Display progress every 100 epochs
        displayTrainingProgress(stats, false);
        
        // Early stopping if loss becomes very small
        if (stats->lossHistory[epoch] < 0.0001) {
            printf("\nEarly stopping: Loss became very small (%.6f)\n", stats->lossHistory[epoch]);
            break;
        }
        
        // Early stopping for overfitting
        if (stats->overfittingDetected) {
            printf("\nEarly stopping: Overfitting detected\n");
            break;
        }
        
        // Adaptive learning rate (optional)
        if (epoch > 100 && epoch % 200 == 0) {
            if (stats->stagnationCount > 150) {
                nn->learningRate *= 0.8;
                printf("Learning rate reduced to %.6f (learning stagnated)\n", nn->learningRate);
            }
        }
    }
    
    // Display enhanced final summary
    displayTrainingSummary(stats, nn, trainingInputs, numSamples);
    
    // Cleanup
    if (stats->logFile) {
        fclose(stats->logFile);
    }
    free(stats->lossHistory);
    free(stats->epochNumbers);
    free(stats);
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
        GameRecord record = playEnhancedSelfPlayGame(networks, numPlayers, rb);
        
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
// Modify the trainTwoPhaseAI function to use tournament evolution
void trainTwoPhaseAI(int numGames, int numPlayers) {
    printf("\n");
    printRepeatedChar('*', 70);
    printf("\n");
    printf("TWO-PHASE AI TRAINING\n");
    printf("Phase 1: Minimal Bootstrap (basic rules)\n");
    printf("Phase 2: Tournament Evolution + Reinforcement Learning\n");
    printRepeatedChar('*', 70);
    printf("\n");
    
    // Phase 1: Quick bootstrap
    NeuralNetwork *bootstrap = trainMinimalBootstrap();
    if (!bootstrap) {
        printf("Bootstrap training failed!\n");
        return;
    }
    
    printf("\nBootstrap complete! Press Enter to begin tournament evolution...");
    getchar();
    
    // Free bootstrap network (it's saved to file)
    freeNetwork(bootstrap);
    
    // Phase 2: Tournament evolution (instead of pure self-play)
    int maxGenerations = 25; // Your failsafe limit
    tournamentEvolution(maxGenerations);
    
    printf("\n");
    printRepeatedChar('*', 70);
    printf("\n");
    printf("TWO-PHASE TRAINING COMPLETE!\n");
    printf("Your AI has learned through tournament competition AND reinforcement learning!\n");
    printf("Files created:\n");
    printf("  - poker_ai_bootstrap.dat (Phase 1 result)\n");
    printf("  - poker_ai_evolved.dat (Final tournament + RL champion)\n");
    printf("  - poker_ai_previous1.dat (Previous generation champion)\n");
    printf("  - poker_ai_previous2.dat (2 generations ago champion)\n");
    printf("  - tournament_evolution.csv (Training progress log)\n");
    printRepeatedChar('*', 70);
    printf("\n");
}

// Main tournament evolution function - replaces pureReinforcementLearning
void tournamentEvolution(int maxGenerations) {
    printf("\n");
    printRepeatedChar('=', 70);
    printf("\n");
    printf("TOURNAMENT EVOLUTION SYSTEM\n");
    printf("Population: 6 AIs | Games per Generation: 50 | Max Generations: %d\n", maxGenerations);
    printf("Stop Condition: Diversity < 0.05 for 3 generations\n");
    printRepeatedChar('=', 70);
    printf("\n");
    
    // Initialize tournament state
    TournamentState *tournament = initializeTournament(6, 50, maxGenerations);
    if (!tournament) {
        printf("Error: Failed to initialize tournament system\n");
        return;
    }
    
    printf("Tournament system initialized successfully.\n");
    printf("Starting evolution...\n\n");
    
    // Evolution loop
    bool converged = false;
    while (tournament->generation < tournament->maxGenerations && !converged) {
        printf("GENERATION %d/%d\n", tournament->generation + 1, tournament->maxGenerations);
        printRepeatedChar('-', 40);
        printf("\n");
        
        // Run one generation (50 games)
        runGeneration(tournament);
        
        // Check convergence (but not before generation 5)
        double diversity = calculatePopulationDiversity(tournament->population, tournament->populationSize);
        tournament->diversityHistory[tournament->generation] = diversity;
        
        printf("Generation diversity: %.4f\n", diversity);
        
        if (tournament->generation >= 5 && diversity < tournament->diversityThreshold) {
            tournament->lowDiversityCount++;
            printf("Low diversity detected: %.4f (count: %d/3)\n", diversity, tournament->lowDiversityCount);
        } else {
            tournament->lowDiversityCount = 0;
        }
        
        if (tournament->lowDiversityCount >= 3) {
            converged = true;
            printf("Population converged - stopping evolution.\n");
        }
        
        // Evolve population for next generation (unless converged or last generation)
        if (!converged && tournament->generation < tournament->maxGenerations - 1) {
            evolvePopulation(tournament);
        }
        
        tournament->generation++;
        printf("\n");
    }
    
    // Final results
    clock_t endTime = clock();
    double totalTime = ((double)(endTime - tournament->startTime)) / CLOCKS_PER_SEC;
    
    printf("\n");
    printRepeatedChar('=', 70);
    printf("\n");
    printf("TOURNAMENT EVOLUTION COMPLETE!\n");
    printRepeatedChar('=', 70);
    printf("\n");
    printf("Generations completed: %d\n", tournament->generation);
    printf("Total time: %.2f seconds (%.2f minutes)\n", totalTime, totalTime / 60.0);
    printf("Total games played: %d\n", tournament->generation * tournament->gamesPerGeneration);
    printf("Final champion fitness: %.4f\n", tournament->championFitness);
    
    if (converged) {
        printf("Reason: Population converged (diversity < %.3f)\n", tournament->diversityThreshold);
    } else {
        printf("Reason: Maximum generations reached\n");
    }
    
    // Save final champions
    saveChampions(tournament);
    
    printf("\nFiles saved:\n");
    printf("- poker_ai_evolved.dat (final champion)\n");
    printf("- poker_ai_previous1.dat (previous generation champion)\n");
    printf("- poker_ai_previous2.dat (2 generations ago champion)\n");
    printf("- tournament_evolution.csv (detailed log)\n");
    printRepeatedChar('=', 70);
    printf("\n");
    
    // Cleanup
    freeTournamentState(tournament);
}

// Initialize tournament state
TournamentState* initializeTournament(int populationSize, int gamesPerGeneration, int maxGenerations) {
    TournamentState *state = malloc(sizeof(TournamentState));
    if (!state) return NULL;
    
    // Initialize basic parameters
    state->generation = 0;
    state->populationSize = populationSize;
    state->gamesPerGeneration = gamesPerGeneration;
    state->maxGenerations = maxGenerations;
    state->championFitness = -1.0;
    state->lowDiversityCount = 0;
    state->diversityThreshold = 0.02;  // Lower threshold for convergence
    state->startTime = clock();
    
    // Allocate arrays
    state->population = malloc(populationSize * sizeof(NeuralNetwork*));
    state->fitness = malloc(populationSize * sizeof(double));
    state->wins = malloc(populationSize * sizeof(int));
    state->avgCredits = malloc(populationSize * sizeof(double));
    state->diversityHistory = malloc(maxGenerations * sizeof(double));
    
    // Allocate experience collection arrays
    state->maxExperiences = gamesPerGeneration * 10; // Up to 10 experiences per game per AI
    state->aiExperiences = malloc(populationSize * sizeof(Experience*));
    state->experienceCount = malloc(populationSize * sizeof(int));
    
    if (!state->population || !state->fitness || !state->wins || 
        !state->avgCredits || !state->diversityHistory || 
        !state->aiExperiences || !state->experienceCount) {
        printf("Error: Memory allocation failed\n");
        free(state);
        return NULL;
    }
    
    // Initialize experience buffers for each AI
    for (int i = 0; i < populationSize; i++) {
        state->aiExperiences[i] = malloc(state->maxExperiences * sizeof(Experience));
        state->experienceCount[i] = 0;
        if (!state->aiExperiences[i]) {
            printf("Error: Experience buffer allocation failed\n");
            free(state);
            return NULL;
        }
    }
    
    // Initialize champions as NULL
    state->currentChampion = NULL;
    state->previousChampion1 = NULL;
    state->previousChampion2 = NULL;
    
    // Create initial population from bootstrap
    NeuralNetwork *bootstrap = loadNetwork("poker_ai_bootstrap.dat");
    if (!bootstrap) {
        printf("No bootstrap found - creating random base network\n");
        bootstrap = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    }
    
    for (int i = 0; i < populationSize; i++) {
        state->population[i] = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
        if (bootstrap) {
            copyNetworkWeights(bootstrap, state->population[i]);
        }
        
        // Add much more noise for initial diversity
        double noiseLevel;
        if (i == 0) {
            noiseLevel = 0.1;   // Light mutation
        } else if (i == 1) {
            noiseLevel = 0.2;   // Medium mutation
        } else if (i == 2) {
            noiseLevel = 0.3;   // Heavy mutation
        } else if (i == 3) {
            noiseLevel = 0.5;   // Very heavy mutation
        } else if (i == 4) {
            noiseLevel = 0.7;   // Extreme mutation
        } else {
            noiseLevel = 1.0;   // Nearly random network
        }
        
        mutateNetwork(state->population[i], noiseLevel);
        
        state->fitness[i] = 0.0;
        state->wins[i] = 0;
        state->avgCredits[i] = 0.0;
        
        printf("AI_%d created with %.1f mutation strength\n", i, noiseLevel);
    }
    
    if (bootstrap) freeNetwork(bootstrap);
    
    // Open log file
    state->logFile = fopen("tournament_evolution.csv", "w");
    if (state->logFile) {
        fprintf(state->logFile, "Generation,Winner_ID,Best_Fitness,Best_WinRate,Best_Credits,Avg_Fitness,Population_Diversity\n");
    }
    
    printf("Initial population created with %d AIs\n", populationSize);
    return state;
}

// Run one generation (50 games)
void runGeneration(TournamentState *state) {
    // Reset scores and experience counts
    for (int i = 0; i < state->populationSize; i++) {
        state->wins[i] = 0;
        state->avgCredits[i] = 0.0;
        state->fitness[i] = 0.0;
        state->experienceCount[i] = 0;  // Reset experience counter
    }
    
    // Initialize opponent profiles for this generation
    initializeOpponentProfiles(state->populationSize);
    
    printf("Playing %d games with %d AIs...\n", state->gamesPerGeneration, state->populationSize);
    
    // Play games and collect experiences
    for (int game = 0; game < state->gamesPerGeneration; game++) {
        int winner = playTournamentGame(state->population, state->populationSize, state, game);
        if (winner >= 0) {
            state->wins[winner]++;
        }
        
        // Show progress every 10 games
        if ((game + 1) % 10 == 0) {
            printf("Games completed: %d/%d\n", game + 1, state->gamesPerGeneration);
        }
    }
    
    printf("All games completed. Starting reinforcement learning phase...\n");
    
    // Perform reinforcement learning on collected experiences
    performReinforcementLearning(state);
    
    printf("Reinforcement learning complete. Calculating fitness...\n");
    
    // Calculate fitness scores
    calculateFitness(state);
    
    // Find generation winner and update champion
    int winner = 0;
    double bestFitness = state->fitness[0];
    for (int i = 1; i < state->populationSize; i++) {
        if (state->fitness[i] > bestFitness) {
            bestFitness = state->fitness[i];
            winner = i;
        }
    }
    
    printf("Generation results:\n");
    for (int i = 0; i < state->populationSize; i++) {
        printf("  AI_%d: %d wins (%.1f%%), %.0f credits, %.4f fitness%s\n", 
               i, state->wins[i], 
               (state->wins[i] * 100.0) / state->gamesPerGeneration,
               state->avgCredits[i], state->fitness[i],
               (i == winner) ? " <- WINNER" : "");
    }
    
    // Update champion tracking
    if (bestFitness > state->championFitness) {
        // Shift previous champions
        if (state->previousChampion1) freeNetwork(state->previousChampion1);
        state->previousChampion1 = state->previousChampion2;
        state->previousChampion2 = state->currentChampion;
        
        // Create new champion
        state->currentChampion = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
        copyNetworkWeights(state->population[winner], state->currentChampion);
        state->championFitness = bestFitness;
        
        printf("NEW CHAMPION! AI_%d with fitness %.4f\n", winner, bestFitness);
    } else {
        printf("No improvement. Current champion fitness: %.4f\n", state->championFitness);
    }
    
    // Log generation results
    if (state->logFile) {
        double avgFitness = 0.0;
        for (int i = 0; i < state->populationSize; i++) {
            avgFitness += state->fitness[i];
        }
        avgFitness /= state->populationSize;
        
        double diversity = calculatePopulationDiversity(state->population, state->populationSize);
        
        fprintf(state->logFile, "%d,%d,%.4f,%.4f,%.2f,%.4f,%.4f\n",
                state->generation + 1, winner, bestFitness,
                (double)state->wins[winner] / state->gamesPerGeneration,
                state->avgCredits[winner], avgFitness, diversity);
        fflush(state->logFile);
    }
}

// Play one tournament game with all 6 AIs
int playTournamentGame(NeuralNetwork **networks, int numPlayers, TournamentState *state, int gameNum) {
    Player players[MAXPLAYERS];
    
    // Initialize players
    for (int i = 0; i < numPlayers; i++) {
        sprintf(players[i].name, "AI_%d", i);
        players[i].credits = STARTING_CREDITS;
        players[i].status = ACTIVE;
        players[i].hand = NULL;
        players[i].dealer = (i == 0);
        players[i].currentBet = 0;
    }
    
    // Use existing game mechanics but simplified
    Hand *deck = createDeck(1, 1);
    Hand *communityCards = createHand();
    
    // Find dealer and post blinds
    int dealerPos = 0;
    int smallBlindPos = (dealerPos + 1) % numPlayers;
    int bigBlindPos = (dealerPos + 2) % numPlayers;
    
    players[smallBlindPos].credits -= SMALL_BLIND;
    players[smallBlindPos].currentBet = SMALL_BLIND;
    players[bigBlindPos].credits -= BIG_BLIND;
    players[bigBlindPos].currentBet = BIG_BLIND;
    
    int pot = SMALL_BLIND + BIG_BLIND;
    
    // Deal cards
    dealHand(players, numPlayers, deck, communityCards);
    
    // Play simplified betting rounds using AI decisions
    int currentBetAmount = BIG_BLIND;
    
    // Pre-flop
    if (!simplifiedBettingRound(players, numPlayers, &pot, &currentBetAmount, 
                               communityCards, 0, networks, state, gameNum)) {
        // Flop
        resetCurrentBets(players, numPlayers);
        currentBetAmount = 0;
        if (!simplifiedBettingRound(players, numPlayers, &pot, &currentBetAmount, 
                                   communityCards, 3, networks, state, gameNum)) {
            // Turn
            resetCurrentBets(players, numPlayers);
            currentBetAmount = 0;
            if (!simplifiedBettingRound(players, numPlayers, &pot, &currentBetAmount, 
                                       communityCards, 4, networks, state, gameNum)) {
                // River
                resetCurrentBets(players, numPlayers);
                currentBetAmount = 0;
                simplifiedBettingRound(players, numPlayers, &pot, &currentBetAmount, 
                                     communityCards, 5, networks, state, gameNum);
            }
        }
    }
    
    // Determine winner
    int winner = determineWinner(players, numPlayers, communityCards);
    if (winner >= 0) {
        players[winner].credits += pot;
        
        // Update experiences for the winner - mark recent experiences as winning
        if (state->experienceCount[winner] > 0) {
            // Mark the last few experiences of the winner as successful
            int experiencesToMark = (state->experienceCount[winner] > 5) ? 5 : state->experienceCount[winner];
            for (int exp = state->experienceCount[winner] - experiencesToMark; 
                 exp < state->experienceCount[winner]; exp++) {
                state->aiExperiences[winner][exp]->gameOutcome = 1; // Use gameOutcome instead of won
                state->aiExperiences[winner][exp]->reward = 1.0; // Positive reward for winning
            }
        }
    }
    
    // Give small negative rewards to non-winners recent experiences
    for (int i = 0; i < numPlayers; i++) {
        if (i != winner && state->experienceCount[i] > 0) {
            int experiencesToMark = (state->experienceCount[i] > 3) ? 3 : state->experienceCount[i];
            for (int exp = state->experienceCount[i] - experiencesToMark; 
                 exp < state->experienceCount[i]; exp++) {
                state->aiExperiences[i][exp].reward = -0.1; // Small negative reward
                state->aiExperiences[i][exp].gameOutcome = 0; // Lost
            }
        }
    }
    
    // Update average credits in tournament state
    for (int i = 0; i < numPlayers; i++) {
        state->avgCredits[i] = (state->avgCredits[i] * gameNum + players[i].credits) / (gameNum + 1);
    }
    
    // Cleanup
    freeHand(deck, 1);
    freeHand(communityCards, 1);
    for (int i = 0; i < numPlayers; i++) {
        if (players[i].hand) {
            freeHand(players[i].hand, 1);
        }
    }
    
    return winner;
}

// Simplified betting round for tournament games
bool simplifiedBettingRound(Player players[], int numPlayers, int *pot, 
                           int *currentBetAmount, Hand *communityCards, 
                           int cardsRevealed, NeuralNetwork **networks, TournamentState *state, int gameNum) {
    int activePlayers = 0;
    
    // Count active players
    for (int i = 0; i < numPlayers; i++) {
        if (players[i].status == ACTIVE) activePlayers++;
    }
    
    if (activePlayers <= 1) return true; // Game over
    
    // Simple betting round - each player acts once
    for (int i = 0; i < numPlayers; i++) {
        if (players[i].status != ACTIVE) continue;
        
        int toCall = *currentBetAmount - players[i].currentBet;
        
        // Get AI decision
        int decision = makeEnhancedDecision(networks[i], &players[i], communityCards,
                                          *pot, *currentBetAmount, activePlayers, i);
        
        // Collect experience for this decision (we'll determine if it was good later)
        collectExperience(state, i, gameNum, &players[i], communityCards, decision,
                         *pot, *currentBetAmount, false); // false = game outcome unknown yet
        
        switch(decision) {
            case 0: // Fold
                players[i].status = FOLDED;
                activePlayers--;
                break;
                
            case 1: // Call/Check
                if (toCall > 0) {
                    int callAmount = (toCall > players[i].credits) ? 
                                   players[i].credits : toCall;
                    players[i].credits -= callAmount;
                    players[i].currentBet += callAmount;
                    *pot += callAmount;
                }
                break;
                
            case 2: // Raise
                int raiseAmount = *currentBetAmount + BIG_BLIND;
                if (raiseAmount > players[i].credits + players[i].currentBet) {
                    raiseAmount = players[i].credits + players[i].currentBet;
                }
                
                int amountToAdd = raiseAmount - players[i].currentBet;
                *currentBetAmount = raiseAmount;
                
                players[i].credits -= amountToAdd;
                players[i].currentBet = raiseAmount;
                *pot += amountToAdd;
                break;
        }
        
        if (activePlayers <= 1) return true; // Game over
    }
    
    return false; // Continue to next round
}

// Calculate fitness scores for all AIs
void calculateFitness(TournamentState *state) {
    for (int i = 0; i < state->populationSize; i++) {
        double winRate = (double)state->wins[i] / state->gamesPerGeneration;
        double creditScore = state->avgCredits[i] / STARTING_CREDITS;
        
        // Fitness formula: 70% win rate + 30% credit performance
        state->fitness[i] = (winRate * 0.7) + (creditScore * 0.3);
    }
}

// Evolve population: keep top 2, create 4 mutated offspring
void evolvePopulation(TournamentState *state) {
    printf("Evolving population...\n");
    
    // Find top 2 performers
    int indices[6] = {0, 1, 2, 3, 4, 5};
    
    // Simple selection sort to find top 2
    for (int i = 0; i < 2; i++) {
        for (int j = i + 1; j < state->populationSize; j++) {
            if (state->fitness[indices[j]] > state->fitness[indices[i]]) {
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
    }
    
    int parent1 = indices[0];
    int parent2 = indices[1];
    
    printf("Parents: AI_%d (fitness: %.4f), AI_%d (fitness: %.4f)\n", 
           parent1, state->fitness[parent1], parent2, state->fitness[parent2]);
    
    // Create temporary networks for new generation
    NeuralNetwork *newGeneration[6];
    
    // Keep the two parents
    newGeneration[0] = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    newGeneration[1] = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    copyNetworkWeights(state->population[parent1], newGeneration[0]);
    copyNetworkWeights(state->population[parent2], newGeneration[1]);
    
    // Create 4 offspring through mutation
    newGeneration[2] = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    newGeneration[3] = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    newGeneration[4] = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    newGeneration[5] = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    
    // Offspring 1: Parent 1 + small mutation
    copyNetworkWeights(state->population[parent1], newGeneration[2]);
    mutateNetwork(newGeneration[2], 0.1);
    
    // Offspring 2: Parent 2 + small mutation  
    copyNetworkWeights(state->population[parent2], newGeneration[3]);
    mutateNetwork(newGeneration[3], 0.1);
    
    // Offspring 3: Parent 1 + medium mutation
    copyNetworkWeights(state->population[parent1], newGeneration[4]);
    mutateNetwork(newGeneration[4], 0.2);
    
    // Offspring 4: Parent 2 + large mutation
    copyNetworkWeights(state->population[parent2], newGeneration[5]);
    mutateNetwork(newGeneration[5], 0.3);
    
    // Replace old population
    for (int i = 0; i < state->populationSize; i++) {
        freeNetwork(state->population[i]);
        state->population[i] = newGeneration[i];
    }
    
    printf("New generation created: 2 parents + 4 mutated offspring\n");
}

// Calculate population diversity (average pairwise weight differences)
double calculatePopulationDiversity(NeuralNetwork **networks, int count) {
    if (count < 2) return 1.0;
    
    double totalDiversity = 0.0;
    int comparisons = 0;
    
    // Compare each pair of networks
    for (int i = 0; i < count; i++) {
        for (int j = i + 1; j < count; j++) {
            double weightDiff = 0.0;
            int totalWeights = 0;
            
            // Compare input-hidden weights
            for (int inp = 0; inp < INPUT_SIZE; inp++) {
                for (int hid = 0; hid < HIDDEN_SIZE; hid++) {
                    weightDiff += fabs(networks[i]->weightsInputHidden[inp][hid] - 
                                     networks[j]->weightsInputHidden[inp][hid]);
                    totalWeights++;
                }
            }
            
            // Compare hidden-output weights
            for (int hid = 0; hid < HIDDEN_SIZE; hid++) {
                for (int out = 0; out < OUTPUT_SIZE; out++) {
                    weightDiff += fabs(networks[i]->weightsHiddenOutput[hid][out] - 
                                     networks[j]->weightsHiddenOutput[hid][out]);
                    totalWeights++;
                }
            }
            
            totalDiversity += weightDiff / totalWeights;
            comparisons++;
        }
    }
    
    return comparisons > 0 ? totalDiversity / comparisons : 0.0;
}

// Deep copy network weights
void copyNetworkWeights(NeuralNetwork *source, NeuralNetwork *target) {
    if (!source || !target) return;
    
    // Copy input-hidden weights
    for (int i = 0; i < source->inputSize; i++) {
        for (int j = 0; j < source->hiddenSize; j++) {
            target->weightsInputHidden[i][j] = source->weightsInputHidden[i][j];
        }
    }
    
    // Copy hidden-output weights
    for (int i = 0; i < source->hiddenSize; i++) {
        for (int j = 0; j < source->outputSize; j++) {
            target->weightsHiddenOutput[i][j] = source->weightsHiddenOutput[i][j];
        }
    }
    
    // Copy biases
    for (int i = 0; i < source->hiddenSize; i++) {
        target->biasHidden[i] = source->biasHidden[i];
    }
    for (int i = 0; i < source->outputSize; i++) {
        target->biasOutput[i] = source->biasOutput[i];
    }
}

// Mutate network weights
void mutateNetwork(NeuralNetwork *network, double mutationStrength) {
    if (!network) return;
    
    // Mutate input-hidden weights
    for (int i = 0; i < network->inputSize; i++) {
        for (int j = 0; j < network->hiddenSize; j++) {
            double noise = ((double)rand() / RAND_MAX - 0.5) * 2 * mutationStrength;
            network->weightsInputHidden[i][j] += noise;
        }
    }
    
    // Mutate hidden-output weights
    for (int i = 0; i < network->hiddenSize; i++) {
        for (int j = 0; j < network->outputSize; j++) {
            double noise = ((double)rand() / RAND_MAX - 0.5) * 2 * mutationStrength;
            network->weightsHiddenOutput[i][j] += noise;
        }
    }
    
    // Mutate biases
    for (int i = 0; i < network->hiddenSize; i++) {
        double noise = ((double)rand() / RAND_MAX - 0.5) * 2 * mutationStrength;
        network->biasHidden[i] += noise;
    }
    for (int i = 0; i < network->outputSize; i++) {
        double noise = ((double)rand() / RAND_MAX - 0.5) * 2 * mutationStrength;
        network->biasOutput[i] += noise;
    }
}

// Save current and previous champions
void saveChampions(TournamentState *state) {
    if (state->currentChampion) {
        saveNetwork(state->currentChampion, "poker_ai_evolved.dat");
    }
    if (state->previousChampion1) {
        saveNetwork(state->previousChampion1, "poker_ai_previous1.dat");
    }
    if (state->previousChampion2) {
        saveNetwork(state->previousChampion2, "poker_ai_previous2.dat");
    }
}

// Free all tournament memory
void freeTournamentState(TournamentState *state) {
    if (!state) return;
    
    // Free population
    if (state->population) {
        for (int i = 0; i < state->populationSize; i++) {
            if (state->population[i]) {
                freeNetwork(state->population[i]);
            }
        }
        free(state->population);
    }
    
    // Free experience arrays
    if (state->aiExperiences) {
        for (int i = 0; i < state->populationSize; i++) {
            if (state->aiExperiences[i]) {
                free(state->aiExperiences[i]);
            }
        }
        free(state->aiExperiences);
    }
    if (state->experienceCount) free(state->experienceCount);
    
    // Free champions
    if (state->currentChampion) freeNetwork(state->currentChampion);
    if (state->previousChampion1) freeNetwork(state->previousChampion1);
    if (state->previousChampion2) freeNetwork(state->previousChampion2);
    
    // Free arrays
    if (state->fitness) free(state->fitness);
    if (state->wins) free(state->wins);
    if (state->avgCredits) free(state->avgCredits);
    if (state->diversityHistory) free(state->diversityHistory);
    
    // Close log file
    if (state->logFile) fclose(state->logFile);
    
    free(state);
}