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
// Phase 1: Train minimal bootstrap with monitoring
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
    
    printf("Allocating memory for %d bootstrap samples...\n", numSamples);
    
    for (int i = 0; i < numSamples; i++) {
        inputs[i] = malloc(INPUT_SIZE * sizeof(double));
        outputs[i] = malloc(OUTPUT_SIZE * sizeof(double));
        if (!inputs[i] || !outputs[i]) {
            printf("Error: Memory allocation failed at sample %d\n", i);
            return NULL;
        }
    }
    
    printf("Generating minimal bootstrap data...\n");
    generateMinimalBootstrap(inputs, outputs, numSamples);
    
    printf("Creating fresh neural network...\n");
    NeuralNetwork *nn = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    if (!nn) {
        printf("Error: Failed to create neural network\n");
        return NULL;
    }
    
    nn->learningRate = 0.05;  // Higher learning rate for quick bootstrap
    
    // Use full monitoring system for bootstrap training
    int bootstrapEpochs = 200;
    printf("Starting bootstrap training with monitoring (%d epochs)...\n\n", bootstrapEpochs);
    
    // Train with full monitoring system
    trainWithMonitoring(nn, inputs, outputs, numSamples, bootstrapEpochs);
    
    printf("\nTesting bootstrap network...\n");
    printRepeatedChar('=', 40);
    printf("\n");
    
    // Test a few examples
    for (int i = 0; i < 3; i++) {
        forwardpropagate(nn, inputs[i]);
        printf("Test %d:\n", i+1);
        printf("  Expected: Fold=%.2f Call=%.2f Raise=%.2f\n",
               outputs[i][0], outputs[i][1], outputs[i][2]);
        printf("  Predicted: Fold=%.2f Call=%.2f Raise=%.2f\n",
               nn->outputLayer[0].value, nn->outputLayer[1].value, nn->outputLayer[2].value);
        
        // Determine actions
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
    
    printf("Saving bootstrap network...\n");
    saveNetwork(nn, "poker_ai_bootstrap.dat");
    
    printf("\n");
    printRepeatedChar('=', 60);
    printf("\n");
    printf("BOOTSTRAP COMPLETE!\n");
    printf("Network learned basic rules - ready for self-play learning.\n");
    printf("Bootstrap saved as: poker_ai_bootstrap.dat\n");
    printf("Bootstrap training log: training_log.csv\n");
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
    printf("PHASE 2: PURE REINFORCEMENT LEARNING\n");
    printRepeatedChar('=', 70);
    printf("\n");
    printf("Starting self-play from bootstrap network...\n");
    printf("The AI will now discover its own strategy through experience!\n");
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
    
    // Initialize training monitoring for the best AI (AI_0)
    int maxTrainingEpochs = numGames / 3;  // One training epoch per 3 games
    TrainingStats *mainStats = initializeTrainingStats(maxTrainingEpochs);
    
    // Validation set for loss calculation
    int validationSize = 500;
    double **validationInputs = malloc(validationSize * sizeof(double*));
    double **validationOutputs = malloc(validationSize * sizeof(double*));
    for (int i = 0; i < validationSize; i++) {
        validationInputs[i] = malloc(INPUT_SIZE * sizeof(double));
        validationOutputs[i] = malloc(OUTPUT_SIZE * sizeof(double));
    }
    
    // Game tracking
    int wins[MAXPLAYERS] = {0};
    double avgCredits[MAXPLAYERS] = {0};
    int progressCheckpoints = 20;
    int gamesPerCheckpoint = numGames / progressCheckpoints;
    
    FILE *progressFile = fopen("selfplay_progress.csv", "w");
    if (progressFile) {
        fprintf(progressFile, "Game,AI_0_WinRate,AI_1_WinRate,AI_2_WinRate,AI_3_WinRate,AvgCredits,ExperienceBufferSize\n");
    }
    
    clock_t startTime = clock();
    int trainingEpochCount = 0;
    
    printf("Starting pure self-play learning...\n");
    printf("Each AI will develop its own strategy through experience!\n\n");
    
    // Main self-play training loop
    for (int game = 0; game < numGames; game++) {
        GameRecord record = playEnhancedSelfPlayGame(networks, numPlayers, rb);
        
        // Update game statistics
        wins[record.winner]++;
        for (int i = 0; i < numPlayers; i++) {
            avgCredits[i] = (avgCredits[i] * game + record.finalCredits[i]) / (game + 1);
        }
        
        // Neural network training from experience
        if (rb->size > 500 && game % 3 == 0) {
            trainingEpochCount++;
            
            // Create validation set from recent experiences
            if (rb->size >= validationSize) {
                for (int i = 0; i < validationSize; i++) {
                    int idx = (rb->size - validationSize + i) % rb->capacity;
                    memcpy(validationInputs[i], rb->buffer[idx].gameState, INPUT_SIZE * sizeof(double));
                    
                    // Create target output based on experience
                    for (int j = 0; j < OUTPUT_SIZE; j++) {
                        validationOutputs[i][j] = 0.33;  // Start neutral
                    }
                    
                    // Adjust based on reward
                    if (rb->buffer[idx].reward > 0) {
                        validationOutputs[i][rb->buffer[idx].action] = 0.7;
                        for (int j = 0; j < OUTPUT_SIZE; j++) {
                            if (j != rb->buffer[idx].action) {
                                validationOutputs[i][j] = 0.15;
                            }
                        }
                    } else if (rb->buffer[idx].reward < -0.5) {
                        validationOutputs[i][rb->buffer[idx].action] = 0.1;
                        for (int j = 0; j < OUTPUT_SIZE; j++) {
                            if (j != rb->buffer[idx].action) {
                                validationOutputs[i][j] = 0.45;
                            }
                        }
                    }
                    
                    // Normalize
                    double sum = 0;
                    for (int j = 0; j < OUTPUT_SIZE; j++) {
                        sum += validationOutputs[i][j];
                    }
                    if (sum > 0) {
                        for (int j = 0; j < OUTPUT_SIZE; j++) {
                            validationOutputs[i][j] /= sum;
                        }
                    }
                }
                
                // Train all AIs from experience
                for (int i = 0; i < numPlayers; i++) {
                    trainFromExperience(networks[i], rb, 128);
                }
                
                // Update training statistics (for AI_0 as representative)
                if (mainStats && trainingEpochCount <= maxTrainingEpochs) {
                    updateTrainingStats(mainStats, networks[0], validationInputs, 
                                      validationOutputs, validationSize, networks[0]->learningRate);
                }
            }
        }
        
        // Show progress every 100 epochs
        if (trainingEpochCount > 0 && trainingEpochCount % 100 == 0 && mainStats) {
            clock_t currentTime = clock();
            double elapsed = ((double)(currentTime - startTime)) / CLOCKS_PER_SEC;
            
            printf("Game %d/%d (%.1f%%) | Time: %.1fs | Experience: %d samples\n", 
                   game, numGames, (game * 100.0) / numGames, elapsed, rb->size);
            displayTrainingProgress(mainStats, false);
        }
        
        // Show game progress checkpoints (without epoch info)
        if (game > 0 && (game % gamesPerCheckpoint == 0 || game == numGames - 1)) {
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
        }
        
        // Progress dots for other games
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
    printf("Neural network training epochs: %d\n", trainingEpochCount);
    
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
    
    // Display detailed training summary at the end
    if (mainStats && rb->size >= validationSize) {
        printf("\n");
        displayTrainingSummary(mainStats, networks[bestAI], validationInputs, validationSize);
    }
    
    // Save the best evolved network
    saveNetwork(networks[bestAI], "poker_ai_evolved.dat");
    printf("Best evolved AI saved as: poker_ai_evolved.dat\n");
    
    // Save all networks as backups
    for (int i = 0; i < numPlayers; i++) {
        char filename[100];
        sprintf(filename, "evolved_ai_%d.dat", i);
        saveNetwork(networks[i], filename);
    }
    
    printf("All evolved AIs saved. Training logs: selfplay_progress.csv\n");
    printRepeatedChar('=', 70);
    printf("\n");
    
    // Cleanup
    if (progressFile) fclose(progressFile);
    if (bootstrap) freeNetwork(bootstrap);
    
    // Cleanup training statistics
    if (mainStats) {
        if (mainStats->logFile) fclose(mainStats->logFile);
        free(mainStats->lossHistory);
        free(mainStats->epochNumbers);
        free(mainStats);
    }
    
    for (int i = 0; i < numPlayers; i++) {
        freeNetwork(networks[i]);
    }
    free(networks);
    
    // Cleanup validation data
    for (int i = 0; i < validationSize; i++) {
        free(validationInputs[i]);
        free(validationOutputs[i]);
    }
    free(validationInputs);
    free(validationOutputs);
    
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
    
    // Phase 2: Pure self-play learning
    pureReinforcementLearning(numGames, numPlayers);
    
    printf("\n");
    printRepeatedChar('*', 70);
    printf("\n");
    printf("TWO-PHASE TRAINING COMPLETE!\n");
    printf("Your AI has evolved its own poker strategy!\n");
    printf("Files created:\n");
    printf("  - poker_ai_bootstrap.dat (Phase 1 result)\n");
    printf("  - poker_ai_evolved.dat (Phase 2 best AI)\n");
    printf("  - evolved_ai_*.dat (All trained AIs)\n");
    printf("  - selfplay_progress.csv (Training progress log)\n");
    printRepeatedChar('*', 70);
    printf("\n");
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
    trainer->gamesPerGeneration = 150;  // Each AI plays ~150 games per generation
    
    // Evolution parameters (your specified values)
    trainer->selectionRate = 0.20;     // Keep top 20%
    trainer->mutationRate = 0.50;      // 50% mutation
    trainer->crossoverRate = 0.30;     // 30% crossover  
    trainer->freshRate = 0.20;         // 20% fresh random
    
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
    printf("Population: %d AIs | Generations: %d | Tables: %d\n", 
           populationSize, maxGenerations, trainer->numTables);
    
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

void runTournamentGeneration(EvolutionaryTrainer *trainer) {
    printf("Running tournament generation %d...\n", trainer->currentGeneration + 1);
    trainer->generationStartTime = clock();
    
    // Reset all individual stats for this generation
    for (int i = 0; i < trainer->populationSize; i++) {
        trainer->population[i].wins = 0;
        trainer->population[i].games = 0;
        trainer->population[i].totalCredits = 0.0;
    }
    
    // Run multiple tournament rounds to get sufficient data
    int roundsPerGeneration = (trainer->gamesPerGeneration + trainer->numTables - 1) / trainer->numTables;
    
    for (int round = 0; round < roundsPerGeneration; round++) {
        printf("  Round %d/%d: ", round + 1, roundsPerGeneration);
        
        // Set up random tables for this round
        setupTournamentTables(trainer);
        
        // Run all tables in parallel (conceptually)
        for (int table = 0; table < trainer->numTables; table++) {
            runSingleTable(trainer, table);
            
            // Progress indicator
            if (table % 50 == 0) {
                printf(".");
                fflush(stdout);
            }
        }
        printf(" Complete!\n");
    }
    
    // Calculate final stats for this generation
    for (int i = 0; i < trainer->populationSize; i++) {
        Individual *individual = &trainer->population[i];
        if (individual->games > 0) {
            individual->winRate = (double)individual->wins / individual->games;
            individual->avgCredits = individual->totalCredits / individual->games;
        } else {
            individual->winRate = 0.0;
            individual->avgCredits = 0.0;
        }
    }
    
    printf("Generation %d tournament complete!\n", trainer->currentGeneration + 1);
}

void runSingleTable(EvolutionaryTrainer *trainer, int tableIndex) {
    TournamentTable *table = &trainer->tables[tableIndex];
    
    // Create temporary player array for this game
    Player players[4];
    NeuralNetwork *networks[4];
    
    // Set up players
    for (int i = 0; i < 4; i++) {
        int aiIndex = table->playerIndices[i];
        networks[i] = trainer->population[aiIndex].network;
        
        // Initialize player
        sprintf(players[i].name, "AI_%d", aiIndex);
        players[i].credits = STARTING_CREDITS;
        players[i].currentBet = 0;
        players[i].hand = NULL;
    }
    
    // Create a temporary replay buffer (not used for training during tournaments)
    ReplayBuffer *tempBuffer = createReplayBuffer(1000);
    
    // Play the game using existing self-play infrastructure
    GameRecord record = playEnhancedSelfPlayGame(networks, 4, tempBuffer);
    
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
    
    // Cleanup
    free(tempBuffer->buffer);
    free(tempBuffer);
    
    for (int i = 0; i < 4; i++) {
        if (players[i].hand) {
            freeHand(players[i].hand, 1);
        }
    }
}

// ===================================================================
// FITNESS EVALUATION
// ===================================================================

double calculateIndividualFitness(Individual *individual) {
    if (individual->games == 0) return 0.0;
    
    // Multi-objective fitness function
    double winRateScore = individual->winRate * 100.0;           // 0-100 points
    double creditsScore = (individual->avgCredits / 1000.0) * 50.0; // Up to 50 points
    double consistencyBonus = 0.0;
    
    // Bonus for consistency (playing many games without huge losses)
    if (individual->games >= 50) {
        double avgCredits = individual->avgCredits;
        if (avgCredits > 900) consistencyBonus = 10.0;      // Very consistent
        else if (avgCredits > 800) consistencyBonus = 5.0;  // Reasonably consistent
    }
    
    // Experience bonus (reward AIs that have played more games)
    double experienceBonus = fmin(individual->games / 100.0, 5.0); // Up to 5 points
    
    // Calculate total fitness
    double fitness = winRateScore + creditsScore + consistencyBonus + experienceBonus;
    
    return fitness;
}

void evaluatePopulationFitness(EvolutionaryTrainer *trainer) {
    printf("Evaluating population fitness...\n");
    
    double totalFitness = 0.0;
    double bestFitness = 0.0;
    int bestIndex = 0;
    
    for (int i = 0; i < trainer->populationSize; i++) {
        Individual *individual = &trainer->population[i];
        individual->fitness = calculateIndividualFitness(individual);
        totalFitness += individual->fitness;
        
        if (individual->fitness > bestFitness) {
            bestFitness = individual->fitness;
            bestIndex = i;
        }
        
        // Assign strategy labels based on behavior
        assignStrategyLabel(individual);
    }
    
    // Update global best
    if (bestFitness > trainer->bestFitnessEver) {
        trainer->bestFitnessEver = bestFitness;
        trainer->bestIndividualEver = bestIndex;
        
        // Update hall of fame
        updateHallOfFame(trainer, &trainer->population[bestIndex]);
        
        printf("NEW BEST AI FOUND! Index %d, Fitness: %.2f\n", bestIndex, bestFitness);
    }
    
    // Record generation average
    double avgFitness = totalFitness / trainer->populationSize;
    trainer->avgFitnessHistory[trainer->currentGeneration] = avgFitness;
    
    printf("Generation %d fitness - Best: %.2f | Average: %.2f\n", 
           trainer->currentGeneration + 1, bestFitness, avgFitness);
}

void assignStrategyLabel(Individual *individual) {
    if (individual->games < 10) {
        strcpy(individual->strategy, "Untested");
        return;
    }
    
    double winRate = individual->winRate;
    double avgCredits = individual->avgCredits;
    
    // Simple strategy classification based on performance
    if (winRate > 0.4 && avgCredits > 1100) {
        strcpy(individual->strategy, "Dominant");
    } else if (winRate > 0.3 && avgCredits > 1000) {
        strcpy(individual->strategy, "Aggressive");
    } else if (winRate > 0.2 && avgCredits > 900) {
        strcpy(individual->strategy, "Conservative");
    } else if (winRate > 0.15) {
        strcpy(individual->strategy, "Survivor");
    } else {
        strcpy(individual->strategy, "Struggling");
    }
}

void updateHallOfFame(EvolutionaryTrainer *trainer, Individual *candidate) {
    // Simple insertion into hall of fame (top 10)
    for (int i = 0; i < 10; i++) {
        if (trainer->hallOfFame[i] == NULL || 
            candidate->fitness > trainer->hallOfFame[i]->fitness) {
            
            // Shift others down
            for (int j = 9; j > i; j--) {
                trainer->hallOfFame[j] = trainer->hallOfFame[j-1];
            }
            
            // Insert new candidate
            trainer->hallOfFame[i] = candidate;
            break;
        }
    }
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
    // Sort population by fitness (highest first)
    qsort(trainer->population, trainer->populationSize, sizeof(Individual), compareFitness);
    
    printf("Population ranked by fitness!\n");
    printf("Top 5 AIs:\n");
    for (int i = 0; i < 5; i++) {
        Individual *ai = &trainer->population[i];
        printf("  #%d: Fitness %.2f | Win Rate %.1f%% | Avg Credits %.0f | %s\n",
               i + 1, ai->fitness, ai->winRate * 100, ai->avgCredits, ai->strategy);
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
    printRepeatedChar('=', 70);
    printf("\n");
    printf("GENERATION %d SUMMARY\n", trainer->currentGeneration);
    printRepeatedChar('=', 70);
    printf("\n");
    
    // Time statistics
    printf("Generation Time: %.1f seconds | Total Time: %.1f minutes\n", 
           generationTime, totalTime / 60.0);
    
    // Performance statistics
    Individual *best = &trainer->population[0]; // Already sorted
    printf("\nBest AI This Generation:\n");
    printf("  Fitness: %.2f | Win Rate: %.1f%% | Avg Credits: %.0f\n",
           best->fitness, best->winRate * 100, best->avgCredits);
    printf("  Strategy: %s | Games Played: %d\n", best->strategy, best->games);
    
    // Population diversity
    printf("\nPopulation Diversity:\n");
    int strategyCounts[6] = {0}; // Dominant, Aggressive, Conservative, Survivor, Struggling, Other
    for (int i = 0; i < trainer->populationSize; i++) {
        if (strcmp(trainer->population[i].strategy, "Dominant") == 0) strategyCounts[0]++;
        else if (strcmp(trainer->population[i].strategy, "Aggressive") == 0) strategyCounts[1]++;
        else if (strcmp(trainer->population[i].strategy, "Conservative") == 0) strategyCounts[2]++;
        else if (strcmp(trainer->population[i].strategy, "Survivor") == 0) strategyCounts[3]++;
        else if (strcmp(trainer->population[i].strategy, "Struggling") == 0) strategyCounts[4]++;
        else strategyCounts[5]++;
    }
    
    printf("  Dominant: %d | Aggressive: %d | Conservative: %d\n", 
           strategyCounts[0], strategyCounts[1], strategyCounts[2]);
    printf("  Survivor: %d | Struggling: %d | Other: %d\n",
           strategyCounts[3], strategyCounts[4], strategyCounts[5]);
    
    // Progress estimate
    double avgTimePerGeneration = totalTime / (trainer->currentGeneration + 1);
    double estimatedRemaining = avgTimePerGeneration * (trainer->maxGenerations - trainer->currentGeneration - 1);
    printf("\nProgress: %d/%d generations (%.1f%%)\n",
           trainer->currentGeneration + 1, trainer->maxGenerations,
           ((trainer->currentGeneration + 1) * 100.0) / trainer->maxGenerations);
    printf("Estimated time remaining: %.1f minutes\n", estimatedRemaining / 60.0);
    
    printRepeatedChar('=', 70);
    printf("\n");
    
    // Log to file
    if (trainer->evolutionLog) {
        for (int i = 0; i < 10; i++) { // Log top 10 this generation
            Individual *ai = &trainer->population[i];
            fprintf(trainer->evolutionLog, "%d,%d,%.4f,%.4f,%.2f,%s,%d,%d\n",
                    trainer->currentGeneration, i, ai->fitness, ai->winRate,
                    ai->avgCredits, ai->strategy, ai->parentA, ai->parentB);
        }
        fflush(trainer->evolutionLog);
    }
}

void displayFinalEvolutionResults(EvolutionaryTrainer *trainer) {
    clock_t endTime = clock();
    double totalTime = ((double)(endTime - trainer->startTime)) / CLOCKS_PER_SEC;
    
    printf("\n");
    printRepeatedChar('*', 80);
    printf("\n");
    printf("EVOLUTIONARY TRAINING COMPLETE!\n");
    printRepeatedChar('*', 80);
    printf("\n");
    
    printf("EVOLUTION SUMMARY:\n");
    printf("Total Generations: %d\n", trainer->maxGenerations);
    printf("Population Size: %d AIs\n", trainer->populationSize);
    printf("Total Training Time: %.2f minutes (%.2f hours)\n", 
           totalTime / 60.0, totalTime / 3600.0);
    printf("Games Played: ~%d total games\n", 
           trainer->populationSize * trainer->gamesPerGeneration * trainer->maxGenerations);
    
    printf("\nCHAMPION AI:\n");
    Individual *champion = &trainer->population[0];
    printf("Final Fitness: %.2f\n", champion->fitness);
    printf("Win Rate: %.1f%% (%d wins in %d games)\n", 
           champion->winRate * 100, champion->wins, champion->games);
    printf("Average Credits: %.0f\n", champion->avgCredits);
    printf("Strategy Type: %s\n", champion->strategy);
    printf("Generation Born: %d\n", champion->generation);
    
    printf("\nHALL OF FAME (Top 5 All-Time):\n");
    for (int i = 0; i < 5 && trainer->hallOfFame[i] != NULL; i++) {
        Individual *legend = trainer->hallOfFame[i];
        printf("  #%d: Fitness %.2f | Win Rate %.1f%% | %s\n",
               i + 1, legend->fitness, legend->winRate * 100, legend->strategy);
    }
    
    printf("\nEvolution log saved to: evolution_log.csv\n");
    printf("Champion AI saved as: poker_ai_evolved_champion.dat\n");
    
    printRepeatedChar('*', 80);
    printf("\n");
}

// ===================================================================
// MAIN EVOLUTIONARY TRAINING FUNCTION
// ===================================================================

void trainEvolutionaryAI(int populationSize, int maxGenerations) {
    printf("\n");
    printRepeatedChar('=', 70);
    printf("\n");
    printf("EVOLUTIONARY AI TRAINING\n");
    printf("Population-based optimization with natural selection\n");
    printRepeatedChar('=', 70);
    printf("\n");
    
    // Initialize trainer
    EvolutionaryTrainer *trainer = createEvolutionaryTrainer(populationSize, maxGenerations);
    if (!trainer) {
        printf("Failed to create evolutionary trainer!\n");
        return;
    }
    
    trainer->startTime = clock();
    
    // Initialize opponent profiles for game playing
    initializeOpponentProfiles(8); // Max players in any game
    
    // Create initial diverse population
    initializePopulation(trainer);
    
    printf("\nStarting evolutionary training...\n");
    printf("This will take approximately %.1f hours\n", 
           (populationSize * maxGenerations * 150.0) / 3600.0); // Rough estimate
    
    // Main evolution loop
    for (trainer->currentGeneration = 0; trainer->currentGeneration < trainer->maxGenerations; trainer->currentGeneration++) {
        printf("\n>>> GENERATION %d/%d <<<\n", 
               trainer->currentGeneration + 1, trainer->maxGenerations);
        
        // 1. Run tournament for this generation
        runTournamentGeneration(trainer);
        
        // 2. Evaluate fitness
        evaluatePopulationFitness(trainer);
        
        // 3. Rank population
        rankPopulation(trainer);
        
        // 4. Display generation summary
        displayGenerationSummary(trainer);
        
        // 5. Save checkpoint
        if (trainer->currentGeneration % 5 == 0) {
            saveEvolutionCheckpoint(trainer);
        }
        
        // 6. Evolve to next generation (except on last generation)
        if (trainer->currentGeneration < trainer->maxGenerations - 1) {
            evolvePopulation(trainer);
        }
    }
    
    // Final results
    displayFinalEvolutionResults(trainer);
    
    // Save the champion
    saveNetwork(trainer->population[0].network, "poker_ai_evolved_champion.dat");
    
    // Save top 10 AIs
    for (int i = 0; i < 10; i++) {
        char filename[100];
        sprintf(filename, "evolved_champion_%d.dat", i);
        saveNetwork(trainer->population[i].network, filename);
    }
    
    // Cleanup
    freeEvolutionaryTrainer(trainer);
    
    printf("Evolutionary training complete! 🎉\n");
}

void saveEvolutionCheckpoint(EvolutionaryTrainer *trainer) {
    // Save current best AI as checkpoint
    char filename[100];
    sprintf(filename, "evolution_checkpoint_gen_%d.dat", trainer->currentGeneration);
    saveNetwork(trainer->population[0].network, filename);
    
    printf("Checkpoint saved: %s\n", filename);
}