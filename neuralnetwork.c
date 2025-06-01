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

// REPLACE your existing encodeGameState function with this enhanced version
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
    
    // FEATURE 17: Betting round phase
    if (numCommunity == 0) output[17] = 0.0;      // Pre-flop
    else if (numCommunity == 3) output[17] = 0.33; // Flop
    else if (numCommunity == 4) output[17] = 0.66; // Turn
    else if (numCommunity == 5) output[17] = 1.0;  // River
    
    // FEATURE 18: Relative bet sizing
    if (pot > 0) {
        output[18] = (double)(currentBet - player->currentBet) / pot;
    }
    
    // FEATURE 19: Last action aggression (simplified - can be enhanced)
    output[19] = 0.5;  // Placeholder for now
    
    // FEATURE 20: Pair in hand
    if (cardCount >= 2) {
        output[20] = (playerCards[0].value == playerCards[1].value) ? 1.0 : 0.0;
    }
    
    // FEATURE 21: High card strength
    if (cardCount >= 2) {
        int highCard = (playerCards[0].value > playerCards[1].value) ? 
                       playerCards[0].value : playerCards[1].value;
        output[21] = (double)highCard / 13.0;
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
            break;
        case 1: // Call/Check
            profile->callCount++;
            if (voluntaryAction) profile->voluntaryPuts++;
            break;
        case 2: // Raise
            profile->raiseCount++;
            profile->voluntaryPuts++;
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



void train(NeuralNetwork *nn, double **trainingInputs, double **trainingOutputs, int numSamples) {
    printf("\n");
    printRepeatedChar('=', 60);
    printf("\n");
    printf("STARTING NEURAL NETWORK TRAINING\n");
    printRepeatedChar('=', 60);
    printf("\n");
    printf("Network Architecture: %d → %d → %d\n", nn->inputSize, nn->hiddenSize, nn->outputSize);
    printf("Training samples: %d\n", numSamples);
    printf("Learning rate: %.4f\n", nn->learningRate);
    printf("Target epochs: %d\n", TRAINING_EPOCHS);
    printf("Activation function: %s\n", 
           nn->activationFunction == ACTIVATION_SIGMOID ? "Sigmoid" : 
           nn->activationFunction == ACTIVATION_RELU ? "ReLU" : "Tanh");
    printRepeatedChar('=', 60);
    printf("\n\n");
    
    // Use the enhanced training with monitoring
    trainWithMonitoring(nn, trainingInputs, trainingOutputs, numSamples, TRAINING_EPOCHS);
    
    printf("\nTraining complete!\n");
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
    stats->accuracyHistory = malloc(maxEpochs * sizeof(double));
    stats->epochNumbers = malloc(maxEpochs * sizeof(int));
    
    if (!stats->lossHistory || !stats->accuracyHistory || !stats->epochNumbers) {
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
    
    // Open log file for training progress
    stats->logFile = fopen("training_log.csv", "w");
    if (stats->logFile) {
        fprintf(stats->logFile, "Epoch,Loss,Accuracy,Time_Elapsed,Learning_Rate\n");
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
    
    // Calculate current loss and accuracy
    double currentLoss = calculateLoss(nn, inputs, targets, numSamples);
    double currentAccuracy = calculateAccuracy(nn, inputs, targets, numSamples);
    
    // Store in history
    stats->lossHistory[stats->currentEpoch] = currentLoss;
    stats->accuracyHistory[stats->currentEpoch] = currentAccuracy;
    stats->epochNumbers[stats->currentEpoch] = stats->currentEpoch;
    
    // Track best performance
    if (currentLoss < stats->bestLoss) {
        stats->bestLoss = currentLoss;
        stats->bestEpoch = stats->currentEpoch;
    }
    
    // Set initial loss on first epoch
    if (stats->currentEpoch == 0) {
        stats->initialLoss = currentLoss;
    }
    
    // Calculate elapsed time
    clock_t currentTime = clock();
    double elapsedSeconds = ((double)(currentTime - stats->startTime)) / CLOCKS_PER_SEC;
    
    // Log to file
    if (stats->logFile) {
        fprintf(stats->logFile, "%d,%.6f,%.4f,%.2f,%.6f\n", 
                stats->currentEpoch, currentLoss, currentAccuracy, elapsedSeconds, learningRate);
        fflush(stats->logFile);  // Ensure data is written immediately
    }
    
    stats->currentEpoch++;
}

// Display training progress
void displayTrainingProgress(TrainingStats *stats, bool verbose) {
    if (!stats || stats->currentEpoch == 0) return;
    
    int epoch = stats->currentEpoch - 1;  // Last completed epoch
    double currentLoss = stats->lossHistory[epoch];
    double currentAccuracy = stats->accuracyHistory[epoch];
    
    // Calculate elapsed time
    clock_t currentTime = clock();
    double elapsedSeconds = ((double)(currentTime - stats->startTime)) / CLOCKS_PER_SEC;
    
    if (verbose || epoch % 100 == 0 || epoch == stats->maxEpochs - 1) {
        printf("Epoch %4d/%d | Loss: %.6f | Accuracy: %.2f%% | Time: %.1fs", 
               epoch, stats->maxEpochs - 1, currentLoss, currentAccuracy * 100, elapsedSeconds);
        
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

// Display final training summary
void displayTrainingSummary(TrainingStats *stats) {
    if (!stats) return;
    
    clock_t endTime = clock();
    double totalTime = ((double)(endTime - stats->startTime)) / CLOCKS_PER_SEC;
    
    printf("\n");
    printRepeatedChar('=', 60);
    printf("\n");
    printf("TRAINING COMPLETED!\n");
    printf("\n");
    printRepeatedChar('=', 60);
    printf("\n");
    printf("Total Epochs:     %d\n", stats->currentEpoch);
    printf("Total Time:       %.2f seconds (%.2f minutes)\n", totalTime, totalTime / 60.0);
    printf("Time per Epoch:   %.4f seconds\n", totalTime / stats->currentEpoch);
    
    printf("\nPERFORMANCE SUMMARY:\n");
    printf("Initial Loss:     %.6f\n", stats->initialLoss);
    printf("Final Loss:       %.6f\n", stats->lossHistory[stats->currentEpoch - 1]);
    printf("Best Loss:        %.6f (Epoch %d)\n", stats->bestLoss, stats->bestEpoch);
    printf("Loss Improvement: %.6f (%.1f%% reduction)\n", 
           stats->initialLoss - stats->bestLoss,
           ((stats->initialLoss - stats->bestLoss) / stats->initialLoss) * 100);
    
    printf("Final Accuracy:   %.2f%%\n", stats->accuracyHistory[stats->currentEpoch - 1] * 100);
    
    // Learning assessment
    double finalLoss = stats->lossHistory[stats->currentEpoch - 1];
    printf("\nLEARNING ASSESSMENT:\n");
    if (finalLoss < stats->initialLoss * 0.1) {
        printf("Status: EXCELLENT - Very strong learning occurred\n");
    } else if (finalLoss < stats->initialLoss * 0.5) {
        printf("Status: GOOD - Solid learning progress\n");
    } else if (finalLoss < stats->initialLoss * 0.8) {
        printf("Status: MODERATE - Some learning occurred\n");
    } else {
        printf("Status: POOR - Limited learning (may need more epochs or different learning rate)\n");
    }
    
    printf("\nDetailed log saved to: training_log.csv\n");
    printRepeatedChar('=', 60);
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
    
    // Training loop with monitoring
    for (int epoch = 0; epoch < epochs; epoch++) {
        double epochStartTime = clock();
        
        // Train on all samples
        for (int sample = 0; sample < numSamples; sample++) {
            forwardpropagate(nn, trainingInputs[sample]);
            backpropagate(nn, trainingOutputs[sample]);
            updateWeights(nn);
        }
        
        // Update statistics and display progress
        updateTrainingStats(stats, nn, trainingInputs, trainingOutputs, numSamples, nn->learningRate);
        
        // Display progress (verbose every 100 epochs, progress bar every 10%)
        bool verbose = (epoch % 100 == 0) || (epoch == epochs - 1);
        displayTrainingProgress(stats, verbose);
        
        // Early stopping if loss becomes very small
        if (stats->lossHistory[epoch] < 0.0001) {
            printf("\nEarly stopping: Loss became very small (%.6f)\n", stats->lossHistory[epoch]);
            break;
        }
        
        // Adaptive learning rate (optional)
        if (epoch > 100 && epoch % 200 == 0) {
            // Check if learning has stagnated
            double recentLossAvg = 0.0;
            for (int i = epoch - 50; i < epoch; i++) {
                recentLossAvg += stats->lossHistory[i];
            }
            recentLossAvg /= 50.0;
            
            if (recentLossAvg > stats->lossHistory[epoch - 100] * 0.95) {
                // Learning stagnated, reduce learning rate
                nn->learningRate *= 0.8;
                printf("Learning rate reduced to %.6f (learning stagnated)\n", nn->learningRate);
            }
        }
    }
    
    // Display final summary
    displayTrainingSummary(stats);
    
    // Cleanup
    if (stats->logFile) {
        fclose(stats->logFile);
    }
    free(stats->lossHistory);
    free(stats->accuracyHistory);
    free(stats->epochNumbers);
    free(stats);
}

// Save training curves to a plottable format
void saveTrainingCurves(TrainingStats *stats, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        printf("Error: Could not create training curves file\n");
        return;
    }
    
    fprintf(file, "# Training Curves Data - Import into Excel/Python for plotting\n");
    fprintf(file, "# Epoch,Loss,Accuracy\n");
    
    for (int i = 0; i < stats->currentEpoch; i++) {
        fprintf(file, "%d,%.6f,%.4f\n", i, stats->lossHistory[i], stats->accuracyHistory[i]);
    }
    
    fclose(file);
    printf("Training curves saved to %s\n", filename);
}

// Quick visualization using ASCII art (for terminals without graphics)
void displayLossGraph(TrainingStats *stats) {
    if (!stats || stats->currentEpoch < 10) return;
    
    printf("\nLOSS CURVE (ASCII Visualization):\n");
    printf("Loss\n");
    printf("^\n");
    
    // Find min and max loss for scaling
    double minLoss = stats->lossHistory[0];
    double maxLoss = stats->lossHistory[0];
    
    for (int i = 1; i < stats->currentEpoch; i++) {
        if (stats->lossHistory[i] < minLoss) minLoss = stats->lossHistory[i];
        if (stats->lossHistory[i] > maxLoss) maxLoss = stats->lossHistory[i];
    }
    
    // Draw graph (simplified)
    int graphHeight = 10;
    for (int row = graphHeight; row >= 0; row--) {
        double threshold = minLoss + (maxLoss - minLoss) * row / graphHeight;
        printf("|");
        
        // Sample every 10th epoch for display
        for (int epoch = 0; epoch < stats->currentEpoch; epoch += stats->currentEpoch / 50) {
            if (stats->lossHistory[epoch] >= threshold) {
                printf("*");
            } else {
                printf(" ");
            }
        }
        printf("\n");
    }
    
    printf("+");
    for (int i = 0; i < 50; i++) printf("-");
    printf("> Epochs\n");
    printf("0");
    for (int i = 0; i < 45; i++) printf(" ");
    printf("%d\n", stats->currentEpoch - 1);
    
    printf("Loss range: %.4f to %.4f\n", minLoss, maxLoss);
}