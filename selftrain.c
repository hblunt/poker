#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "neuralnetwork.h"
#include "aiplayer.h"
#include "player.h"
#include "game.h"
#include "cards.h"
#include "scoringsystem.h"
#include "selftrain.h"

// ===================================================================
// FOUNDATIONAL FUNCTIONS (Basic functionality needed by everything)
// ===================================================================

// Create replay buffer for storing training experiences
ReplayBuffer* createReplayBuffer(int capacity) {
    ReplayBuffer *rb = malloc(sizeof(ReplayBuffer));
    if (!rb) {
        printf("Error: Could not allocate memory for replay buffer\n");
        return NULL;
    }
    
    rb->buffer = malloc(capacity * sizeof(Experience));
    if (!rb->buffer) {
        printf("Error: Could not allocate memory for experience buffer\n");
        free(rb);
        return NULL;
    }
    
    rb->capacity = capacity;
    rb->size = 0;
    rb->writeIndex = 0;
    
    return rb;
}

// Add experience to replay buffer
void addExperience(ReplayBuffer *rb, double *gameState, int action, double reward, 
                  int playerIndex, int handOutcome, int gameOutcome) {
    if (!rb || !gameState) return;
    
    Experience *exp = &rb->buffer[rb->writeIndex];
    
    // Copy game state
    memcpy(exp->gameState, gameState, INPUT_SIZE * sizeof(double));
    exp->action = action;
    exp->reward = reward;
    exp->playerIndex = playerIndex;
    exp->handOutcome = handOutcome;
    exp->gameOutcome = gameOutcome;
    
    // Update buffer indices
    rb->writeIndex = (rb->writeIndex + 1) % rb->capacity;
    if (rb->size < rb->capacity) {
        rb->size++;
    }
}

// Add random noise to network weights for diversity
void addNoiseToWeights(NeuralNetwork *nn, double noiseLevel) {
    if (!nn) return;
    
    // Add noise to input-hidden weights
    for (int i = 0; i < nn->inputSize; i++) {
        for (int j = 0; j < nn->hiddenSize; j++) {
            double noise = ((double)rand() / RAND_MAX - 0.5) * 2 * noiseLevel;
            nn->weightsInputHidden[i][j] += noise;
        }
    }
    
    // Add noise to hidden-output weights
    for (int i = 0; i < nn->hiddenSize; i++) {
        for (int j = 0; j < nn->outputSize; j++) {
            double noise = ((double)rand() / RAND_MAX - 0.5) * 2 * noiseLevel;
            nn->weightsHiddenOutput[i][j] += noise;
        }
    }
    
    // Add noise to biases
    for (int i = 0; i < nn->hiddenSize; i++) {
        double noise = ((double)rand() / RAND_MAX - 0.5) * 2 * noiseLevel;
        nn->biasHidden[i] += noise;
    }
    
    for (int i = 0; i < nn->outputSize; i++) {
        double noise = ((double)rand() / RAND_MAX - 0.5) * 2 * noiseLevel;
        nn->biasOutput[i] += noise;
    }
}

// Determine winner based on hand strength
int determineWinner(Player players[], int numPlayers, Hand *communityCards) {
    int bestPlayer = -1;
    HandScore bestScore = {0};

    for (int i = 0; i < numPlayers; i++) {
        if (players[i].status == ACTIVE) {
            Card combined[7];
            int numCards = 0;

            // Add player's cards
            Card *current = players[i].hand->first;
            while (current && numCards < 2) {
                combined[numCards++] = *current;
                current = current->next;
            }

            // Add community cards
            current = communityCards->first;
            while (current && numCards < 7) {
                combined[numCards++] = *current;
                current = current->next;
            }

            // Find best hand
            HandScore score = findBestHand(combined, numCards);

            if (bestPlayer == -1 || compareHandScores(score, bestScore) > 0) {
                bestScore = score;
                bestPlayer = i;
            }
        }
    }

    return bestPlayer;
}

// Train network from replay buffer experiences
void trainFromExperience(NeuralNetwork *nn, ReplayBuffer *rb, int batchSize) {
    if (!nn || !rb || rb->size < batchSize) return;

    double totalError = 0;

    // Random sampling from replay buffer
    for (int i = 0; i < batchSize; i++) {
        int index = rand() % rb->size;
        Experience *exp = &rb->buffer[index];

        // Forward pass
        forwardpropagate(nn, exp->gameState);

        // Create target output based on experience
        double target[OUTPUT_SIZE];
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            target[j] = nn->outputLayer[j].value;
        }

        // Q-learning update: target[action] = reward + discount * max(future_value)
        double learningRate = 0.1;
        double discount = 0.9;
        
        // Simple reward-based update
        target[exp->action] = exp->reward + discount * target[exp->action];

        // Clamp values to reasonable range
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (target[j] > 1.0) target[j] = 1.0;
            if (target[j] < 0.0) target[j] = 0.0;
        }

        // Normalize to probabilities
        double sum = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            sum += target[j];
        }
        if (sum > 0) {
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                target[j] /= sum;
            }
        }

        // Backpropagate and update
        backpropagate(nn, target);
        updateWeights(nn);

        // Track error
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            double diff = target[j] - nn->outputLayer[j].value;
            totalError += diff * diff;
        }
    }
}

// ===================================================================
// BASIC TRAINING FUNCTIONS (Original functionality)
// ===================================================================

// Generate basic training data with simple poker strategy
void generateTrainingData(double **inputs, double **outputs, int numSamples) {
    srand(time(NULL));
    
    for (int i = 0; i < numSamples; i++) {
        // Create varied game states
        double handStrength = (double)rand() / RAND_MAX;
        double potOdds = (double)rand() / RAND_MAX;
        double stackSize = (double)rand() / RAND_MAX * 10 + 0.5;
        double position = (double)rand() / RAND_MAX;
        
        // Fill input vector (basic version)
        inputs[i][0] = handStrength;
        inputs[i][1] = potOdds;
        inputs[i][2] = stackSize;
        inputs[i][3] = position;
        inputs[i][4] = (double)rand() / RAND_MAX;  // num players
        inputs[i][5] = (double)rand() / RAND_MAX;  // current bet
        inputs[i][6] = (double)rand() / RAND_MAX;  // player bet
        inputs[i][7] = (double)rand() / RAND_MAX;  // card 1
        inputs[i][8] = (double)rand() / RAND_MAX;  // card 2
        inputs[i][9] = (double)rand() / RAND_MAX;  // suited
        inputs[i][10] = (double)rand() / RAND_MAX; // cards revealed
        inputs[i][11] = (double)rand() / RAND_MAX; // aggression
        inputs[i][12] = (double)rand() / RAND_MAX; // round
        inputs[i][13] = (double)rand() / RAND_MAX; // stack committed
        inputs[i][14] = (double)rand() / RAND_MAX; // opponent aggression
        
        // Fill remaining inputs for compatibility
        for (int j = 15; j < INPUT_SIZE; j++) {
            inputs[i][j] = (double)rand() / RAND_MAX;
        }
        
        // Simple strategy for outputs
        double fold = 0.0, call = 0.0, raise = 0.0;
        
        // Basic strategy based on hand strength
        if (handStrength < 0.3) {
            fold = 0.7; call = 0.25; raise = 0.05;
        } else if (handStrength < 0.6) {
            fold = 0.3; call = 0.6; raise = 0.1;
        } else {
            fold = 0.1; call = 0.3; raise = 0.6;
        }
        
        // Add randomness
        double noise = ((double)rand() / RAND_MAX - 0.5) * 0.2;
        fold += noise; call += noise; raise += noise;
        
        // Normalize
        double total = fold + call + raise;
        if (total <= 0) total = 1.0;
        
        outputs[i][0] = fmax(0.01, fold / total);
        outputs[i][1] = fmax(0.01, call / total);
        outputs[i][2] = fmax(0.01, raise / total);
    }
}

// Basic AI training function
void trainBasicAI() {
    printf("Training AI with basic poker strategy...\n");

    int numSamples = 1000;
    double **inputs = malloc(numSamples * sizeof(double*));
    double **outputs = malloc(numSamples * sizeof(double*));

    for (int i = 0; i < numSamples; i++) {
        inputs[i] = malloc(INPUT_SIZE * sizeof(double));
        outputs[i] = malloc(OUTPUT_SIZE * sizeof(double));
    }

    generateTrainingData(inputs, outputs, numSamples);

    NeuralNetwork *nn = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    train(nn, inputs, outputs, numSamples);

    saveNetwork(nn, "poker_ai.dat");
    printf("Basic AI training complete. Saved to poker_ai.dat\n");

    // Cleanup
    for (int i = 0; i < numSamples; i++) {
        free(inputs[i]);
        free(outputs[i]);
    }
    free(inputs);
    free(outputs);
    freeNetwork(nn);
}

// ===================================================================
// SELF-PLAY TRAINING FUNCTIONS (Original implementation)
// ===================================================================

// Self-play decision with exploration
int selfPlayDecision(NeuralNetwork *nn, Player *player, Hand *communityCards, 
                    int pot, int currentBet, int numPlayers, int position, 
                    ReplayBuffer *rb, int playerIndex) {
    double input[INPUT_SIZE];

    // Use enhanced encoding if available, otherwise basic
    encodeEnhancedGameState(player, communityCards, pot, currentBet, numPlayers, position, input);

    forwardpropagate(nn, input);

    // Exploration vs exploitation
    static double epsilon = 0.3;
    epsilon = fmax(0.05, epsilon * 0.999);
    
    int decision;
    if ((double)rand() / RAND_MAX < epsilon) {
        decision = rand() % OUTPUT_SIZE;  // Random exploration
    } else {
        // Choose best action
        decision = 0;
        double bestProb = nn->outputLayer[0].value;
        for (int i = 1; i < OUTPUT_SIZE; i++) {
            if (nn->outputLayer[i].value > bestProb) {
                bestProb = nn->outputLayer[i].value;
                decision = i;
            }
        }
    }

    // Store experience
    addExperience(rb, input, decision, 0, playerIndex, 0, 0);

    return decision;
}

// Self-play prediction round
bool selfPlayPredictionRound(Player players[], int numPlayers, int *pot, int roundNum,
                            Hand* communityCards, int cardsRevealed, int startPosition, 
                            int *currentBetAmount, NeuralNetwork **networks,
                            ReplayBuffer *rb, int *handDecisions) {
    int activePlayers = 0;
    int currentPlayer = startPosition;
    int playersActed = 0;
    bool roundComplete = false;

    // Count active players
    for (int i = 0; i < numPlayers; i++) {
        if (players[i].status == ACTIVE) {
            activePlayers++;
        }
    }

    if (activePlayers <= 1) return true;

    while (!roundComplete) {
        if (players[currentPlayer].status != ACTIVE) {
            currentPlayer = (currentPlayer + 1) % numPlayers;
            continue;
        }

        if (playersActed >= activePlayers && allBetsMatched(players, numPlayers, *currentBetAmount)) {
            roundComplete = true;
            break;
        }

        int toCall = *currentBetAmount - players[currentPlayer].currentBet;

        // AI decision
        int decision = selfPlayDecision(networks[currentPlayer], &players[currentPlayer],
                                       communityCards, *pot, *currentBetAmount,
                                       activePlayers, currentPlayer, rb, currentPlayer);

        handDecisions[currentPlayer] = decision;

        // Execute decision
        switch(decision) {
            case 0: // Fold
                players[currentPlayer].status = FOLDED;
                activePlayers--;
                break;

            case 1: // Call/Check
                if (toCall > 0) {
                    int callAmount = (toCall > players[currentPlayer].credits) ?
                                   players[currentPlayer].credits : toCall;
                    players[currentPlayer].credits -= callAmount;
                    players[currentPlayer].currentBet += callAmount;
                    *pot += callAmount;
                }
                break;

            case 2: // Raise
                int raiseAmount = *currentBetAmount + (2 * BIG_BLIND);
                if (raiseAmount > players[currentPlayer].credits + players[currentPlayer].currentBet) {
                    raiseAmount = players[currentPlayer].credits + players[currentPlayer].currentBet;
                }

                int amountToAdd = raiseAmount - players[currentPlayer].currentBet;
                *currentBetAmount = raiseAmount;

                players[currentPlayer].credits -= amountToAdd;
                players[currentPlayer].currentBet = raiseAmount;
                *pot += amountToAdd;
                playersActed = 1;
                break;
        }

        if (activePlayers <= 1) return true;

        playersActed++;
        currentPlayer = (currentPlayer + 1) % numPlayers;
    }

    return false;
}

// Play single self-play hand
int playSelfPlayHand(Player players[], int numPlayers, NeuralNetwork **networks,
                     ReplayBuffer *rb, GameRecord *record) {
    int pot = 0;
    int cardsRevealed = 0;
    bool gameOver = false;
    int handDecisions[MAXPLAYERS] = {0};
    int handStartCredits[MAXPLAYERS];

    // Save starting credits
    for (int i = 0; i < numPlayers; i++) {
        handStartCredits[i] = players[i].credits;
        if (players[i].credits > 0 && players[i].status != NOT_PLAYING) {
            players[i].status = ACTIVE;
        }
    }

    // Find dealer
    int currentDealer = -1;
    for (int i = 0; i < numPlayers; i++) {
        if (players[i].dealer) {
            currentDealer = i;
            break;
        }
    }

    if (currentDealer == -1) {
        for (int i = 0; i < numPlayers; i++) {
            if (players[i].status == ACTIVE) {
                players[i].dealer = true;
                currentDealer = i;
                break;
            }
        }
    }

    // Create deck and deal
    Hand *deck = createDeck(1, 1);
    Hand *communityCards = createHand();

    // Post blinds
    int smallBlindPos = findNextActivePlayer(players, numPlayers, currentDealer, 1);
    int bigBlindPos = findNextActivePlayer(players, numPlayers, smallBlindPos, 1);

    players[smallBlindPos].credits -= SMALL_BLIND;
    players[smallBlindPos].currentBet = SMALL_BLIND;
    pot += SMALL_BLIND;

    players[bigBlindPos].credits -= BIG_BLIND;
    players[bigBlindPos].currentBet = BIG_BLIND;
    pot += BIG_BLIND;

    dealHand(players, numPlayers, deck, communityCards);

    // Betting rounds
    int currentBetAmount = BIG_BLIND;
    int startPosition = findNextActivePlayer(players, numPlayers, bigBlindPos, 1);

    // Pre-flop
    gameOver = selfPlayPredictionRound(players, numPlayers, &pot, 1, communityCards,
                                      cardsRevealed, startPosition, &currentBetAmount,
                                      networks, rb, handDecisions);

    if (!gameOver) {
        // Flop
        resetCurrentBets(players, numPlayers);
        currentBetAmount = 0;
        cardsRevealed = 3;
        startPosition = findNextActivePlayer(players, numPlayers, currentDealer, 1);
        gameOver = selfPlayPredictionRound(players, numPlayers, &pot, 2, communityCards,
                                          cardsRevealed, startPosition, &currentBetAmount,
                                          networks, rb, handDecisions);
    }

    if (!gameOver) {
        // Turn
        resetCurrentBets(players, numPlayers);
        currentBetAmount = 0;
        cardsRevealed = 4;
        gameOver = selfPlayPredictionRound(players, numPlayers, &pot, 3, communityCards,
                                          cardsRevealed, startPosition, &currentBetAmount,
                                          networks, rb, handDecisions);
    }

    if (!gameOver) {
        // River
        resetCurrentBets(players, numPlayers);
        currentBetAmount = 0;
        cardsRevealed = 5;
        gameOver = selfPlayPredictionRound(players, numPlayers, &pot, 4, communityCards,
                                          cardsRevealed, startPosition, &currentBetAmount,
                                          networks, rb, handDecisions);
    }

    // Determine winner
    int handWinner = determineWinner(players, numPlayers, communityCards);
    if (handWinner >= 0) {
        players[handWinner].credits += pot;
    }

    // Calculate rewards
    for (int i = 0; i < numPlayers; i++) {
        if (handStartCredits[i] > 0) {
            double reward = (players[i].credits - handStartCredits[i]) / 100.0;

            if (i == handWinner) {
                reward += 0.5;
            } else {
                reward -= 0.1;
            }

            // Update recent experiences
            int expCount = 0;
            for (int j = rb->size - 1; j >= 0 && expCount < 4; j--) {
                if (rb->buffer[j].playerIndex == i && rb->buffer[j].reward == 0) {
                    rb->buffer[j].reward = reward;
                    rb->buffer[j].handOutcome = (i == handWinner) ? 1 : 0;
                    expCount++;
                }
            }

            // Track decisions
            if (record->decisionCount[i] < 1000) {
                record->decisions[i][record->decisionCount[i]++] = handDecisions[i];
            }
        }
    }

    // Move dealer
    players[currentDealer].dealer = false;
    int nextDealer = findNextActivePlayer(players, numPlayers, currentDealer, 1);
    if (nextDealer >= 0) {
        players[nextDealer].dealer = true;
    }

    // Cleanup
    freeHand(deck, 1);
    freeHand(communityCards, 1);
    for (int i = 0; i < numPlayers; i++) {
        if (players[i].hand) {
            freeHand(players[i].hand, 1);
            players[i].hand = NULL;
        }
    }

    return handWinner;
}

// Play complete self-play game
GameRecord playSelfPlayGame(NeuralNetwork **networks, int numPlayers, ReplayBuffer *rb) {
    GameRecord record = {0};
    record.numPlayers = numPlayers;

    Player players[MAXPLAYERS];

    // Initialize players
    for (int i = 0; i < numPlayers; i++) {
        sprintf(players[i].name, "AI %d", i);
        players[i].credits = STARTING_CREDITS;
        players[i].status = ACTIVE;
        players[i].hand = NULL;
        players[i].dealer = (i == 0);
        players[i].currentBet = 0;
        record.decisionCount[i] = 0;
    }

    int handsPlayed = 0;
    int maxHands = 100;
    int experienceStartIndex = rb->size;

    // Play until game ends
    while (handsPlayed < maxHands) {
        int activePlayers = 0;
        int lastActive = -1;

        for (int i = 0; i < numPlayers; i++) {
            if (players[i].credits > 0) {
                activePlayers++;
                lastActive = i;
            }
        }

        if (activePlayers <= 1) {
            record.winner = lastActive;
            break;
        }

        int handWinner = playSelfPlayHand(players, numPlayers, networks, rb, &record);
        handsPlayed++;
    }

    record.totalHands = handsPlayed;

    // Record final credits
    for (int i = 0; i < numPlayers; i++) {
        record.finalCredits[i] = players[i].credits;
    }

    // Find winner if not already determined
    if (record.winner == -1) {
        int maxCredits = 0;
        for (int i = 0; i < numPlayers; i++) {
            if (players[i].credits > maxCredits) {
                maxCredits = players[i].credits;
                record.winner = i;
            }
        }
    }

    // Update rewards
    updateRewards(rb, experienceStartIndex, &record);

    return record;
}

// Update rewards based on game outcome
void updateRewards(ReplayBuffer *rb, int startIndex, GameRecord *record) {
    for (int i = startIndex; i < rb->size; i++) {
        Experience *exp = &rb->buffer[i];

        if (exp->playerIndex == record->winner) {
            exp->reward += 1.0;
            exp->gameOutcome = 1;
        } else {
            exp->reward -= 0.1;
            exp->gameOutcome = 0;
        }

        double positionBonus = record->finalCredits[exp->playerIndex] / 1000.0;
        exp->reward += positionBonus;
    }
}

// Main self-play training function
void selfPlayTraining(int numGames, int numPlayers) {
    printf("\n--- SELF-PLAY TRAINING ---\n");
    printf("Training %d AI players for %d games...\n", numPlayers, numGames);

    // Create neural networks
    NeuralNetwork **networks = malloc(numPlayers * sizeof(NeuralNetwork*));
    for (int i = 0; i < numPlayers; i++) {
        networks[i] = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
        addNoiseToWeights(networks[i], 0.1);
    }

    ReplayBuffer *rb = createReplayBuffer(100000);

    // Statistics
    int wins[MAXPLAYERS] = {0};
    double avgCredits[MAXPLAYERS] = {0};

    clock_t startTime = clock();

    // Training loop
    for (int game = 0; game < numGames; game++) {
        GameRecord record = playSelfPlayGame(networks, numPlayers, rb);

        wins[record.winner]++;
        for (int i = 0; i < numPlayers; i++) {
            avgCredits[i] = (avgCredits[i] * game + record.finalCredits[i]) / (game + 1);
        }

        // Train networks
        if (rb->size > 100 && game % 10 == 0) {
            for (int i = 0; i < numPlayers; i++) {
                trainFromExperience(networks[i], rb, 32);
            }
        }

        // Progress update
        if (game % 100 == 0 && game > 0) {
            printf("\nProgress: %d/%d games\n", game, numGames);
            for (int i = 0; i < numPlayers; i++) {
                printf("AI_%d: %.1f%% ", i, (wins[i] * 100.0) / game);
            }
            printf("\n");
        }
    }

    clock_t endTime = clock();
    double elapsed = ((double)(endTime - startTime)) / CLOCKS_PER_SEC;

    printf("\n--- TRAINING COMPLETE ---\n");
    printf("Time elapsed: %.2f seconds\n", elapsed);

    // Find best network
    int bestPlayer = 0;
    int maxWins = wins[0];
    for (int i = 1; i < numPlayers; i++) {
        if (wins[i] > maxWins) {
            maxWins = wins[i];
            bestPlayer = i;
        }
    }

    saveNetwork(networks[bestPlayer], "poker_ai_selfplay.dat");
    printf("Best network saved to poker_ai_selfplay.dat\n");

    // Cleanup
    for (int i = 0; i < numPlayers; i++) {
        freeNetwork(networks[i]);
    }
    free(networks);
    free(rb->buffer);
    free(rb);
}

// Advanced self-play training menu
void advancedSelfPlayTraining() {
    int choice;

    printf("\n--- ADVANCED SELF-PLAY TRAINING ---\n");
    printf("1. Quick training (100 games, 4 players)\n");
    printf("2. Standard training (1000 games, 4 players)\n");
    printf("3. Intensive training (5000 games, 6 players)\n");
    printf("4. Custom training\n");
    printf("Choice: ");
    scanf("%d", &choice);
    clearInputBuffer();

    int numGames, numPlayers;

    switch(choice) {
        case 1:
            numGames = 100;
            numPlayers = 4;
            break;
        case 2:
            numGames = 1000;
            numPlayers = 4;
            break;
        case 3:
            numGames = 5000;
            numPlayers = 6;
            break;
        case 4:
            printf("Number of training games: ");
            scanf("%d", &numGames);
            printf("Number of AI players (2-8): ");
            scanf("%d", &numPlayers);
            clearInputBuffer();
            break;
        default:
            return;
    }

    selfPlayTraining(numGames, numPlayers);
}

// ===================================================================
// ENHANCED TRAINING FUNCTIONS (New enhanced functionality)
// ===================================================================

// Enhanced training data generation
void generateEnhancedTrainingData(double **inputs, double **outputs, int numSamples) {
    srand(time(NULL));
    
    printf("Generating enhanced training data with %d samples...\n", numSamples);
    
    for (int i = 0; i < numSamples; i++) {
        // Create more realistic game states
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

// Enhanced basic AI training
void trainEnhancedBasicAI() {
    printf("Training enhanced AI with advanced poker strategy...\n");
    
    int numSamples = 2000;
    double **inputs = malloc(numSamples * sizeof(double*));
    double **outputs = malloc(numSamples * sizeof(double*));
    
    for (int i = 0; i < numSamples; i++) {
        inputs[i] = malloc(INPUT_SIZE * sizeof(double));
        outputs[i] = malloc(OUTPUT_SIZE * sizeof(double));
    }
    
    generateEnhancedTrainingData(inputs, outputs, numSamples);
    
    NeuralNetwork *nn = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    nn->learningRate = 0.05;
    
    printf("Training network with %d samples...\n", numSamples);
    train(nn, inputs, outputs, numSamples);
    
    saveNetwork(nn, "poker_ai_enhanced.dat");
    printf("Enhanced AI training complete. Saved to poker_ai_enhanced.dat\n");
    
    // Test network
    printf("\nTesting trained network:\n");
    for (int i = 0; i < 5; i++) {
        forwardpropagate(nn, inputs[i]);
        printf("Sample %d - Expected: F=%.2f C=%.2f R=%.2f | Got: F=%.2f C=%.2f R=%.2f\n",
               i, outputs[i][0], outputs[i][1], outputs[i][2],
               nn->outputLayer[0].value, nn->outputLayer[1].value, nn->outputLayer[2].value);
    }
    
    // Cleanup
    for (int i = 0; i < numSamples; i++) {
        free(inputs[i]);
        free(outputs[i]);
    }
    free(inputs);
    free(outputs);
    freeNetwork(nn);
}

// Enhanced self-play decision with better exploration
int enhancedSelfPlayDecision(NeuralNetwork *nn, Player *player, Hand *communityCards, 
                           int pot, int currentBet, int numPlayers, int position, 
                           ReplayBuffer *rb, int playerIndex) {
    double input[INPUT_SIZE];
    
    encodeEnhancedGameState(player, communityCards, pot, currentBet, numPlayers, position, input);
    
    forwardpropagate(nn, input);
    
    // Enhanced exploration strategy
    static double epsilon = 0.2;
    epsilon = fmax(0.05, epsilon * 0.9995);
    
    int decision;
    if ((double)rand() / RAND_MAX < epsilon) {
        decision = rand() % OUTPUT_SIZE;
    } else {
        decision = 0;
        double bestProb = nn->outputLayer[0].value;
        for (int i = 1; i < OUTPUT_SIZE; i++) {
            if (nn->outputLayer[i].value > bestProb) {
                bestProb = nn->outputLayer[i].value;
                decision = i;
            }
        }
    }
    
    addExperience(rb, input, decision, 0, playerIndex, 0, 0);
    return decision;
}

// Enhanced self-play prediction round
bool enhancedSelfPlayPredictionRound(Player players[], int numPlayers, int *pot, int roundNum,
                                    Hand* communityCards, int cardsRevealed, int startPosition, 
                                    int *currentBetAmount, NeuralNetwork **networks,
                                    ReplayBuffer *rb, int *handDecisions) {
    int activePlayers = 0;
    int currentPlayer = startPosition;
    int playersActed = 0;
    bool roundComplete = false;
    
    // Count active players
    for (int i = 0; i < numPlayers; i++) {
        if (players[i].status == ACTIVE) {
            activePlayers++;
        }
    }
    
    if (activePlayers <= 1) return true;
    
    while (!roundComplete) {
        if (players[currentPlayer].status != ACTIVE) {
            currentPlayer = (currentPlayer + 1) % numPlayers;
            continue;
        }
        
        if (playersActed >= activePlayers && allBetsMatched(players, numPlayers, *currentBetAmount)) {
            roundComplete = true;
            break;
        }
        
        int toCall = *currentBetAmount - players[currentPlayer].currentBet;
        
        // Enhanced decision making
        int decision = enhancedSelfPlayDecision(networks[currentPlayer], &players[currentPlayer],
                                              communityCards, *pot, *currentBetAmount,
                                              activePlayers, currentPlayer, rb, currentPlayer);
        
        handDecisions[currentPlayer] = decision;
        
        // Update opponent profile if function exists
        bool voluntaryAction = (decision != 0 || toCall == 0);
        updateOpponentProfile(currentPlayer, decision, voluntaryAction, 
                            players[currentPlayer].currentBet, *pot);
        
        // Execute decision
        switch(decision) {
            case 0: // Fold
                players[currentPlayer].status = FOLDED;
                activePlayers--;
                break;
                
            case 1: // Call/Check
                if (toCall > 0) {
                    int callAmount = (toCall > players[currentPlayer].credits) ?
                                   players[currentPlayer].credits : toCall;
                    players[currentPlayer].credits -= callAmount;
                    players[currentPlayer].currentBet += callAmount;
                    *pot += callAmount;
                }
                break;
                
            case 2: // Raise
                // Enhanced raise sizing
                int baseRaise = BIG_BLIND * 2;
                if (roundNum == 1) baseRaise = BIG_BLIND * 3;
                if (cardsRevealed >= 4) baseRaise = *pot / 2;
                
                int raiseAmount = *currentBetAmount + baseRaise;
                if (raiseAmount > players[currentPlayer].credits + players[currentPlayer].currentBet) {
                    raiseAmount = players[currentPlayer].credits + players[currentPlayer].currentBet;
                }
                
                int amountToAdd = raiseAmount - players[currentPlayer].currentBet;
                *currentBetAmount = raiseAmount;
                
                players[currentPlayer].credits -= amountToAdd;
                players[currentPlayer].currentBet = raiseAmount;
                *pot += amountToAdd;
                playersActed = 1;
                break;
        }
        
        if (activePlayers <= 1) return true;
        
        playersActed++;
        currentPlayer = (currentPlayer + 1) % numPlayers;
    }
    
    return false;
}

// Enhanced self-play hand
int playEnhancedSelfPlayHand(Player players[], int numPlayers, NeuralNetwork **networks,
                           ReplayBuffer *rb, GameRecord *record) {
    int pot = 0;
    int cardsRevealed = 0;
    bool gameOver = false;
    int handDecisions[MAXPLAYERS] = {0};
    int handStartCredits[MAXPLAYERS];
    
    // Save starting credits
    for (int i = 0; i < numPlayers; i++) {
        handStartCredits[i] = players[i].credits;
        if (players[i].credits > 0 && players[i].status != NOT_PLAYING) {
            players[i].status = ACTIVE;
        }
    }
    
    // Find dealer
    int currentDealer = -1;
    for (int i = 0; i < numPlayers; i++) {
        if (players[i].dealer) {
            currentDealer = i;
            break;
        }
    }
    
    if (currentDealer == -1) {
        for (int i = 0; i < numPlayers; i++) {
            if (players[i].status == ACTIVE) {
                players[i].dealer = true;
                currentDealer = i;
                break;
            }
        }
    }
    
    // Create deck and deal
    Hand *deck = createDeck(1, 1);
    Hand *communityCards = createHand();
    
    // Post blinds
    int smallBlindPos = findNextActivePlayer(players, numPlayers, currentDealer, 1);
    int bigBlindPos = findNextActivePlayer(players, numPlayers, smallBlindPos, 1);
    
    players[smallBlindPos].credits -= SMALL_BLIND;
    players[smallBlindPos].currentBet = SMALL_BLIND;
    pot += SMALL_BLIND;
    
    players[bigBlindPos].credits -= BIG_BLIND;
    players[bigBlindPos].currentBet = BIG_BLIND;
    pot += BIG_BLIND;
    
    dealHand(players, numPlayers, deck, communityCards);
    
    // Betting rounds with enhanced prediction
    int currentBetAmount = BIG_BLIND;
    int startPosition = findNextActivePlayer(players, numPlayers, bigBlindPos, 1);
    
    // Pre-flop
    gameOver = enhancedSelfPlayPredictionRound(players, numPlayers, &pot, 1, communityCards,
                                              cardsRevealed, startPosition, &currentBetAmount,
                                              networks, rb, handDecisions);
    
    if (!gameOver) {
        // Flop
        resetCurrentBets(players, numPlayers);
        currentBetAmount = 0;
        cardsRevealed = 3;
        startPosition = findNextActivePlayer(players, numPlayers, currentDealer, 1);
        gameOver = enhancedSelfPlayPredictionRound(players, numPlayers, &pot, 2, communityCards,
                                                  cardsRevealed, startPosition, &currentBetAmount,
                                                  networks, rb, handDecisions);
    }
    
    if (!gameOver) {
        // Turn
        resetCurrentBets(players, numPlayers);
        currentBetAmount = 0;
        cardsRevealed = 4;
        gameOver = enhancedSelfPlayPredictionRound(players, numPlayers, &pot, 3, communityCards,
                                                  cardsRevealed, startPosition, &currentBetAmount,
                                                  networks, rb, handDecisions);
    }
    
    if (!gameOver) {
        // River
        resetCurrentBets(players, numPlayers);
        currentBetAmount = 0;
        cardsRevealed = 5;
        gameOver = enhancedSelfPlayPredictionRound(players, numPlayers, &pot, 4, communityCards,
                                                  cardsRevealed, startPosition, &currentBetAmount,
                                                  networks, rb, handDecisions);
    }
    
    // Determine winner
    int handWinner = determineWinner(players, numPlayers, communityCards);
    if (handWinner >= 0) {
        players[handWinner].credits += pot;
    }
    
    // Enhanced reward calculation
    for (int i = 0; i < numPlayers; i++) {
        if (handStartCredits[i] > 0) {
            double reward = (players[i].credits - handStartCredits[i]) / 100.0;
            
            if (i == handWinner) {
                reward += 1.0;
                if (handDecisions[i] == 2) {
                    reward += 0.3; // Bonus for winning with aggression
                }
            } else {
                reward -= 0.2;
                if (handDecisions[i] == 0 && handStartCredits[i] - players[i].credits <= BIG_BLIND) {
                    reward += 0.1; // Reward good folds
                }
            }
            
            // Update experiences
            int expCount = 0;
            for (int j = rb->size - 1; j >= 0 && expCount < 6; j--) {
                if (rb->buffer[j].playerIndex == i && rb->buffer[j].reward == 0) {
                    rb->buffer[j].reward = reward;
                    rb->buffer[j].handOutcome = (i == handWinner) ? 1 : 0;
                    expCount++;
                }
            }
            
            // Track decisions
            if (record->decisionCount[i] < 1000) {
                record->decisions[i][record->decisionCount[i]++] = handDecisions[i];
            }
        }
    }
    
    // Move dealer
    players[currentDealer].dealer = false;
    int nextDealer = findNextActivePlayer(players, numPlayers, currentDealer, 1);
    if (nextDealer >= 0) {
        players[nextDealer].dealer = true;
    }
    
    // Cleanup
    freeHand(deck, 1);
    freeHand(communityCards, 1);
    for (int i = 0; i < numPlayers; i++) {
        if (players[i].hand) {
            freeHand(players[i].hand, 1);
            players[i].hand = NULL;
        }
    }
    
    return handWinner;
}

// Enhanced self-play game
GameRecord playEnhancedSelfPlayGame(NeuralNetwork **networks, int numPlayers, ReplayBuffer *rb) {
    GameRecord record = {0};
    record.numPlayers = numPlayers;
    
    Player players[MAXPLAYERS];
    
    // Initialize players
    for (int i = 0; i < numPlayers; i++) {
        sprintf(players[i].name, "Enhanced_AI_%d", i);
        players[i].credits = STARTING_CREDITS;
        players[i].status = ACTIVE;
        players[i].hand = NULL;
        players[i].dealer = (i == 0);
        players[i].currentBet = 0;
        record.decisionCount[i] = 0;
    }
    
    int handsPlayed = 0;
    int maxHands = 50;
    int experienceStartIndex = rb->size;
    
    // Initialize opponent profiles
    initializeOpponentProfiles(numPlayers);
    
    // Play until game ends
    while (handsPlayed < maxHands) {
        int activePlayers = 0;
        int lastActive = -1;
        
        for (int i = 0; i < numPlayers; i++) {
            if (players[i].credits > 0) {
                activePlayers++;
                lastActive = i;
            }
        }
        
        if (activePlayers <= 1) {
            record.winner = lastActive;
            break;
        }
        
        int handWinner = playEnhancedSelfPlayHand(players, numPlayers, networks, rb, &record);
        handsPlayed++;
        
        // Early termination check
        for (int i = 0; i < numPlayers; i++) {
            if (players[i].credits > STARTING_CREDITS * (numPlayers - 1)) {
                record.winner = i;
                goto game_end;
            }
        }
    }
    
    game_end:
    record.totalHands = handsPlayed;
    
    // Record final credits
    for (int i = 0; i < numPlayers; i++) {
        record.finalCredits[i] = players[i].credits;
    }
    
    // Find winner if not determined
    if (record.winner == -1) {
        int maxCredits = 0;
        for (int i = 0; i < numPlayers; i++) {
            if (players[i].credits > maxCredits) {
                maxCredits = players[i].credits;
                record.winner = i;
            }
        }
    }
    
    // Update rewards
    updateEnhancedRewards(rb, experienceStartIndex, &record);
    
    return record;
}

// Enhanced reward updating
void updateEnhancedRewards(ReplayBuffer *rb, int startIndex, GameRecord *record) {
    for (int i = startIndex; i < rb->size; i++) {
        Experience *exp = &rb->buffer[i];
        
        // Game outcome bonus
        if (exp->playerIndex == record->winner) {
            exp->reward += 2.0;
            exp->gameOutcome = 1;
        } else {
            exp->reward -= 0.3;
            exp->gameOutcome = 0;
        }
        
        // Position bonus
        double positionBonus = record->finalCredits[exp->playerIndex] / 1000.0;
        exp->reward += positionBonus;
        
        // Consistency bonus
        if (record->finalCredits[exp->playerIndex] > STARTING_CREDITS * 0.8) {
            exp->reward += 0.5;
        }
    }
}

// Enhanced self-play training
void enhancedSelfPlayTraining(int numGames, int numPlayers) {
    printf("\n--- ENHANCED SELF-PLAY TRAINING ---\n");
    printf("Training %d AI players for %d games with enhanced features...\n", numPlayers, numGames);
    
    initializeOpponentProfiles(numPlayers);
    
    // Create networks
    NeuralNetwork **networks = malloc(numPlayers * sizeof(NeuralNetwork*));
    for (int i = 0; i < numPlayers; i++) {
        networks[i] = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
        addNoiseToWeights(networks[i], 0.1);
    }
    
    ReplayBuffer *rb = createReplayBuffer(50000);
    
    // Statistics
    int wins[MAXPLAYERS] = {0};
    double avgCredits[MAXPLAYERS] = {0};
    
    clock_t startTime = clock();
    
    // Training loop
    for (int game = 0; game < numGames; game++) {
        GameRecord record = playEnhancedSelfPlayGame(networks, numPlayers, rb);
        
        wins[record.winner]++;
        for (int i = 0; i < numPlayers; i++) {
            avgCredits[i] = (avgCredits[i] * game + record.finalCredits[i]) / (game + 1);
        }
        
        // Enhanced training frequency
        if (rb->size > 200 && game % 5 == 0) {
            for (int i = 0; i < numPlayers; i++) {
                trainFromExperience(networks[i], rb, 64);
            }
        }
        
        // Progress updates
        if (game % 50 == 0 && game > 0) {
            printf("\nProgress: %d/%d games (%.1f%%)\n", game, numGames, (game * 100.0) / numGames);
            printf("Win distribution: ");
            for (int i = 0; i < numPlayers; i++) {
                printf("AI_%d: %.1f%% ", i, (wins[i] * 100.0) / game);
            }
            printf("\n");
        }
    }
    
    clock_t endTime = clock();
    double elapsed = ((double)(endTime - startTime)) / CLOCKS_PER_SEC;
    
    printf("\n--- ENHANCED TRAINING COMPLETE ---\n");
    printf("Time elapsed: %.2f seconds (%.2f games/sec)\n", elapsed, numGames / elapsed);
    
    // Find best network
    int bestPlayer = 0;
    int maxWins = wins[0];
    for (int i = 1; i < numPlayers; i++) {
        if (wins[i] > maxWins) {
            maxWins = wins[i];
            bestPlayer = i;
        }
    }
    
    printf("\nBest performing AI: AI %d with %.1f%% win rate\n",
           bestPlayer, (wins[bestPlayer] * 100.0) / numGames);
    
    saveNetwork(networks[bestPlayer], "poker_ai_enhanced_selfplay.dat");
    printf("Best enhanced network saved to poker_ai_enhanced_selfplay.dat\n");
    
    // Cleanup
    for (int i = 0; i < numPlayers; i++) {
        freeNetwork(networks[i]);
    }
    free(networks);
    free(rb->buffer);
    free(rb);
}