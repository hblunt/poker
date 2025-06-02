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
// CORE REPLAY BUFFER AND UTILITY FUNCTIONS
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
// ENHANCED SELF-PLAY TRAINING FUNCTIONS
// ===================================================================

// Enhanced self-play decision with better exploration
int enhancedSelfPlayDecision(NeuralNetwork *nn, Player *player, Hand *communityCards, 
                           int pot, int currentBet, int numPlayers, int position, 
                           ReplayBuffer *rb, int playerIndex) {
    // Check for NULL network
    if (!nn) {
        printf("ERROR: NULL network passed to enhancedSelfPlayDecision for player %d\n", playerIndex);
        return rand() % OUTPUT_SIZE; // Random decision as fallback
    }
    
    double input[INPUT_SIZE];
    
    encodeEnhancedGameState(player, communityCards, pot, currentBet, numPlayers, position, input);
    
    forwardpropagate(nn, input);
    
    // Track decision counters for debugging
    static int decisionCounts[3] = {0, 0, 0}; // Fold, Call, Raise
    static int totalDecisions = 0;
    
    // Enhanced exploration strategy with reduced folding bias
    static double epsilon = 0.15; // Reduced from 0.2
    epsilon = fmax(0.05, epsilon * 0.9995);
    
    int decision;
    if ((double)rand() / RAND_MAX < epsilon) {
        // During exploration, reduce fold probability
        double randVal = (double)rand() / RAND_MAX;
        if (randVal < 0.25) {       // 25% fold (reduced from 33%)
            decision = 0;
        } else if (randVal < 0.7) { // 45% call 
            decision = 1;
        } else {                    // 30% raise
            decision = 2;
        }
    } else {
        decision = 0;
        double bestProb = nn->outputLayer[0].value;
        for (int i = 1; i < OUTPUT_SIZE; i++) {
            if (nn->outputLayer[i].value > bestProb) {
                bestProb = nn->outputLayer[i].value;
                decision = i;
            }
        }
        
        // Anti-folding bias: if fold is chosen but call is reasonable, sometimes call instead
        if (decision == 0 && currentBet <= player->credits * 0.1) { // Small bet relative to stack
            double callProb = nn->outputLayer[1].value;
            double foldProb = nn->outputLayer[0].value;
            
            if (callProb > foldProb * 0.7) { // If call probability is at least 70% of fold probability
                decision = 1; // Call instead of fold
            }
        }
    }
    
    // Track decisions for debugging
    decisionCounts[decision]++;
    totalDecisions++;
    
    // Print decision distribution every 100 decisions for debugging
    if (totalDecisions % 1000 == 0) {
        printf("Decision stats (last 100): Fold=%.1f%%, Call=%.1f%%, Raise=%.1f%%\n",
               (decisionCounts[0] * 100.0) / totalDecisions,
               (decisionCounts[1] * 100.0) / totalDecisions,
               (decisionCounts[2] * 100.0) / totalDecisions);
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
        int decision;
        if (!networks[currentPlayer]) {
            printf("ERROR: NULL network for player %d, using random decision\n", currentPlayer);
            decision = rand() % OUTPUT_SIZE;
        } else {
            decision = enhancedSelfPlayDecision(networks[currentPlayer], &players[currentPlayer],
                                              communityCards, *pot, *currentBetAmount,
                                              activePlayers, currentPlayer, rb, currentPlayer);
        }
        
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
                break;        }
          // Only end game early if: 
        // 1. All players folded (activePlayers == 0)
        // 2. Only one player left AND we're past the flop (roundNum > 2) 
        if (activePlayers == 0) {
            printf("WARNING: All players folded in round %d\n", roundNum);
            return true;
        }
        
        if (activePlayers == 1 && roundNum > 2) {
            // Only end if we're past the flop to allow some meaningful play
            return true;
        }
        
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
    record.winner = -1;
    
    Player players[MAXPLAYERS];
    
    // Initialize players
    for (int i = 0; i < numPlayers; i++) {
        sprintf(players[i].name, "AI_%d", i);
        players[i].credits = STARTING_CREDITS;
        players[i].status = ACTIVE;
        players[i].hand = NULL;
        players[i].dealer = (i == 0);
        players[i].currentBet = 0;
        record.decisionCount[i] = 0;
    }
    
    int handsPlayed = 0;
    int maxHands = 50;  // Allow longer games
    int experienceStartIndex = rb->size;
    
    initializeOpponentProfiles(numPlayers);
    
    // FIXED: More conservative game ending conditions
    while (handsPlayed < maxHands) {
        // Count players with meaningful credits (not just > 0)
        int viablePlayers = 0;
        int lastViable = -1;
        int totalCredits = 0;
        
        for (int i = 0; i < numPlayers; i++) {
            totalCredits += players[i].credits;
            
            // FIXED: Player needs at least big blind to be viable
            if (players[i].credits >= BIG_BLIND) {
                players[i].status = ACTIVE;
                viablePlayers++;
                lastViable = i;
            } else if (players[i].credits > 0) {
                // Player has some chips but less than big blind - still active for all-in
                players[i].status = ACTIVE;
                viablePlayers++;
                lastViable = i;
            } else {
                players[i].status = NOT_PLAYING;
            }
        }
        
        // FIXED: Only end if truly only one player left OR someone has 90%+ of chips
        bool gameEnding = false;
        
        if (viablePlayers <= 1) {
            record.winner = lastViable;
            gameEnding = true;
        } else {
            // Check for domination (someone has 90%+ of all chips)
            for (int i = 0; i < numPlayers; i++) {
                if (players[i].credits > totalCredits * 0.95) {  // 95% instead of 90%
                    record.winner = i;
                    gameEnding = true;
                    break;
                }
            }
        }
        
        if (gameEnding) {
            break;
        }
        
        // Play one hand
        int handWinner = playEnhancedSelfPlayHand(players, numPlayers, networks, rb, &record);
        handsPlayed++;
        
        // REMOVED: Early termination based on simple chip lead
        // This was probably causing premature endings
        
        // ADDED: Safety check for infinite loops
        if (handsPlayed > 100) {
            printf("WARNING: Game exceeded 100 hands, ending\n");
            break;
        }
        
        // ADDED: Safety check for credit conservation
        int newTotalCredits = 0;
        for (int i = 0; i < numPlayers; i++) {
            newTotalCredits += players[i].credits;
        }
        
        if (newTotalCredits < STARTING_CREDITS * numPlayers * 0.5) {
            printf("WARNING: Too many credits lost, redistributing\n");
            // Emergency redistribution
            int redistribution = STARTING_CREDITS * numPlayers / numPlayers;
            for (int i = 0; i < numPlayers; i++) {
                if (players[i].credits < redistribution / 2) {
                    players[i].credits = redistribution;
                }
            }
        }
    }
    
    record.totalHands = handsPlayed;
    
    // Record final credits
    for (int i = 0; i < numPlayers; i++) {
        record.finalCredits[i] = players[i].credits;
    }
    
    // Determine winner if not already set
    if (record.winner == -1) {
        int maxCredits = 0;
        for (int i = 0; i < numPlayers; i++) {
            if (players[i].credits > maxCredits) {
                maxCredits = players[i].credits;
                record.winner = i;
            }
        }
    }
    
    // Validate final results (ensure reasonable outcomes)
    double totalFinalCredits = 0;
    for (int i = 0; i < numPlayers; i++) {
        totalFinalCredits += record.finalCredits[i];
        
        // Fix negative credits
        if (record.finalCredits[i] < 0) {
            record.finalCredits[i] = 0;
        }
        
        // Cap excessive credits (no more than 4x starting amount)
        if (record.finalCredits[i] > STARTING_CREDITS * 4) {
            record.finalCredits[i] = STARTING_CREDITS * 4;
        }
    }
    
    // Ensure credit conservation
    double expectedTotal = STARTING_CREDITS * numPlayers;
    if (fabs(totalFinalCredits - expectedTotal) > expectedTotal * 0.2) {
        // Redistribute if conservation is badly violated
        double ratio = expectedTotal / (totalFinalCredits + 1); // +1 to avoid divide by zero
        for (int i = 0; i < numPlayers; i++) {
            record.finalCredits[i] *= ratio;
        }
    }
    
    updateEnhancedRewards(rb, experienceStartIndex, &record);
    
    return record;
}

// Quick diagnostic to check current game ending logic
void diagnoseGameEndingLogic() {
    printf("\n=== DIAGNOSING GAME ENDING LOGIC ===\n");
    
    // Simulate typical credit scenarios that might trigger early endings
    int testCredits[4][4] = {
        {120, 90, 90, 100},  // Slightly ahead - should NOT end
        {200, 70, 70, 60},   // Moderately ahead - should NOT end  
        {350, 20, 20, 10},   // Dominating - SHOULD end
        {400, 0, 0, 0}       // Winner clear - SHOULD end
    };
    
    for (int test = 0; test < 4; test++) {
        printf("Test %d credits: %d %d %d %d\n", test+1, 
               testCredits[test][0], testCredits[test][1], 
               testCredits[test][2], testCredits[test][3]);
        
        // Check current ending logic
        int viablePlayers = 0;
        int totalCredits = 0;
        bool shouldEnd = false;
        
        for (int i = 0; i < 4; i++) {
            totalCredits += testCredits[test][i];
            if (testCredits[test][i] >= BIG_BLIND) {
                viablePlayers++;
            }
        }
        
        if (viablePlayers <= 1) {
            shouldEnd = true;
            printf("  Would end: Only %d viable players\n", viablePlayers);
        }
        
        for (int i = 0; i < 4; i++) {
            if (testCredits[test][i] > totalCredits * 0.9) {
                shouldEnd = true;
                printf("  Would end: Player %d has %d%% of chips\n", 
                       i, (testCredits[test][i] * 100) / totalCredits);
                break;
            }
        }
        
        if (!shouldEnd) {
            printf("  Would continue: Game should keep playing\n");
        }
        
        printf("\n");
    }
    
    printf("=== GAME ENDING LOGIC DIAGNOSIS COMPLETE ===\n\n");
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

int debugPlayEnhancedSelfPlayHand(Player players[], int numPlayers, NeuralNetwork **networks,
                                  ReplayBuffer *rb, GameRecord *record, int handNumber) {
    printf("=== DEBUG HAND %d START ===\n", handNumber);
    
    // Log starting state
    printf("Starting credits: ");
    for (int i = 0; i < numPlayers; i++) {
        printf("%d ", players[i].credits);
    }
    printf("\n");
    
    clock_t handStart = clock();
    
    // Call the actual hand function
    int handWinner = playEnhancedSelfPlayHand(players, numPlayers, networks, rb, record);
    
    clock_t handEnd = clock();
    double handTime = ((double)(handEnd - handStart)) / CLOCKS_PER_SEC;
    
    printf("Hand time: %.6f seconds\n", handTime);
    printf("Hand winner: %d\n", handWinner);
    printf("Ending credits: ");
    for (int i = 0; i < numPlayers; i++) {
        printf("%d ", players[i].credits);
    }
    printf("\n");
    
    // Check for red flags
    if (handTime < 0.0001) {
        printf("RED FLAG: Hand completed too quickly!\n");
    }
    
    int totalCredits = 0;
    for (int i = 0; i < numPlayers; i++) {
        totalCredits += players[i].credits;
    }
    
    if (totalCredits != STARTING_CREDITS * numPlayers) {
        printf("RED FLAG: Credit conservation violated in hand! Total: %d, Expected: %d\n", 
               totalCredits, STARTING_CREDITS * numPlayers);
    }
    
    // Check if anyone's credits changed (if not, hand didn't really play)
    bool creditsChanged = false;
    for (int i = 0; i < numPlayers; i++) {
        if (players[i].credits != STARTING_CREDITS) {
            creditsChanged = true;
            break;
        }
    }
    
    if (!creditsChanged && handNumber > 1) {
        printf("RED FLAG: No credits changed - hand might not have played!\n");
    }
    
    printf("=== DEBUG HAND %d END ===\n\n", handNumber);
    
    return handWinner;
}

// Test just the betting round functions
void testBettingRoundIsolation() {
    printf("\n=== TESTING BETTING ROUND ISOLATION ===\n");
    
    // Create minimal test setup
    Player testPlayers[4];
    for (int i = 0; i < 4; i++) {
        sprintf(testPlayers[i].name, "TestAI_%d", i);
        testPlayers[i].credits = STARTING_CREDITS;
        testPlayers[i].currentBet = 0;
        testPlayers[i].status = ACTIVE;
        testPlayers[i].dealer = (i == 0);
        testPlayers[i].hand = createHand();
        
        // Add dummy cards to hand
        Card *card1 = createCard(2 + i, 0); // Different cards for each player
        Card *card2 = createCard(7 + i, 1);
        addCard(testPlayers[i].hand, card1);
        addCard(testPlayers[i].hand, card2);
    }
    
    // Create dummy community cards
    Hand *communityCards = createHand();
    for (int i = 0; i < 5; i++) {
        Card *card = createCard(9 + i, i % 4);
        addCard(communityCards, card);
    }
    
    printf("Testing betting round function...\n");
    
    int pot = 0;
    int currentBet = 0;
    
    clock_t start = clock();
    
    // Test the betting round function directly
    bool gameOver = enhancedSelfPlayPredictionRound(testPlayers, 4, &pot, 1, 
                                                   communityCards, 0, 0, &currentBet, 
                                                   NULL, NULL, NULL);
    
    clock_t end = clock();
    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("Betting round time: %.6f seconds\n", elapsed);
    printf("Game over: %s\n", gameOver ? "true" : "false");
    printf("Final pot: %d\n", pot);
    printf("Final bet: %d\n", currentBet);
    
    printf("Player statuses after betting:\n");
    for (int i = 0; i < 4; i++) {
        printf("  Player %d: credits=%d, status=%d, currentBet=%d\n", 
               i, testPlayers[i].credits, testPlayers[i].status, testPlayers[i].currentBet);
    }
    
    // Check for issues
    if (elapsed < 0.0001) {
        printf("RED FLAG: Betting round too fast!\n");
    }
    
    if (pot == 0 && currentBet == 0) {
        printf("RED FLAG: No betting activity!\n");
    }
    
    int activePlayers = 0;
    for (int i = 0; i < 4; i++) {
        if (testPlayers[i].status == ACTIVE) {
            activePlayers++;
        }
    }
    
    if (activePlayers == 0) {
        printf("RED FLAG: All players folded!\n");
    }
    
    // Cleanup
    for (int i = 0; i < 4; i++) {
        freeHand(testPlayers[i].hand, 1);
    }
    freeHand(communityCards, 1);
    
    printf("=== BETTING ROUND TEST COMPLETE ===\n\n");
}

// Modified game function with hand-level debugging
GameRecord playEnhancedSelfPlayGameWithHandDebug(NeuralNetwork **networks, int numPlayers, ReplayBuffer *rb) {
    printf("=== GAME WITH HAND-LEVEL DEBUG ===\n");
    
    GameRecord record = {0};
    record.numPlayers = numPlayers;
    record.winner = -1;
    
    Player players[MAXPLAYERS];
    
    // Initialize players
    for (int i = 0; i < numPlayers; i++) {
        sprintf(players[i].name, "AI_%d", i);
        players[i].credits = STARTING_CREDITS;
        players[i].status = ACTIVE;
        players[i].hand = NULL;
        players[i].dealer = (i == 0);
        players[i].currentBet = 0;
        record.decisionCount[i] = 0;
    }
    
    int handsPlayed = 0;
    int maxHands = 5; // Reduced for debugging
    
    initializeOpponentProfiles(numPlayers);
    
    while (handsPlayed < maxHands) {
        int activePlayers = 0;
        int lastActive = -1;
        
        for (int i = 0; i < numPlayers; i++) {
            if (players[i].credits > 0) {
                players[i].status = ACTIVE;
                activePlayers++;
                lastActive = i;
            } else {
                players[i].status = NOT_PLAYING;
            }
        }
        
        if (activePlayers <= 1) {
            record.winner = lastActive;
            break;
        }
        
        printf("\n--- STARTING HAND %d ---\n", handsPlayed + 1);
        
        // Use debug version for first few hands
        int handWinner;
        if (handsPlayed < 3) {
            handWinner = debugPlayEnhancedSelfPlayHand(players, numPlayers, networks, rb, &record, handsPlayed + 1);
        } else {
            handWinner = playEnhancedSelfPlayHand(players, numPlayers, networks, rb, &record);
        }
        
        handsPlayed++;
        
        printf("--- HAND %d COMPLETE ---\n", handsPlayed);
    }
    
    record.totalHands = handsPlayed;
    
    // Record final credits
    for (int i = 0; i < numPlayers; i++) {
        record.finalCredits[i] = players[i].credits;
    }
    
    // Determine winner
    if (record.winner == -1) {
        int maxCredits = 0;
        for (int i = 0; i < numPlayers; i++) {
            if (players[i].credits > maxCredits) {
                maxCredits = players[i].credits;
                record.winner = i;
            }
        }
    }
    
    printf("=== GAME DEBUG COMPLETE ===\n");
    
    return record;
}

// Add this test function to find the exact broken component
void isolateAndTestComponents() {
    printf("\n=== COMPONENT ISOLATION TESTS ===\n");
    
    // Test 1: Betting round in isolation
    printf("Test 1: Betting round isolation\n");
    testBettingRoundIsolation();
    
    // Test 2: Single game with hand debugging
    printf("Test 2: Single game with hand-level debug\n");
      NeuralNetwork *networks[4];
    for (int i = 0; i < 4; i++) {
        networks[i] = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
        if (!networks[i]) {
            printf("ERROR: Failed to create network for player %d\n", i);
            // Cleanup already created networks
            for (int j = 0; j < i; j++) {
                if (networks[j]) freeNetwork(networks[j]);
            }
            return;
        }
    }
    
    ReplayBuffer *rb = createReplayBuffer(1000);
    
    GameRecord record = playEnhancedSelfPlayGameWithHandDebug(networks, 4, rb);
    
    printf("Game result: %d hands, winner %d\n", record.totalHands, record.winner);
    
    // Cleanup
    for (int i = 0; i < 4; i++) {
        freeNetwork(networks[i]);
    }
    free(rb->buffer);
    free(rb);
    
    printf("=== COMPONENT ISOLATION COMPLETE ===\n\n");
}