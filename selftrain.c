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



// Add these functions to selftrain.c

// Initialize self-play statistics tracking
SelfPlayStats* initializeSelfPlayStats(int maxCheckpoints) {
    SelfPlayStats *stats = malloc(sizeof(SelfPlayStats));
    if (!stats) {
        printf("Error: Could not allocate memory for self-play statistics\n");
        return NULL;
    }
    
    stats->rewardHistory = malloc(maxCheckpoints * sizeof(double));
    stats->winRateHistory = malloc(maxCheckpoints * sizeof(double));
    stats->confidenceHistory = malloc(maxCheckpoints * sizeof(double));
    stats->experienceLoss = malloc(maxCheckpoints * sizeof(double));
    stats->strategyStability = malloc(maxCheckpoints * sizeof(double));
    stats->gamesPlayed = malloc(maxCheckpoints * sizeof(int));
    
    if (!stats->rewardHistory || !stats->winRateHistory || !stats->confidenceHistory || 
        !stats->experienceLoss || !stats->strategyStability || !stats->gamesPlayed) {
        printf("Error: Could not allocate memory for self-play history\n");
        free(stats);
        return NULL;
    }
    
    stats->currentCheckpoint = 0;
    stats->maxCheckpoints = maxCheckpoints;
    stats->startTime = clock();
    
    // Open log file for self-play progress
    stats->selfPlayLogFile = fopen("selfplay_loss_log.csv", "w");
    if (stats->selfPlayLogFile) {
        fprintf(stats->selfPlayLogFile, "Checkpoint,Games,Avg_Reward,Win_Rate,Network_Confidence,Experience_Loss,Strategy_Stability,Time_Elapsed\n");
    }
    
    printf("Self-play loss tracking initialized for %d checkpoints\n", maxCheckpoints);
    return stats;
}

// Calculate MSE loss on recent experiences
double calculateExperienceLoss(NeuralNetwork *nn, ReplayBuffer *rb, int sampleSize) {
    if (!nn || !rb || rb->size < sampleSize) return 0.0;
    
    double totalLoss = 0.0;
    int samplesUsed = 0;
    
    // Sample recent experiences
    for (int i = 0; i < sampleSize && i < rb->size; i++) {
        int index = (rb->size - 1 - i) % rb->capacity;  // Get recent experiences
        Experience *exp = &rb->buffer[index];
        
        // Forward pass
        forwardpropagate(nn, exp->gameState);
        
        // Create target based on reward
        double target[OUTPUT_SIZE];
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            target[j] = nn->outputLayer[j].value;  // Current prediction
        }
        
        // Adjust target based on experience outcome
        if (exp->reward > 0) {
            target[exp->action] = fmin(1.0, target[exp->action] + 0.1);  // Reinforce good actions
        } else if (exp->reward < 0) {
            target[exp->action] = fmax(0.0, target[exp->action] - 0.1);  // Discourage bad actions
        }
        
        // Normalize to probabilities
        double sum = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) sum += target[j];
        if (sum > 0) {
            for (int j = 0; j < OUTPUT_SIZE; j++) target[j] /= sum;
        }
        
        // Calculate MSE
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            double error = target[j] - nn->outputLayer[j].value;
            totalLoss += error * error;
        }
        samplesUsed++;
    }
    
    return samplesUsed > 0 ? totalLoss / (samplesUsed * OUTPUT_SIZE) : 0.0;
}

// Calculate network confidence (entropy-based)
double calculateNetworkConfidence(NeuralNetwork *nn, ReplayBuffer *rb, int sampleSize) {
    if (!nn || !rb || rb->size < sampleSize) return 0.5;
    
    double totalConfidence = 0.0;
    int samplesUsed = 0;
    
    for (int i = 0; i < sampleSize && i < rb->size; i++) {
        int index = (rb->size - 1 - i) % rb->capacity;
        Experience *exp = &rb->buffer[index];
        
        forwardpropagate(nn, exp->gameState);
        
        // Calculate confidence as max probability
        double maxProb = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (nn->outputLayer[j].value > maxProb) {
                maxProb = nn->outputLayer[j].value;
            }
        }
        
        totalConfidence += maxProb;
        samplesUsed++;
    }
    
    return samplesUsed > 0 ? totalConfidence / samplesUsed : 0.5;
}

// Calculate strategy stability (how much network output changes)
double calculateStrategyStability(NeuralNetwork **networks, int numPlayers, 
                                 ReplayBuffer *rb, int sampleSize) {
    if (!networks || !rb || rb->size < sampleSize) return 0.0;
    
    double totalVariance = 0.0;
    int samplesUsed = 0;
    
    for (int i = 0; i < sampleSize && i < rb->size; i++) {
        int index = (rb->size - 1 - i) % rb->capacity;
        Experience *exp = &rb->buffer[index];
        
        // Get predictions from all networks for same game state
        double predictions[MAXPLAYERS][OUTPUT_SIZE];
        for (int p = 0; p < numPlayers; p++) {
            forwardpropagate(networks[p], exp->gameState);
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                predictions[p][j] = networks[p]->outputLayer[j].value;
            }
        }
        
        // Calculate variance across networks
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            double mean = 0.0;
            for (int p = 0; p < numPlayers; p++) {
                mean += predictions[p][j];
            }
            mean /= numPlayers;
            
            double variance = 0.0;
            for (int p = 0; p < numPlayers; p++) {
                double diff = predictions[p][j] - mean;
                variance += diff * diff;
            }
            variance /= numPlayers;
            totalVariance += variance;
        }
        samplesUsed++;
    }
    
    // Return stability (inverse of variance)
    double avgVariance = samplesUsed > 0 ? totalVariance / (samplesUsed * OUTPUT_SIZE) : 1.0;
    return fmax(0.0, 1.0 - avgVariance);  // Higher = more stable
}

// Update self-play statistics
void updateSelfPlayStats(SelfPlayStats *stats, NeuralNetwork **networks, int numPlayers,
                        ReplayBuffer *rb, int *wins, int totalGames, double *avgCredits) {
    if (!stats || stats->currentCheckpoint >= stats->maxCheckpoints) return;
    
    int checkpoint = stats->currentCheckpoint;
    
    // Calculate average reward from recent experiences
    double avgReward = 0.0;
    int recentExperiences = fmin(1000, rb->size);
    for (int i = 0; i < recentExperiences; i++) {
        int index = (rb->size - 1 - i) % rb->capacity;
        avgReward += rb->buffer[index].reward;
    }
    avgReward = recentExperiences > 0 ? avgReward / recentExperiences : 0.0;
    
    // Calculate current win rate (best AI)
    double bestWinRate = 0.0;
    for (int i = 0; i < numPlayers; i++) {
        double winRate = totalGames > 0 ? (double)wins[i] / totalGames : 0.0;
        if (winRate > bestWinRate) bestWinRate = winRate;
    }
    
    // Calculate other metrics
    double confidence = calculateNetworkConfidence(networks[0], rb, 500);
    double expLoss = calculateExperienceLoss(networks[0], rb, 500);
    double stability = calculateStrategyStability(networks, numPlayers, rb, 200);
    
    // Store in history
    stats->rewardHistory[checkpoint] = avgReward;
    stats->winRateHistory[checkpoint] = bestWinRate;
    stats->confidenceHistory[checkpoint] = confidence;
    stats->experienceLoss[checkpoint] = expLoss;
    stats->strategyStability[checkpoint] = stability;
    stats->gamesPlayed[checkpoint] = totalGames;
    
    // Calculate elapsed time
    clock_t currentTime = clock();
    double elapsedSeconds = ((double)(currentTime - stats->startTime)) / CLOCKS_PER_SEC;
    
    // Log to file
    if (stats->selfPlayLogFile) {
        fprintf(stats->selfPlayLogFile, "%d,%d,%.4f,%.4f,%.4f,%.6f,%.4f,%.2f\n",
                checkpoint, totalGames, avgReward, bestWinRate, confidence, 
                expLoss, stability, elapsedSeconds);
        fflush(stats->selfPlayLogFile);
    }
    
    stats->currentCheckpoint++;
}

// Display self-play training progress with loss information
void displaySelfPlayProgress(SelfPlayStats *stats, int currentGame, int totalGames,
                           int *wins, int numPlayers, double *avgCredits, int bufferSize) {
    if (!stats || stats->currentCheckpoint == 0) return;
    
    int latest = stats->currentCheckpoint - 1;
    
    printf("\n");
    printRepeatedChar('=', 60);
    printf("\n");
    printf("SELF-PLAY LEARNING PROGRESS: Game %d/%d (%.1f%%)\n", 
           currentGame, totalGames, (currentGame * 100.0) / totalGames);
    printRepeatedChar('=', 60);
    printf("\n");
    
    // Loss-like metrics
    printf("LEARNING METRICS:\n");
    printf("  Experience Loss:     %.6f", stats->experienceLoss[latest]);
    if (latest > 0) {
        double change = stats->experienceLoss[latest] - stats->experienceLoss[latest-1];
        printf(" (%s%.6f)", change < 0 ? "" : "+", change);
        if (change < -0.001) printf(" IMPROVING");
        else if (change > 0.001) printf(" DEGRADING");
        else printf(" STABLE");
    }
    printf("\n");
    
    printf("  Average Reward:      %.4f", stats->rewardHistory[latest]);
    if (latest > 0) {
        double change = stats->rewardHistory[latest] - stats->rewardHistory[latest-1];
        printf(" (%s%.4f)", change < 0 ? "" : "+", change);
        if (change > 0.01) printf(" IMPROVING");
        else if (change < -0.01) printf(" DEGRADING");
        else printf(" STABLE");
    }
    printf("\n");
    
    printf("  Network Confidence:  %.4f (%.1f%%)", 
           stats->confidenceHistory[latest], stats->confidenceHistory[latest] * 100);
    if (latest > 0) {
        double change = stats->confidenceHistory[latest] - stats->confidenceHistory[latest-1];
        if (change > 0.01) printf(" ↑ MORE DECISIVE");
        else if (change < -0.01) printf(" ↓ LESS DECISIVE");
        else printf(" → STABLE");
    }
    printf("\n");
    
    printf("  Strategy Stability:  %.4f (%.1f%%)", 
           stats->strategyStability[latest], stats->strategyStability[latest] * 100);
    if (latest > 0) {
        double change = stats->strategyStability[latest] - stats->strategyStability[latest-1];
        if (change > 0.01) printf(" ↑ CONVERGING");
        else if (change < -0.01) printf(" ↓ DIVERGING");
        else printf(" → STABLE");
    }
    printf("\n");
    
    // Win rates
    printf("\nWIN RATE PROGRESSION:\n");
    for (int i = 0; i < numPlayers; i++) {
        double winRate = (double)wins[i] / currentGame;
        printf("  AI_%d: %.1f%% (%d wins) | Credits: %.0f\n", 
               i, winRate * 100, wins[i], avgCredits[i]);
    }
    
    printf("\nExperience buffer: %d samples\n", bufferSize);
    printf("Detailed log: selfplay_loss_log.csv\n");
    printRepeatedChar('=', 60);
    printf("\n");
}

// Display final self-play summary
void displaySelfPlaySummary(SelfPlayStats *stats, int totalGames, int *wins, 
                           int numPlayers, double *avgCredits) {
    if (!stats || stats->currentCheckpoint == 0) return;
    
    printf("\n");
    printRepeatedChar('=', 70);
    printf("\n");
    printf("SELF-PLAY LEARNING COMPLETE!\n");
    printRepeatedChar('=', 70);
    printf("\n");
    
    // Learning progression analysis
    int first = 0;
    int last = stats->currentCheckpoint - 1;
    
    printf("LEARNING PROGRESSION:\n");
    printf("Initial Experience Loss:  %.6f\n", stats->experienceLoss[first]);
    printf("Final Experience Loss:    %.6f\n", stats->experienceLoss[last]);
    printf("Experience Loss Change:   %.6f", stats->experienceLoss[last] - stats->experienceLoss[first]);
    if (stats->experienceLoss[last] < stats->experienceLoss[first]) {
        printf(" (%.1f%% improvement)\n", 
               ((stats->experienceLoss[first] - stats->experienceLoss[last]) / stats->experienceLoss[first]) * 100);
    } else {
        printf(" (%.1f%% increase)\n", 
               ((stats->experienceLoss[last] - stats->experienceLoss[first]) / stats->experienceLoss[first]) * 100);
    }
    
    printf("\nREWARD PROGRESSION:\n");
    printf("Initial Avg Reward:       %.4f\n", stats->rewardHistory[first]);
    printf("Final Avg Reward:         %.4f\n", stats->rewardHistory[last]);
    printf("Reward Improvement:       %.4f\n", stats->rewardHistory[last] - stats->rewardHistory[first]);
    
    printf("\nSTRATEGY ANALYSIS:\n");
    printf("Final Network Confidence: %.1f%% (", stats->confidenceHistory[last] * 100);
    if (stats->confidenceHistory[last] > 0.7) {
        printf("Very Decisive)\n");
    } else if (stats->confidenceHistory[last] > 0.5) {
        printf("Moderately Decisive)\n");
    } else {
        printf("Uncertain/Exploratory)\n");
    }
    
    printf("Strategy Stability:       %.1f%% (", stats->strategyStability[last] * 100);
    if (stats->strategyStability[last] > 0.8) {
        printf("Highly Converged)\n");
    } else if (stats->strategyStability[last] > 0.6) {
        printf("Moderately Stable)\n");
    } else {
        printf("Still Evolving)\n");
    }
    
    // Find best performing AI
    int bestAI = 0;
    int maxWins = wins[0];
    for (int i = 1; i < numPlayers; i++) {
        if (wins[i] > maxWins) {
            maxWins = wins[i];
            bestAI = i;
        }
    }
    
    printf("\nFINAL RESULTS:\n");
    printf("Best AI: AI_%d (%.1f%% win rate)\n", bestAI, (wins[bestAI] * 100.0) / totalGames);
    printf("Total games: %d\n", totalGames);
    printf("Training log: selfplay_loss_log.csv\n");
    printRepeatedChar('=', 70);
    printf("\n");
}

// Free self-play statistics
void freeSelfPlayStats(SelfPlayStats *stats) {
    if (!stats) return;
    
    if (stats->selfPlayLogFile) {
        fclose(stats->selfPlayLogFile);
    }
    
    free(stats->rewardHistory);
    free(stats->winRateHistory);
    free(stats->confidenceHistory);
    free(stats->experienceLoss);
    free(stats->strategyStability);
    free(stats->gamesPlayed);
    free(stats);
}

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