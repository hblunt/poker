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