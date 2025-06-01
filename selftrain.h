#ifndef SELFTRAIN_H
#define SELFTRAIN_H

#include "neuralnetwork.h"
#include "player.h"

// ===================================================================
// CORE DATA STRUCTURES
// ===================================================================

// Experience structure for replay buffer
typedef struct {
    double gameState[INPUT_SIZE];
    int action;
    double reward;
    int playerIndex;
    int handOutcome;
    int gameOutcome;
} Experience;

// Replay buffer for storing training experiences
typedef struct {
    Experience *buffer;
    int capacity;
    int size;
    int writeIndex;
} ReplayBuffer;

// Game record for tracking training progress
typedef struct {
    int numPlayers;
    int winner;
    int totalHands;
    double finalCredits[MAXPLAYERS];
    int decisions[MAXPLAYERS][1000];
    int decisionCount[MAXPLAYERS];
} GameRecord;

// ===================================================================
// CORE UTILITY FUNCTIONS
// ===================================================================

// Replay buffer management
ReplayBuffer* createReplayBuffer(int capacity);
void addExperience(ReplayBuffer *rb, double *gameState, int action, double reward, 
                  int playerIndex, int handOutcome, int gameOutcome);

// Network and training utilities
void addNoiseToWeights(NeuralNetwork *nn, double noiseLevel);
int determineWinner(Player players[], int numPlayers, Hand *communityCards);
void trainFromExperience(NeuralNetwork *nn, ReplayBuffer *rb, int batchSize);

// ===================================================================
// ENHANCED SELF-PLAY TRAINING FUNCTIONS
// ===================================================================

// Enhanced self-play decision making
int enhancedSelfPlayDecision(NeuralNetwork *nn, Player *player, Hand *communityCards, 
                           int pot, int currentBet, int numPlayers, int position, 
                           ReplayBuffer *rb, int playerIndex);

// Enhanced self-play prediction round
bool enhancedSelfPlayPredictionRound(Player players[], int numPlayers, int *pot, int roundNum,
                                    Hand* communityCards, int cardsRevealed, int startPosition, 
                                    int *currentBetAmount, NeuralNetwork **networks,
                                    ReplayBuffer *rb, int *handDecisions);

// Enhanced self-play game management
int playEnhancedSelfPlayHand(Player players[], int numPlayers, NeuralNetwork **networks,
                           ReplayBuffer *rb, GameRecord *record);
GameRecord playEnhancedSelfPlayGame(NeuralNetwork **networks, int numPlayers, ReplayBuffer *rb);

// Enhanced reward system
void updateEnhancedRewards(ReplayBuffer *rb, int startIndex, GameRecord *record);

#endif // SELFTRAIN_H