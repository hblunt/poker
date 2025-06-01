// Updated selftrain.h - replace your existing selftrain.h with this
#ifndef SELFTRAIN_H
#define SELFTRAIN_H

#include "neuralnetwork.h"
#include "player.h"

// Experience structure for replay buffer (unchanged)
typedef struct {
    double gameState[INPUT_SIZE];
    int action;
    double reward;
    int playerIndex;
    int handOutcome;
    int gameOutcome;
} Experience;

// Replay buffer for storing training experiences (unchanged)
typedef struct {
    Experience *buffer;
    int capacity;
    int size;
    int writeIndex;
} ReplayBuffer;

// Game record for tracking training progress (unchanged)
typedef struct {
    int numPlayers;
    int winner;
    int totalHands;
    double finalCredits[MAXPLAYERS];
    int decisions[MAXPLAYERS][1000];
    int decisionCount[MAXPLAYERS];
} GameRecord;

// === ORIGINAL TRAINING FUNCTIONS ===
// Basic training functions (keep your existing ones)
void trainBasicAI();
void selfPlayTraining(int numGames, int numPlayers);
void advancedSelfPlayTraining();

// Original self-play functions
void generateTrainingData(double **inputs, double **outputs, int numSamples);
int playSelfPlayHand(Player players[], int numPlayers, NeuralNetwork **networks, ReplayBuffer *rb, GameRecord *record);
void updateRewards(ReplayBuffer *rb, int startIndex, GameRecord *record);
void trainFromExperience(NeuralNetwork *nn, ReplayBuffer *rb, int batchSize);
int determineWinner(Player players[], int numPlayers, Hand *communityCards);

// Original replay buffer functions
ReplayBuffer* createReplayBuffer(int capacity);
void addExperience(ReplayBuffer *rb, double *gameState, int action, double reward, int playerIndex, int handOutcome, int gameOutcome);

// Original self-play decision function
int selfPlayDecision(NeuralNetwork *nn, Player *player, Hand *communityCards, int pot, int currentBet, int numPlayers, int position, ReplayBuffer *rb, int playerIndex);
bool selfPlayPredictionRound(Player players[], int numPlayers, int *pot, int roundNum,Hand* communityCards, int cardsRevealed, int startPosition, int *currentBetAmount, NeuralNetwork **networks, ReplayBuffer *rb, int *handDecisions);
GameRecord playSelfPlayGame(NeuralNetwork **networks, int numPlayers, ReplayBuffer *rb);

// === NEW ENHANCED TRAINING FUNCTIONS ===
// Enhanced basic training with better strategy and monitoring
void trainEnhancedBasicAI();
void trainEnhancedBasicAIWithMonitoring();
void generateEnhancedTrainingData(double **inputs, double **outputs, int numSamples);

// Enhanced self-play training with monitoring
void enhancedSelfPlayTraining(int numGames, int numPlayers);
void enhancedSelfPlayTrainingWithMonitoring(int numGames, int numPlayers);

// Enhanced self-play game functions
GameRecord playEnhancedSelfPlayGame(NeuralNetwork **networks, int numPlayers, ReplayBuffer *rb);
int playEnhancedSelfPlayHand(Player players[], int numPlayers, NeuralNetwork **networks, ReplayBuffer *rb, GameRecord *record);

// Enhanced prediction round with better opponent modeling
bool enhancedSelfPlayPredictionRound(Player players[], int numPlayers, int *pot, int roundNum,
                                    Hand* communityCards, int cardsRevealed, int startPosition, 
                                    int *currentBetAmount, NeuralNetwork **networks,
                                    ReplayBuffer *rb, int *handDecisions);

// Enhanced decision making with better exploration
int enhancedSelfPlayDecision(NeuralNetwork *nn, Player *player, Hand *communityCards, 
                           int pot, int currentBet, int numPlayers, int position, 
                           ReplayBuffer *rb, int playerIndex);

// Enhanced reward system
void updateEnhancedRewards(ReplayBuffer *rb, int startIndex, GameRecord *record);

// Weight manipulation for diversity
void addNoiseToWeights(NeuralNetwork *nn, double noiseLevel);


// Enhanced self-play functions with monitoring
GameRecord playEnhancedSelfPlayGame(NeuralNetwork **networks, int numPlayers, ReplayBuffer *rb);
int playEnhancedSelfPlayHand(Player players[], int numPlayers, NeuralNetwork **networks,
                           ReplayBuffer *rb, GameRecord *record);
bool enhancedSelfPlayPredictionRound(Player players[], int numPlayers, int *pot, int roundNum,
                                    Hand* communityCards, int cardsRevealed, int startPosition, 
                                    int *currentBetAmount, NeuralNetwork **networks,
                                    ReplayBuffer *rb, int *handDecisions);
int enhancedSelfPlayDecision(NeuralNetwork *nn, Player *player, Hand *communityCards, 
                           int pot, int currentBet, int numPlayers, int position, 
                           ReplayBuffer *rb, int playerIndex);
void updateEnhancedRewards(ReplayBuffer *rb, int startIndex, GameRecord *record);

// Add these to main.c or create a new logs.h


#endif // SELFTRAIN_H