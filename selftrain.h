#ifndef SELFTRAIN_H
#define SELFTRAIN_H

#include "neuralnetwork.h"
#include "player.h"

typedef struct {
    double gameState[INPUT_SIZE];
    int action;
    double reward;
    int playerIndex;
    int handOutcome;
    int gameOutcome;
} Experience;

typedef struct {
    Experience *buffer;
    int capacity;
    int size;
    int writeIndex;
} ReplayBuffer;

typedef struct {
    int numPlayers;
    int winner;
    int totalHands;
    double finalCredits[MAXPLAYERS];
    int decisions[MAXPLAYERS][1000];
    int decisionCount[MAXPLAYERS];
} GameRecord;

// Main training functions
void trainBasicAI();
void selfPlayTraining(int numGames, int numPlayers);
void advancedSelfPlayTraining();

// Internal functions
void generateTrainingData(double **inputs, double **outputs, int numSamples);
int playSelfPlayHand(Player players[], int numPlayers, NeuralNetwork **networks, ReplayBuffer *rb, GameRecord *record);
void updateRewards(ReplayBuffer *rb, int startIndex, GameRecord *record);
void trainFromExperience(NeuralNetwork *nn, ReplayBuffer *rb, int batchSize);
int determineWinner(Player players[], int numPlayers, Hand *communityCards);


#endif
