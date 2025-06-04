#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include "scoringsystem.h"
#include "player.h"

// Enhanced neural network configuration
#define INPUT_SIZE 20           // 20 features
#define HIDDEN_SIZE 30          // Optimal hidden layer size
#define OUTPUT_SIZE 3           // Fold, Call, Raise

#define LEARNING_RATE 0.05      // Conservative learning rate
#define TRAINING_EPOCHS 1000    // Standard training duration

// Activation function types
#define ACTIVATION_RELU 0
#define ACTIVATION_SIGMOID 1
#define ACTIVATION_TANH 2

// ===================================================================
// CORE DATA STRUCTURES
// ===================================================================

// Training statistics structure for monitoring neural network training progress
typedef struct {
    int currentEpoch;               // Current training epoch
    int maxEpochs;                  // Maximum number of epochs
    double *lossHistory;            // Array storing loss for each epoch
    int *epochNumbers;              // Array storing epoch numbers
    double bestLoss;                // Best (lowest) loss achieved
    int bestEpoch;                  // Epoch when best loss was achieved
    double initialLoss;             // Loss at the start of training
    clock_t startTime;              // Training start time
    double recentLossAverage;       // Average loss over recent epochs (for overfitting detection)
    bool overfittingDetected;       // Flag indicating if overfitting was detected
    int stagnationCount;            // Number of epochs without improvement
    FILE *logFile;                  // File pointer for training log output
} TrainingStats;

// Opponent tracking structure for enhanced decision making
typedef struct {
    double aggressionLevel;      // 0.0 (passive) to 1.0 (aggressive)
    double tightness;           // 0.0 (loose) to 1.0 (tight)
    double bluffFrequency;      // Estimated bluff rate
    double foldToRaise;         // How often they fold when facing a raise
    int totalActions;           // Total actions observed
    int raiseCount;             // Number of raises made
    int callCount;              // Number of calls made
    int foldCount;              // Number of folds made
    int handsPlayed;            // Total hands they've played
    int voluntaryPuts;          // Hands where they voluntarily put money in pot
    double lastAggressiveAmount; // Most recent raise/bet amount this round
    int roundNumber;            // Which round this aggression occurred
    bool aggressiveActionThisRound; // Has there been a raise/bet this round?
} OpponentProfile;

// Neuron structure
typedef struct {
    double value;
    double gradient;
} Neuron;

// Neural network structure
typedef struct {
    int inputSize;
    int hiddenSize;
    int outputSize;

    // Neurons
    Neuron *inputLayer;
    Neuron *hiddenLayer;
    Neuron *outputLayer;

    // Weights and biases
    double **weightsInputHidden;
    double **weightsHiddenOutput;
    double *biasHidden;
    double *biasOutput;

    // Training parameters
    double learningRate;
    double activationFunction;
} NeuralNetwork;



// Network creation and management
NeuralNetwork* createNetwork(int inputSize, int hiddenSize, int outputSize);
void freeNetwork(NeuralNetwork *nn);
void initialiseWeights(NeuralNetwork *nn);

// Forward propagation and activation functions
void forwardpropagate(NeuralNetwork *nn, double *input);
double activate(double x, int activationType);
double activateDerivative(double x, int activationType);

// Training functions
void backpropagate(NeuralNetwork *nn, double *targetOutput);
void updateWeights(NeuralNetwork *nn);

// Network I/O
void saveNetwork(NeuralNetwork *nn, const char *filename);
NeuralNetwork* loadNetwork(const char *filename);
void printNetworkState(NeuralNetwork *nn);

// Opponent modeling
void initializeOpponentProfiles(int numPlayers);
void updateOpponentProfile(int playerIndex, int action, bool voluntaryAction, int betAmount, int potSize);

// Enhanced game state encoding
void encodeGameState(Player *player, Hand *communityCards, int pot, int currentBet, int numPlayers, int position, double *output);

// Poker-specific analysis functions
double calculateHandPotential(Card playerCards[], Card communityCards[], int numCommunity);
double calculateBoardTexture(Card communityCards[], int numCommunity);
void resetRoundAggression();
void analyzeNetworkConfidence(NeuralNetwork *nn, double **inputs, int numSamples);

// Enhanced decision making
int makeEnhancedDecision(NeuralNetwork *nn, Player *player, Hand *communityCards, int pot, int currentBet, int numPlayers, int position);

// Two-phase training functions
void trainTwoPhaseAI(int numGames, int numPlayers);
NeuralNetwork* trainMinimalBootstrap();
void pureReinforcementLearning(int numGames, int numPlayers);

// Bootstrap data generation
void generateMinimalBootstrap(double **inputs, double **outputs, int numSamples);

// Training monitoring functions
TrainingStats* initializeTrainingStats(int maxEpochs);
void updateTrainingStats(TrainingStats *stats, NeuralNetwork *nn, double **inputs, 
                        double **targets, int numSamples, double learningRate);
double calculateLoss(NeuralNetwork *nn, double **inputs, double **targets, int numSamples);
double calculateAccuracy(NeuralNetwork *nn, double **inputs, double **targets, int numSamples);

// Utility functions
void printRepeatedChar(char c, int count);

#endif