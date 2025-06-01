#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include "scoringsystem.h"
#include "player.h"

// UPDATED: Enhanced input size for better decision making
#define INPUT_SIZE 22           // Increased from 15
#define HIDDEN_SIZE 30          // Increased from 20 for more complexity
#define OUTPUT_SIZE 3           // Call, raise, fold (unchanged)

#define LEARNING_RATE 0.05      // Reduced for more stable learning
#define TRAINING_EPOCHS 1000

// Choices for activation functions (unchanged)
#define ACTIVATION_RELU 0
#define ACTIVATION_SIGMOID 1
#define ACTIVATION_TANH 2

// NEW: Opponent tracking structure
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
} OpponentProfile;

// Neuron structure (unchanged)
typedef struct {
    double value;
    double gradient;
} Neuron;

// Neural network structure (unchanged)
typedef struct {
    // NN architecture
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

    // Training params
    double learningRate;
    double activationFunction;
} NeuralNetwork;

// Training statistics structure
typedef struct {
    double *lossHistory;        // Loss for each epoch
    double *accuracyHistory;    // Accuracy for each epoch
    int *epochNumbers;          // Epoch numbers for plotting
    int currentEpoch;           // Current epoch number
    int maxEpochs;              // Total epochs planned
    double bestLoss;            // Best loss achieved
    int bestEpoch;              // Epoch where best loss occurred
    double initialLoss;         // Loss at start of training
    clock_t startTime;          // Training start time
    FILE *logFile;              // File for logging training progress
} TrainingStats;

// Network creation and management (unchanged)
NeuralNetwork* createNetwork(int inputSize, int hiddenSize, int outputSize);
void freeNetwork(NeuralNetwork *nn);
void initialiseWeights(NeuralNetwork *nn);

// Forward propagation (unchanged)
void forwardpropagate(NeuralNetwork *nn, double *input);
double activate(double x, int activationType);
double activateDerivative(double x, int activationType);

// Training functions (unchanged)
void train(NeuralNetwork *nn, double **trainingInputs, double **trainingOutputs, int numSamples);
void backpropagate(NeuralNetwork *nn, double *targetOutput);
void updateWeights(NeuralNetwork *nn);

// NEW: Enhanced game integration
void initializeOpponentProfiles(int numPlayers);
void updateOpponentProfile(int playerIndex, int action, bool voluntaryAction, int betAmount, int potSize);
void encodeEnhancedGameState(Player *player, Hand *communityCards, int pot, int currentBet, 
                           int numPlayers, int position, double *output);
double calculateHandPotential(Card playerCards[], Card communityCards[], int numCommunity);
double calculateBoardTexture(Card communityCards[], int numCommunity);

// UPDATED: Enhanced decision making
int makeEnhancedDecision(NeuralNetwork *nn, Player *player, Hand *communityCards, 
                        int pot, int currentBet, int numPlayers, int position);

// Utility functions (unchanged)
void saveNetwork(NeuralNetwork *nn, const char *filename);
NeuralNetwork* loadNetwork(const char *filename);
void printNetworkState(NeuralNetwork *nn);


// Training monitoring function declarations
TrainingStats* initializeTrainingStats(int maxEpochs);
double calculateLoss(NeuralNetwork *nn, double **inputs, double **targets, int numSamples);
double calculateAccuracy(NeuralNetwork *nn, double **inputs, double **targets, int numSamples);
void updateTrainingStats(TrainingStats *stats, NeuralNetwork *nn, double **inputs, 
                        double **targets, int numSamples, double learningRate);
void displayTrainingProgress(TrainingStats *stats, bool verbose);
void displayTrainingSummary(TrainingStats *stats);
void trainWithMonitoring(NeuralNetwork *nn, double **trainingInputs, double **trainingOutputs, 
                        int numSamples, int epochs);
void saveTrainingCurves(TrainingStats *stats, const char *filename);
void displayLossGraph(TrainingStats *stats);

// Enhanced training functions
void trainEnhancedBasicAIWithMonitoring();
void enhancedSelfPlayTrainingWithMonitoring(int numGames, int numPlayers);
void printRepeatedChar(char c, int count);

#endif // NEURALNETWORK_H