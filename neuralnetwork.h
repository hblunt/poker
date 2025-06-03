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
#define INPUT_SIZE 20           // Enhanced feature set (22 features)
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
    // Network architecture
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

// Training statistics structure for monitoring
typedef struct {
    double *lossHistory;        // Loss for each epoch
    int *epochNumbers;          // Epoch numbers for plotting
    int currentEpoch;           // Current epoch number
    int maxEpochs;              // Total epochs planned
    double bestLoss;            // Best loss achieved
    int bestEpoch;              // Epoch where best loss occurred
    double initialLoss;         // Loss at start of training
    clock_t startTime;          // Training start time
    FILE *logFile;              // File for logging training progress
    double recentLossAverage;   // Average loss over last 50 epochs
    bool overfittingDetected;   // Flag for overfitting
    int stagnationCount;        // Epochs without improvement
} TrainingStats;

// Opponent strategy types for diversified training
typedef enum {
    STRATEGY_TIGHT,           // Only plays premium hands
    STRATEGY_AGGRESSIVE,      // Frequent raises and bluffs  
    STRATEGY_CALLING_STATION, // Rarely folds, calls most bets
    STRATEGY_RANDOM,          // Unpredictable decisions
    STRATEGY_ADAPTIVE,        // Mimics target AI behavior
    STRATEGY_NEURAL           // Uses neural network (target AI only)
} OpponentStrategy;

// Target AI adaptation metrics (replaces broken loss metrics)
typedef struct {
    double *aggressionHistory;     // Target AI's raise frequency over time
    double *adaptationRate;        // How much strategy changes vs opponent types
    double *exploitationScore;     // Success rate against each opponent type
    int *strategyRotations;        // Track when opponent strategies changed
    int currentCheckpoint;
    int maxCheckpoints;
    FILE *adaptationLogFile;
    clock_t startTime;
} AdaptationStats;

// ===================================================================
// CORE NEURAL NETWORK FUNCTIONS
// ===================================================================

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

// ===================================================================
// ENHANCED GAME INTEGRATION FUNCTIONS
// ===================================================================

// Opponent modeling
void initializeOpponentProfiles(int numPlayers);
void updateOpponentProfile(int playerIndex, int action, bool voluntaryAction, int betAmount, int potSize);

// Enhanced game state encoding
void encodeEnhancedGameState(Player *player, Hand *communityCards, int pot, int currentBet, 
                           int numPlayers, int position, double *output);

// Poker-specific analysis functions
double calculateHandPotential(Card playerCards[], Card communityCards[], int numCommunity);
double calculateBoardTexture(Card communityCards[], int numCommunity);
void resetRoundAggression();
void analyzeNetworkConfidence(NeuralNetwork *nn, double **inputs, int numSamples);

// Enhanced decision making
int makeEnhancedDecision(NeuralNetwork *nn, Player *player, Hand *communityCards, 
                        int pot, int currentBet, int numPlayers, int position);

// ===================================================================
// NEW TWO-PHASE TRAINING SYSTEM
// ===================================================================

// Two-phase training functions
void trainTwoPhaseAI(int numGames, int numPlayers);
NeuralNetwork* trainMinimalBootstrap();
void pureReinforcementLearning(int numGames, int numPlayers);

// Bootstrap data generation
void generateMinimalBootstrap(double **inputs, double **outputs, int numSamples);

// ===================================================================
// NEW
// ==================================================================

// Diversified training functions
void trainDiversifiedAI(int numGames);
int makeOpponentDecision(OpponentStrategy strategy, Player *player, Hand *communityCards, 
                        int pot, int currentBet, int numPlayers, int position, NeuralNetwork *targetAI);
void rotateOpponentStrategies(OpponentStrategy strategies[], int numOpponents);
void initializeOpponentStrategies(OpponentStrategy strategies[], int numOpponents);
int getNextRotationInterval();

// New meaningful metrics
AdaptationStats* initializeAdaptationStats(int maxCheckpoints);
void updateAdaptationStats(AdaptationStats *stats, NeuralNetwork *targetAI, Player players[], 
                          int numPlayers, int *wins, int totalGames, OpponentStrategy strategies[]);
void displayAdaptationProgress(AdaptationStats *stats, int currentGame, int totalGames,
                              int *wins, int numPlayers, OpponentStrategy strategies[]);
void freeAdaptationStats(AdaptationStats *stats);

// ===================================================================
// TRAINING MONITORING FUNCTIONS
// ===================================================================

// Training statistics management
TrainingStats* initializeTrainingStats(int maxEpochs);
double calculateLoss(NeuralNetwork *nn, double **inputs, double **targets, int numSamples);
double calculateAccuracy(NeuralNetwork *nn, double **inputs, double **targets, int numSamples);
void updateTrainingStats(TrainingStats *stats, NeuralNetwork *nn, double **inputs, 
                        double **targets, int numSamples, double learningRate);

// Training progress display
void displayTrainingProgress(TrainingStats *stats, bool verbose);
void displayTrainingSummary(TrainingStats *stats, NeuralNetwork *nn, double **inputs, int numSamples);

// Core monitored training function
void trainWithMonitoring(NeuralNetwork *nn, double **trainingInputs, double **trainingOutputs, 
                        int numSamples, int epochs);

// Utility functions
void printRepeatedChar(char c, int count);

// ===================================================================
// LEGACY FUNCTIONS (DEPRECATED - kept for compatibility)
// ===================================================================

// Old training functions - will be removed in future versions
void trainEnhancedBasicAIWithMonitoring();  // DEPRECATED: Use trainTwoPhaseAI instead
void enhancedSelfPlayTrainingWithMonitoring(int numGames, int numPlayers);  // DEPRECATED
void generateEnhancedTrainingData(double **inputs, double **outputs, int numSamples);  // DEPRECATED

#endif // NEURALNETWORK_H