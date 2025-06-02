#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include "scoringsystem.h"
#include "player.h"

// Neural network configuration
#define INPUT_SIZE 20
#define HIDDEN_SIZE 30
#define OUTPUT_SIZE 3

#define LEARNING_RATE 0.05
#define TRAINING_EPOCHS 1000

// Activation function types
#define ACTIVATION_RELU 0
#define ACTIVATION_SIGMOID 1
#define ACTIVATION_TANH 2

// ===================================================================
// CORE DATA STRUCTURES
// ===================================================================

// Opponent tracking for enhanced decision making
typedef struct {
    double aggressionLevel;
    double tightness;
    double bluffFrequency;
    double foldToRaise;
    int totalActions;
    int raiseCount;
    int callCount;
    int foldCount;
    int handsPlayed;
    int voluntaryPuts;
    double lastAggressiveAmount;
    bool aggressiveActionThisRound;
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
    Neuron *inputLayer;
    Neuron *hiddenLayer;
    Neuron *outputLayer;
    double **weightsInputHidden;
    double **weightsHiddenOutput;
    double *biasHidden;
    double *biasOutput;
    double learningRate;
    double activationFunction;
} NeuralNetwork;

// Training statistics for monitoring
typedef struct {
    double *lossHistory;
    int *epochNumbers;
    int currentEpoch;
    int maxEpochs;
    double bestLoss;
    int bestEpoch;
    double initialLoss;
    clock_t startTime;
    FILE *logFile;
    double recentLossAverage;
    bool overfittingDetected;
    int stagnationCount;
} TrainingStats;

// ===================================================================
// EVOLUTIONARY TRAINING STRUCTURES
// ===================================================================

// Individual AI in population
typedef struct {
    NeuralNetwork *network;
    double fitness;
    int wins;
    int games;
    double totalCredits;
    double avgCredits;
    double winRate;
    int generation;
    int parentA, parentB;
    char strategy[50];
} Individual;

// Tournament table for matchups
typedef struct {
    int playerIndices[4];
    int winner;
    double finalCredits[4];
    bool completed;
} TournamentTable;

// Evolutionary trainer state
typedef struct {
    Individual *population;
    int populationSize;
    int currentGeneration;
    int maxGenerations;
    int gamesPerGeneration;
    double selectionRate;
    double mutationRate;
    double crossoverRate;
    double freshRate;
    double bestFitnessEver;
    double avgFitnessHistory[50];
    int bestIndividualEver;
    Individual *hallOfFame[10];
    clock_t startTime;
    clock_t generationStartTime;
    FILE *evolutionLog;
    TournamentTable *tables;
    int numTables;
} EvolutionaryTrainer;

// ===================================================================
// CORE NEURAL NETWORK FUNCTIONS
// ===================================================================

// Network creation and management
NeuralNetwork* createNetwork(int inputSize, int hiddenSize, int outputSize);
void freeNetwork(NeuralNetwork *nn);
void initialiseWeights(NeuralNetwork *nn);

// Forward propagation and activation
void forwardpropagate(NeuralNetwork *nn, double *input);
double activate(double x, int activationType);
double activateDerivative(double x, int activationType);

// Training functions
void backpropagate(NeuralNetwork *nn, double *targetOutput);
void updateWeights(NeuralNetwork *nn);

// Network I/O
void saveNetwork(NeuralNetwork *nn, const char *filename);
NeuralNetwork* loadNetwork(const char *filename);

// ===================================================================
// GAME INTEGRATION FUNCTIONS
// ===================================================================

// Opponent modeling
void initializeOpponentProfiles(int numPlayers);
void updateOpponentProfile(int playerIndex, int action, bool voluntaryAction, int betAmount, int potSize);
void resetRoundAggression();

// Game state encoding
void encodeEnhancedGameState(Player *player, Hand *communityCards, int pot, int currentBet, 
                           int numPlayers, int position, double *output);

// Poker analysis functions
double calculateHandPotential(Card playerCards[], Card communityCards[], int numCommunity);
double calculateBoardTexture(Card communityCards[], int numCommunity);

// Decision making
int makeEnhancedDecision(NeuralNetwork *nn, Player *player, Hand *communityCards, 
                        int pot, int currentBet, int numPlayers, int position);

// ===================================================================
// TWO-PHASE TRAINING SYSTEM
// ===================================================================

// Main training functions
void trainTwoPhaseAI(int numGames, int numPlayers);
NeuralNetwork* trainBootstrap();
void trainSelfPlay(int numGames, int numPlayers);

// Bootstrap data generation
void generateBootstrapData(double **inputs, double **outputs, int numSamples);

// ===================================================================
// TRAINING MONITORING
// ===================================================================

// Training statistics
TrainingStats* initTrainingStats(int maxEpochs);
double calculateLoss(NeuralNetwork *nn, double **inputs, double **targets, int numSamples);
void updateTrainingStats(TrainingStats *stats, NeuralNetwork *nn, double **inputs, 
                        double **targets, int numSamples, double learningRate);
void displayTrainingProgress(TrainingStats *stats);
void displayTrainingSummary(TrainingStats *stats);

// Core training function
void trainWithMonitoring(NeuralNetwork *nn, double **inputs, double **outputs, 
                        int numSamples, int epochs);

// ===================================================================
// EVOLUTIONARY TRAINING SYSTEM
// ===================================================================

// Core evolutionary functions
EvolutionaryTrainer* createEvolutionaryTrainer(int populationSize, int maxGenerations);
void freeEvolutionaryTrainer(EvolutionaryTrainer *trainer);

// Population management
void initializePopulation(EvolutionaryTrainer *trainer);

// Tournament system
void runTournamentGeneration(EvolutionaryTrainer *trainer);
void runSingleTable(EvolutionaryTrainer *trainer, int tableIndex);

// Fitness evaluation
void evaluatePopulationFitness(EvolutionaryTrainer *trainer);
double calculateIndividualFitness(Individual *individual);
void rankPopulation(EvolutionaryTrainer *trainer);

// Evolution operations
void evolvePopulation(EvolutionaryTrainer *trainer);
Individual* createMutatedOffspring(Individual *parent, int parentIndex, int generation);
Individual* createCrossoverOffspring(Individual *parentA, Individual *parentB, int parentAIndex, int parentBIndex, int generation);
Individual* createFreshIndividual(int generation);

// Genetic operations
void mutateWeights(NeuralNetwork *nn, double mutationStrength);
NeuralNetwork* crossoverNetworks(NeuralNetwork *parentA, NeuralNetwork *parentB);
void addGeneticNoise(NeuralNetwork *nn, double noiseLevel);

// Progress tracking
void displayGenerationSummary(EvolutionaryTrainer *trainer);
void displayEvolutionResults(EvolutionaryTrainer *trainer);
void saveEvolutionCheckpoint(EvolutionaryTrainer *trainer);

// Main evolutionary training
void trainEvolutionaryAI(int populationSize, int maxGenerations);

// Utility functions
void assignStrategyLabel(Individual *individual);
void updateHallOfFame(EvolutionaryTrainer *trainer, Individual *candidate);
void printChar(char c, int count);

// Debugging 
void diagnoseTrainingIssues(EvolutionaryTrainer *trainer);
void runSingleTableDebug(EvolutionaryTrainer *trainer, int tableIndex, bool verbose);
void debugSelfPlayGame();

#endif // NEURALNETWORK_H