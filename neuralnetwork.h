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
// EVOLUTIONARY TRAINING SYSTEM
// ===================================================================

// Individual AI in the population
typedef struct {
    NeuralNetwork *network;     // The neural network
    double fitness;             // Overall fitness score
    int wins;                   // Total wins this generation
    int games;                  // Total games played this generation
    double totalCredits;        // Total credits earned
    double avgCredits;          // Average credits per game
    double winRate;             // Win percentage
    int generation;             // Generation this AI was born
    int parentA, parentB;       // Parent indices (for tracking lineage)
    char strategy[50];          // Strategy description (e.g., "Aggressive", "Tight")
} Individual;

// Tournament table for random matchups
typedef struct {
    int playerIndices[4];       // Indices of AIs playing at this table
    int winner;                 // Index of winner
    double finalCredits[4];     // Final credits for each player
    bool completed;             // Whether this table finished
} TournamentTable;

// Evolutionary trainer state
typedef struct {
    Individual *population;     // Array of all AIs
    int populationSize;         // Total population (1000)
    int currentGeneration;      // Current generation number
    int maxGenerations;         // Maximum generations to run
    int gamesPerGeneration;     // Games each AI plays per generation
    
    // Evolution parameters
    double selectionRate;       // Percentage to keep (0.2 = 20%)
    double mutationRate;        // How much to mutate (0.5 = 50%)
    double crossoverRate;       // How much crossover (0.3 = 30%)
    double freshRate;           // How much fresh blood (0.2 = 20%)
    
    // Performance tracking
    double bestFitnessEver;     // Best fitness achieved
    double avgFitnessHistory[50]; // Average fitness per generation
    int bestIndividualEver;     // Index of best individual
    Individual *hallOfFame[10]; // Top 10 AIs of all time
    
    // Progress tracking
    clock_t startTime;          // Training start time
    clock_t generationStartTime; // Current generation start
    FILE *evolutionLog;         // Detailed evolution log
    
    // Tournament management
    TournamentTable *tables;    // Current tournament tables
    int numTables;              // Number of tables (populationSize / 4)
    
} EvolutionaryTrainer;

// ===================================================================
// FUNCTION DECLARATIONS
// ===================================================================

// Core evolutionary functions
EvolutionaryTrainer* createEvolutionaryTrainer(int populationSize, int maxGenerations);
void freeEvolutionaryTrainer(EvolutionaryTrainer *trainer);

// Population management
void initializePopulation(EvolutionaryTrainer *trainer);
void createDiversePopulation(EvolutionaryTrainer *trainer);

// Tournament system
void setupTournamentTables(EvolutionaryTrainer *trainer);
void runTournamentGeneration(EvolutionaryTrainer *trainer);
void runSingleTable(EvolutionaryTrainer *trainer, int tableIndex);

// Fitness evaluation
void evaluatePopulationFitness(EvolutionaryTrainer *trainer);
double calculateIndividualFitness(Individual *individual);
void rankPopulation(EvolutionaryTrainer *trainer);

// Evolution operations
void evolvePopulation(EvolutionaryTrainer *trainer);
void performSelection(EvolutionaryTrainer *trainer);
Individual* createMutatedOffspring(Individual *parent, int parentIndex, int generation);
Individual* createCrossoverOffspring(Individual *parentA, Individual *parentB, int parentAIndex, int parentBIndex, int generation);
Individual* createFreshIndividual(int generation);

// Weight manipulation for evolution
void mutateWeights(NeuralNetwork *nn, double mutationStrength);
NeuralNetwork* crossoverNetworks(NeuralNetwork *parentA, NeuralNetwork *parentB);
void addGeneticNoise(NeuralNetwork *nn, double noiseLevel);

// Progress tracking and display
void displayEvolutionProgress(EvolutionaryTrainer *trainer);
void displayGenerationSummary(EvolutionaryTrainer *trainer);
void displayFinalEvolutionResults(EvolutionaryTrainer *trainer);
void saveEvolutionCheckpoint(EvolutionaryTrainer *trainer);
void analyzePopulationDiversity(EvolutionaryTrainer *trainer);

// Main evolutionary training function
void trainEvolutionaryAI(int populationSize, int maxGenerations);

// Utility functions
void assignStrategyLabel(Individual *individual);
double calculateNetworkSimilarity(NeuralNetwork *a, NeuralNetwork *b);
void updateHallOfFame(EvolutionaryTrainer *trainer, Individual *candidate);

// ===================================================================
// LEGACY FUNCTIONS (DEPRECATED - kept for compatibility)
// ===================================================================

// Old training functions - will be removed in future versions
void trainEnhancedBasicAIWithMonitoring();  // DEPRECATED: Use trainTwoPhaseAI instead
void enhancedSelfPlayTrainingWithMonitoring(int numGames, int numPlayers);  // DEPRECATED
void generateEnhancedTrainingData(double **inputs, double **outputs, int numSamples);  // DEPRECATED

#endif // NEURALNETWORK_H