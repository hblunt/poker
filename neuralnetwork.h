#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "scoringsystem.h"
#include "player.h"

// Tracked features
#define INPUT_SIZE 15
// Complexity
#define HIDDEN_SIZE 20
// Call, raise, fold
#define OUTPUT_SIZE 3

#define LEARNING_RATE 0.1
#define TRAINING_EPOCHS 1000

// Choices for activation functions
#define ACTIVATION_RELU 0
#define ACTIVATION_SIGMOID 1
#define ACTIVATION_TANH 2

typedef struct {
    double value;
    double gradient;
} Neuron;

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

// Network creation and management
NeuralNetwork* createNetwork(int inputSize, int hiddenSize, int outputSize);
void freeNetwork(NeuralNetwork *nn);
void initialiseWeights(NeuralNetwork *nn);

// Forward propagation
void forwardpropagate(NeuralNetwork *nn, double *input);
double activate(double x, int activationType);
double activateDerivative(double x, int activationType);

// Training functions
void train(NeuralNetwork *nn, double **trainingInputs, double **trainingOutputs, int numSamples);
void backpropagate(NeuralNetwork *nn, double *targetOutput);
void updateWeights(NeuralNetwork *nn);

// Game integration
void encodeGameState(Player *player, Hand *communityCards, int pot, int currentBet, int numPlayers, int position, double *output);
int makeDecision(NeuralNetwork *nn, Player *player, Hand *communityCards, int pot, int currentBet, int numPlayers, int position);

// Utility functions
void saveNetwork(NeuralNetwork *nn, const char *filename);
NeuralNetwork* loadNetwork(const char *filename);
void printNetworkState(NeuralNetwork *nn);


#endif // NEURALNETWORK_H
