#include "neuralnetwork.h"
#include "cards.h"
#include "player.h"
#include "scoringsystem.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void generateTrainingData(double **inputs, double **outputs, int numSamples)
{
    srand(time(NULL));
    double gameState[INPUT_SIZE];

    for (int i = 0; i < numSamples; i++)
    {
        // Create more varied game states
        double handStrength = (double)rand() / RAND_MAX;
        gameState[0] = handStrength;

        double potOdds = (double)rand() / RAND_MAX;
        gameState[1] = potOdds;

        // Add more randomness and edge cases
        double stackSize = (double)rand() / RAND_MAX * 15 + 0.5;
        gameState[2] = stackSize;

        // Rest of the inputs...
        for (int j = 3; j < INPUT_SIZE; j++)
        {
            gameState[j] = (double)rand() / RAND_MAX;
        }

        for (int j = 0; j < INPUT_SIZE; j++)
        {
            inputs[i][j] = gameState[j];
        }

        // Improved strategy with more variety
        double fold = 0.0, call = 0.0, raise = 0.0;

        // Add some randomness to prevent overfitting
        double randomFactor = ((double)rand() / RAND_MAX - 0.5) * 0.2;

        if (handStrength < 0.2) {
            fold = 0.7 + randomFactor;
            call = 0.25;
            raise = 0.05;
        } else if (handStrength < 0.4) {
            fold = 0.4 + randomFactor;
            call = 0.5;
            raise = 0.1;
        } else if (handStrength < 0.7) {
            fold = 0.2;
            call = 0.6 + randomFactor;
            raise = 0.2;
        } else {
            fold = 0.05;
            call = 0.25;
            raise = 0.7 + randomFactor;
        }

        // Normalize and ensure valid probabilities
        double sum = fold + call + raise;
        if (sum <= 0) sum = 1.0;

        outputs[i][0] = fmax(0.01, fold / sum);
        outputs[i][1] = fmax(0.01, call / sum);
        outputs[i][2] = fmax(0.01, raise / sum);
    }
}

void trainBasicAI()
{
    printf("Training AI with a basic poker strategy...\n");

    // Create training data
    int numSamples = 1000;
    double **inputs = malloc(numSamples * sizeof(double*));
    double **outputs = malloc(numSamples * sizeof(double*));

    for (int i = 0; i < numSamples; i++)
    {
        inputs[i] = malloc(INPUT_SIZE * sizeof(double));
        outputs[i] = malloc(OUTPUT_SIZE * sizeof(double));
    }

    generateTrainingData(inputs, outputs, numSamples);

    NeuralNetwork *nn = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    train(nn, inputs, outputs, numSamples);

    saveNetwork(nn, "poker_ai.dat");
    printf("AI training complete. Data was saved.");

    // Cleanup
    for (int i = 0; i < numSamples; i++)
    {
        free(inputs[i]);
        free(outputs[i]);
    }
    free(inputs);
    free(outputs);
    freeNetwork(nn);
}

