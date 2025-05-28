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

    for (int i = 0; i < numSamples; i++)
    {
        // Create random game state
        double gameState[INPUT_SIZE];

        double handStrength = (double)rand() / RAND_MAX;
        gameState[0] = handStrength;

        double potOdds = (double)rand() / RAND_MAX * 0.5; // 0 to 0.5
        gameState[1] = potOdds;

        double stackSize = (double)rand() / RAND_MAX * 10 + 1; // 1 to 11
        gameState[2] = stackSize;

        double position = (double)rand() / RAND_MAX;
        gameState[3] = position;

        double numPlayers = (2 + rand() % 7) / 8.0;
        gameState[4] = numPlayers;

        double currentBet = (double)(rand() % 10) / 10.0;
        gameState[5] = currentBet;

        for (int j = 6; j < INPUT_SIZE; j++)
        {
            gameState[j] = (double)rand() / RAND_MAX;
        }

        for (int j = 0; j < INPUT_SIZE; j++)
        {
            inputs[i][j] = gameState[j];
        }

        double fold = 0.0, call = 0.0, raise = 0.0;

        // Basic strategy
        if (handStrength < 0.3) {
            // Weak hand
            if (potOdds > 0.3) {
                fold = 0.8;
                call = 0.2;
            } else {
                fold = 0.6;
                call = 0.4;
            }
        } else if (handStrength < 0.6) {
            // Medium hand
            if (potOdds > 0.2) {
                call = 0.7;
                fold = 0.2;
                raise = 0.1;
            } else {
                call = 0.5;
                raise = 0.3;
                fold = 0.2;
            }
        } else {
            // Strong hand
            if (position > 0.5) {
                // Late position
                raise = 0.7;
                call = 0.2;
                fold = 0.1;
            } else {
                // Early position
                raise = 0.5;
                call = 0.4;
                fold = 0.1;
            }
        }

        // Normalize outputs
        double sum = fold + call + raise;
        outputs[i][0] = fold / sum;
        outputs[i][1] = call / sum;
        outputs[i][2] = raise / sum;
    }
}

void trainBasicAI()
{
    printf("Training AI with a basic poker strategy...\n");

    // Create training data
    int numSamples = 1000;
    double **inputs = malloc(numSamples * sizeof(double));
    double **outputs = malloc(numSamples * sizeof(double));

    for (int i = 0; i < numSamples; i++)
    {
        inputs[i] = malloc(INPUT_SIZE * sizeof(double));
        outputs[i] = malloc(INPUT_SIZE * sizeof(double));
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

void selfPlayTraining(int numGames)
{
    printf("Starting self-play training for %d games...\n");
    // To be made more advanced later
    trainBasicAI();
}
