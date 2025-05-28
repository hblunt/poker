#ifndef TRAININGGENERATOR_H
#define TRAININGGENERATOR_H

#include "neuralnetwork.h"
#include "cards.h"
#include "player.h"
#include "scoringsystem.h"

void generateTrainingData(double **inputs, double **outputs, int numSamples);
void trainBasicAI();
void selfPlayTraining(int numGames);

#endif
