#ifndef AIPLAYER_H
#define AIPLAYER_H

#include "neuralnetwork.h"
#include "player.h"
#include "cards.h"

extern NeuralNetwork *aiNetwork;
void initialiseAI();
void saveAI();
void cleanAI();

int aiMakeDecision(Player *player, Hand *communityCards, int pot, int currentBet, int numPlayers, int position);
bool aiPredictionRound(Player players[],int numPlayers, int *pot, int roundNum, Hand* communityCards, int cardsRevealed, int startPosition, int *currentBetAmount);
int makeEnhancedDecision(NeuralNetwork *nn, Player *player, Hand *communityCards, int pot, int currentBet, int numPlayers, int position);

#endif
