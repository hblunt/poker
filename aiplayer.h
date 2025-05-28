#ifndef AIPLAYER_H
#define AIPLAYER_H

#include "player.h"
#include "cards.h"

void initialiseAI();
void saveAI();
void cleanAI();

int aiMakeDecision(Player *player, Hand *communityCards, int pot, int currentBet, int numPlayers, int position);
bool aiPredictionRound(Player players[],int numPlayers, int *pot, int roundNum, Hand* communityCards, int cardsRevealed, int startPosition, int *currentBetAmount);

#endif
